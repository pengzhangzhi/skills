#!/usr/bin/env python3
"""R0 unit tests for train_watch.py — the babysit training-health detector.

Stdlib-only, CPU-only. Exercises the pure metric functions on synthetic metrics.csv fixtures and the
SLURM-output parsers on captured strings, so the whole health verdict is tested without a live
cluster. Run: python3 -m pytest tests/test_train_watch.py  (or  python3 tests/test_train_watch.py).
"""
from __future__ import annotations

import csv
import importlib.util
import math
import os
import pathlib
import tempfile
import time
import unittest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "train_watch.py"
_spec = importlib.util.spec_from_file_location("train_watch", _SCRIPT)
tw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tw)

# Realistic column subset (order mirrors a real production metrics.csv).
COLS = ["step", "epoch", "train/loss", "train/ce", "train/grad_norm",
        "train/nonfinite_steps", "train/nonfinite_grad_steps", "train/nonfinite_param_steps",
        "lr-AdamW/pg1", "lr-AdamW/pg2", "optim/lr", "perf/sec_per_step"]


def write_version(run_dir, vnum, steps, loss_fn, *, cadence=50, grad=0, gsteps=0, psteps=0,
                  two_rows=True, sec_per_step=0.9, mtime=None):
    """Write one csv/version_<vnum>/metrics.csv. Emits the real TWO-rows-per-step pattern:
    a bare lr row (train/loss blank) then a full-metrics row."""
    vdir = os.path.join(run_dir, "csv", f"version_{vnum}")
    os.makedirs(vdir, exist_ok=True)
    p = os.path.join(vdir, "metrics.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for st in steps:
            lv = loss_fn(st)
            if two_rows:  # bare lr-only row first (no train/loss) -> detector must skip it
                bare = {c: "" for c in COLS}
                bare.update({"step": st, "lr-AdamW/pg1": 1e-4, "lr-AdamW/pg2": 1e-4})
                w.writerow([bare[c] for c in COLS])
            full = {c: "" for c in COLS}
            full.update({"step": st, "epoch": 0, "train/loss": lv, "train/ce": lv,
                         "train/grad_norm": grad, "train/nonfinite_steps": gsteps + psteps,
                         "train/nonfinite_grad_steps": gsteps, "train/nonfinite_param_steps": psteps,
                         "optim/lr": 1e-4, "perf/sec_per_step": sec_per_step})
            w.writerow([full[c] for c in COLS])
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def write_config(run_dir, max_steps=300000, stable_frac=0.8, warmup=2000, final_frac=0.1):
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w") as f:
        f.write("run:\n  name: demo\n  out_dir: /tmp/runs\n")
        f.write(f"train:\n  max_steps: {max_steps}\n  batch_size: 4\n")
        f.write(f"sched:\n  warmup: {warmup}\n  stable_frac: {stable_frac}\n  final_frac: {final_frac}\n")
        f.write("optim:\n  lr: 1.0e-4\n")


class Cadence(unittest.TestCase):
    def test_measure_cadence_50(self):
        self.assertEqual(tw.measure_cadence([49, 99, 149, 199, 249]), 50)

    def test_measure_cadence_handles_resume_overlap(self):
        # a version boundary can regress the step; median must still recover the cadence
        self.assertEqual(tw.measure_cadence([49, 99, 149, 120, 170, 220]), 50)


class ReadRun(unittest.TestCase):
    def test_two_rows_per_step_deduped(self):
        with tempfile.TemporaryDirectory() as d:
            rd = os.path.join(d, "run"); os.makedirs(rd)
            write_config(rd)
            steps = list(range(49, 49 + 50 * 20, 50))  # 20 steps
            write_version(rd, 0, steps, lambda s: 2.0 - s * 1e-5, mtime=time.time())
            rec = tw.read_run(rd)
            # 40 physical rows but only 20 loss-bearing steps
            self.assertEqual(len(rec["loss_series"]), 20)
            self.assertEqual(rec["cadence"], 50)
            self.assertEqual(rec["max_steps"], 300000)

    def test_healthy_run_ok(self):
        with tempfile.TemporaryDirectory() as d:
            rd = os.path.join(d, "run"); os.makedirs(rd)
            write_config(rd)
            steps = list(range(49, 49 + 50 * 400, 50))
            write_version(rd, 0, steps, lambda s: max(0.2, 3.0 - s * 2e-5), grad=0.8, mtime=time.time())
            rec = tw.read_run(rd)
            self.assertEqual(rec["severity"], 0, rec["flags"])
            self.assertEqual(rec["nonfinite"], 0)

    def test_nan_midseries_is_critical(self):
        with tempfile.TemporaryDirectory() as d:
            rd = os.path.join(d, "run"); os.makedirs(rd)
            write_config(rd)
            steps = list(range(49, 49 + 50 * 100, 50))
            nan_at = steps[50]
            write_version(rd, 0, steps, lambda s: (float("nan") if s >= nan_at else 2.0), mtime=time.time())
            rec = tw.read_run(rd)
            self.assertGreaterEqual(rec["nonfinite"], 1)
            self.assertEqual(rec["first_nonfinite_step"], nan_at)
            self.assertEqual(rec["severity"], 2)

    def test_param_dead_counter_is_critical(self):
        with tempfile.TemporaryDirectory() as d:
            rd = os.path.join(d, "run"); os.makedirs(rd)
            write_config(rd)
            steps = list(range(49, 49 + 50 * 30, 50))
            write_version(rd, 0, steps, lambda s: 2.0, psteps=3, mtime=time.time())
            rec = tw.read_run(rd)
            self.assertEqual(rec["severity"], 2)
            self.assertTrue(any("PARAM" in f[2].upper() or "DEAD" in f[2].upper() for f in rec["flags"]))

    def test_grad_nan_only_is_warn_selfheal(self):
        with tempfile.TemporaryDirectory() as d:
            rd = os.path.join(d, "run"); os.makedirs(rd)
            write_config(rd)
            steps = list(range(49, 49 + 50 * 30, 50))
            write_version(rd, 0, steps, lambda s: 2.0, gsteps=2, mtime=time.time())  # loss finite, params live
            rec = tw.read_run(rd)
            self.assertEqual(rec["severity"], 1)


class Signals(unittest.TestCase):
    def test_divergence_detected_on_tail_jump(self):
        series = [(i * 50, 2.0) for i in range(200)] + [(200 * 50, 4.5)]
        hit, at = tw.detect_divergence(series)
        self.assertTrue(hit)
        self.assertEqual(at, 200 * 50)

    def test_no_divergence_on_smooth_decrease(self):
        series = [(i * 50, 3.0 - i * 1e-3) for i in range(300)]
        hit, _ = tw.detect_divergence(series)
        self.assertFalse(hit)

    def test_plateau_flat_before_decay(self):
        series = [(i * 50, 2.0) for i in range(600)]  # dead flat, 30k steps
        self.assertTrue(tw.detect_plateau(series, cadence=50, cur_step=30000, decay_start=240000))

    def test_no_plateau_when_still_decreasing(self):
        series = [(i * 50, 3.0 - i * 2e-3) for i in range(600)]
        self.assertFalse(tw.detect_plateau(series, cadence=50, cur_step=30000, decay_start=240000))

    def test_no_plateau_in_decay_phase(self):
        series = [(i * 50, 2.0) for i in range(600)]
        # cur_step past decay_start -> flatness is expected, not a plateau flag
        self.assertFalse(tw.detect_plateau(series, cadence=50, cur_step=260000, decay_start=240000))

    def test_throughput_drop(self):
        versions = [
            {"step_lo": 0, "step_hi": 17000}, {"step_lo": 17000, "step_hi": 34000},
            {"step_lo": 34000, "step_hi": 44000},  # last COMPLETED span 10000 < 0.7*17000
            {"step_lo": 44000, "step_hi": 44500},  # in-flight, excluded
        ]
        hit, _ = tw.throughput_drop(versions)
        self.assertTrue(hit)

    def test_no_throughput_drop_steady(self):
        versions = [{"step_lo": i * 17000, "step_hi": (i + 1) * 17000} for i in range(4)] + \
                   [{"step_lo": 68000, "step_hi": 68500}]
        hit, _ = tw.throughput_drop(versions)
        self.assertFalse(hit)


class Discovery(unittest.TestCase):
    def test_discovers_recent_ignores_stale(self):
        with tempfile.TemporaryDirectory() as d:
            now = time.time()
            for name, age_min in [("fresh", 30), ("stale", 60 * 30)]:  # stale = 30h old
                rd = os.path.join(d, name); os.makedirs(rd)
                write_config(rd)
                write_version(rd, 0, [49, 99, 149], lambda s: 2.0, mtime=now - age_min * 60)
            found = tw.discover_runs(d, mmin=720, now=now)
            names = {os.path.basename(p) for p in found}
            self.assertIn("fresh", names)
            self.assertNotIn("stale", names)


class Config(unittest.TestCase):
    def test_read_config_scalars(self):
        with tempfile.TemporaryDirectory() as d:
            write_config(d, max_steps=250000, stable_frac=0.8)
            self.assertEqual(tw.read_config_scalar(d, "train.max_steps"), 250000)
            self.assertAlmostEqual(tw.read_config_scalar(d, "sched.stable_frac"), 0.8)
            self.assertEqual(tw.read_config_scalar(d, "train.max_steps", default=-1), 250000)

    def test_missing_key_returns_default(self):
        with tempfile.TemporaryDirectory() as d:
            write_config(d)
            self.assertIsNone(tw.read_config_scalar(d, "nope.nothere"))


class Slurm(unittest.TestCase):
    def test_sacct_cmd_uses_name_flag_not_dash_n(self):
        # regression guard for a real bug: sacct -n == --noheader (not --name)
        cmd = tw.sacct_cmd("train-jobA", user="me", since="now-24hours")
        self.assertTrue(any(str(a).startswith("--name=") for a in cmd),
                        f"sacct must use --name=, got {cmd}")
        self.assertNotIn("-n", cmd)

    def test_training_names_selects_chains_drops_eval_singletons(self):
        # afterany training chain = many same-named pending; eval jobs = singletons
        counts = {"train-jobA": 150, "train-jobB": 6,
                  "eval-gen-run_a": 1, "eval-report-x": 1}
        names = tw.training_job_names(counts, running_names=set(), cached_names=set())
        self.assertEqual(names, {"train-jobA", "train-jobB"})

    def test_training_names_unions_running_and_cached(self):
        names = tw.training_job_names({"fe-gen-x": 1}, running_names={"my-run"}, cached_names={"old-run"})
        self.assertEqual(names, {"my-run", "old-run"})

    def test_parse_squeue(self):
        out = "311|train-jobA|RUNNING|1:02:03|2:57:00|node5\n" \
              "312|train-jobA|PENDING|0:00|4:00:00|\n"
        jobs = tw.parse_squeue(out)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["state"], "RUNNING")
        self.assertEqual(jobs[0]["nodes"], "node5")
        self.assertEqual(jobs[1]["state"], "PENDING")

    def test_job_to_run_reads_config_token(self):
        with tempfile.TemporaryDirectory() as logs:
            with open(os.path.join(logs, "train_999.out"), "w") as f:
                f.write("[ablation.sbatch] job=999 host=n1 nodes=8 world=64 config=run_a\n")
            self.assertEqual(tw.job_to_run("999", logs), "run_a")

    def test_all_pending_is_not_stalled(self):
        # a run whose csv is quiet but NO segment on-node and jobs pending == healthy
        rec = {"age_s": 3600, "step": 25000, "max_steps": 300000, "flags": [], "severity": 0}
        tw.classify_slurm(rec, on_node=False, any_pending=True)
        self.assertEqual(rec["severity"], 0)
        self.assertFalse(any("STALL" in f[2].upper() for f in rec["flags"]))

    def test_on_node_quiet_is_stalled(self):
        rec = {"age_s": 3600, "step": 25000, "max_steps": 300000, "flags": [], "severity": 0}
        tw.classify_slurm(rec, on_node=True, any_pending=True)
        self.assertEqual(rec["severity"], 2)
        self.assertTrue(any("STALL" in f[2].upper() for f in rec["flags"]))

    def test_chain_exhausted_is_critical(self):
        rec = {"age_s": 3600, "step": 25000, "max_steps": 300000, "flags": [], "severity": 0}
        tw.classify_slurm(rec, on_node=False, any_pending=False)
        self.assertEqual(rec["severity"], 2)
        self.assertTrue(any("EXHAUST" in f[2].upper() or "CHAIN" in f[2].upper() for f in rec["flags"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
