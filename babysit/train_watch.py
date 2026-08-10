#!/usr/bin/env python3
"""Run-agnostic training-health detector for the `babysit` skill.

No hardcoded run list, job name, or max_steps. Answers, per run, the questions that matter while
training runs unattended for days — IS IT ALIVE / HEALTHY / MOVING / WILL IT FINISH — and adds
generative-model signals (divergence, plateau, throughput drop) plus a since-last-check delta so a
watcher can produce a real "caught up" digest.

  python3 train_watch.py                 # auto-discover active runs, human report
  python3 train_watch.py --json           # machine-readable (the skill consumes this)
  python3 train_watch.py --runs a b        # explicit run names or abs dirs

Assumes a PyTorch-Lightning-style layout: each run is a directory holding `csv/version_*/metrics.csv`
(one version_* per resume/segment) and optionally a `resolved_config.yaml`. Configure the roots with
$BABYSIT_RUNS_ROOT / $BABYSIT_LOGS_DIR (or the --runs-root / --logs-dir flags). Adapt the column names
in read_run() if your logger differs.

Stdlib only, CPU-only, READ-ONLY (never touches SLURM state). Design notes (hard-won on a real
multi-day SLURM training study — the traps a naive reader falls into):
  * A CSVLogger may write TWO physical rows per logged step (a bare lr row then a full row) — group by
    the `step` column and use loss-bearing rows only.
  * Don't rely on a dedicated nan flag column: scan `train/loss` for non-finite over EVERY row of
    EVERY version (a NaN can appear then "train" thousands more steps looking fine), and read any
    guard counters (train/nonfinite_{steps,grad,param}_steps) if present. Under FSDP a grad-NaN that a
    guard zeros can self-heal (WARN); a param-NaN or a non-finite loss is fatal (CRITICAL).
  * Loss aliases: some loops log `train/loss` == `loss/total`; pick one canonical column. grad_norm is
    optional (not always logged).
  * version_N == one resume segment; segments can overlap in step on resume (restore lags the last
    flush) — measure cadence from file order, dedup by step, throughput from median COMPLETED spans.
  * On a chained SLURM job, all-PENDING / zero-RUNNING is HEALTHY (between segments / preempted +
    requeued). Never flag it as dead — only 0-running AND 0-pending below max_steps is a real stop.
  * Use `sacct --name=<job>`, NOT `-n` (that's --noheader and silently makes the name positional, so
    the query reports clean through every failure).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import time

# Configure your layout here or via env / CLI flags (no site-specific paths are baked in).
RUNS_ROOT = os.environ.get("BABYSIT_RUNS_ROOT", os.path.abspath("runs"))
LOGS_DIR = os.environ.get("BABYSIT_LOGS_DIR", os.path.abspath("logs"))
STATE_PATH = os.path.join(LOGS_DIR, "train_watch_state.json")

SEGMENT_HOURS = float(os.environ.get("BABYSIT_SEGMENT_HOURS", "4.0"))  # walltime of one SLURM segment
DISCOVER_MMIN = 720            # 12h; a short window misses the healthy between-segment gap
STALL_AGE_MIN = 15             # on-node + csv quiet >15min == wedged (typical flush cadence ~90s)
BURN_SECONDS = 300            # segment dying <5min == burned
DIVERGE_K = 6
PLATEAU_WIN_STEPS = 20000
THRUPUT_DROP = 0.70
CADENCE_DEFAULT = 50
LEVEL = {0: "OK", 1: "WARN", 2: "CRIT"}


# ----------------------------------------------------------------------------- small helpers
def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _coerce(s):
    s = s.strip().strip('"').strip("'")
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def _lstsq_slope(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


# ----------------------------------------------------------------------------- config reading
def _yaml_scalars(path):
    """Minimal indent-scoped scalar extractor for machine-emitted (OmegaConf) YAML — enough to read
    dotted scalar keys like train.max_steps / sched.stable_frac without a yaml dependency. Ignores
    list items; only simple `key: value` and `key:` mapping parents are tracked."""
    out, stack = {}, []  # stack: list of (indent, key)
    with open(path) as fh:
        lines = fh.readlines()
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        indent = len(line) - len(line.lstrip(" "))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        key, val = key.strip(), val.split(" #")[0].strip()
        dotted = ".".join([k for _, k in stack] + [key])
        if val == "":
            stack.append((indent, key))
        else:
            out[dotted] = val
    return out


def read_config_scalar(run_dir, dotted_key, default=None):
    cfg = run_dir if run_dir.endswith((".yaml", ".yml")) else os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.exists(cfg):
        return default
    scalars = _yaml_scalars(cfg)
    return _coerce(scalars[dotted_key]) if dotted_key in scalars else default


# ----------------------------------------------------------------------------- metric signals
def measure_cadence(steps):
    """Steps between consecutive logged steps. Uses consecutive diffs in the GIVEN (file) order and
    keeps only positive ones, so a resume step-regression at a version boundary doesn't corrupt it."""
    steps = [int(s) for s in steps]
    diffs = sorted(b - a for a, b in zip(steps, steps[1:]) if b > a)
    return diffs[len(diffs) // 2] if diffs else None


def detect_divergence(series, k=DIVERGE_K, win=30):
    """Upward tail jump: last loss > mean(trailing window) + k*std. Only fires on a jump UP."""
    vals = [l for _, l in series if math.isfinite(l)]
    if len(vals) < win + 2:
        return (False, None)
    recent, base = vals[-1], vals[-(win + 1):-1]
    mu = sum(base) / len(base)
    sd = math.sqrt(sum((v - mu) ** 2 for v in base) / len(base)) or (abs(mu) * 1e-3 + 1e-9)
    return (True, series[-1][0]) if recent - mu > k * sd else (False, None)


def detect_plateau(series, cadence, cur_step, decay_start, win_steps=PLATEAU_WIN_STEPS):
    """Near-zero loss slope over a long trailing window while still in the stable phase. WARN only —
    a plateau is often real convergence (the stage6b-plateau-by-10k lesson), never auto-'fixed'."""
    if decay_start and cur_step >= decay_start:
        return False   # loss is expected to move in the decay phase; flatness there isn't a plateau
    finite = [(s, l) for s, l in series if math.isfinite(l)]
    win_pts = max(50, win_steps // max(cadence or CADENCE_DEFAULT, 1))
    if len(finite) < win_pts:
        return False
    w = finite[-win_pts:]
    xs, ys = [s for s, _ in w], [l for _, l in w]
    total_change = _lstsq_slope(xs, ys) * (xs[-1] - xs[0])
    return abs(total_change) < 0.005 * abs(sum(ys) / len(ys)) + 1e-9


def throughput_drop(versions):
    """Last COMPLETED version span < THRUPUT_DROP * median of earlier completed spans."""
    completed = [v["step_hi"] - v["step_lo"] for v in versions[:-1] if v["step_hi"] > v["step_lo"]]
    if len(completed) < 2:
        return (False, None)
    last, med = completed[-1], _median(completed[:-1])
    return (True, last / med) if med and last < THRUPUT_DROP * med else (False, None)


# ----------------------------------------------------------------------------- per-run health record
def read_run(run_dir, now=None, cadence_default=CADENCE_DEFAULT):
    """Fold every csv/version_*/metrics.csv of one run into a single metric-only health record.
    SLURM-dependent flags (stall, chain-exhausted) are added later by classify_slurm()."""
    now = now or time.time()
    vpaths = sorted(glob.glob(os.path.join(run_dir, "csv", "version_*", "metrics.csv")),
                    key=lambda p: int(re.search(r"version_(\d+)", p).group(1)))
    if not vpaths:
        return None

    loss_series, steps_in_order, versions = [], [], []
    guard = {"grad": 0.0, "param": 0.0, "steps": 0.0}
    newest_mtime = 0.0
    for vp in vpaths:
        pts = []
        with open(vp) as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            st, ls = _f(r.get("step")), _f(r.get("train/loss"))
            for col, key in (("train/nonfinite_grad_steps", "grad"),
                             ("train/nonfinite_param_steps", "param"),
                             ("train/nonfinite_steps", "steps")):
                v = _f(r.get(col))
                if v is not None and v > guard[key]:
                    guard[key] = v
            if st is None or ls is None:   # skip the bare lr-only row (blank train/loss)
                continue
            pts.append((st, ls))
            steps_in_order.append(st)
        if not pts:
            continue
        mt = os.path.getmtime(vp)
        newest_mtime = max(newest_mtime, mt)
        versions.append({"version": int(re.search(r"version_(\d+)", vp).group(1)), "mtime": mt,
                         "n": len(pts), "step_lo": pts[0][0], "step_hi": pts[-1][0]})
        loss_series.extend(pts)
    if not loss_series:
        return None

    loss_series.sort(key=lambda x: x[0])
    nf = [s for s, l in loss_series if not math.isfinite(l)]
    cadence = measure_cadence(steps_in_order) or cadence_default
    max_steps = read_config_scalar(run_dir, "train.max_steps")
    stable_frac = read_config_scalar(run_dir, "sched.stable_frac")
    warmup = read_config_scalar(run_dir, "sched.warmup") or 0
    decay_start = max(int(warmup), int(stable_frac * max_steps)) if (stable_frac and max_steps) else None

    step = loss_series[-1][0]
    tail = [l for _, l in loss_series[-20:] if math.isfinite(l)]
    prev = [l for _, l in loss_series[-120:-100] if math.isfinite(l)]
    tail_mean = sum(tail) / len(tail) if tail else float("nan")
    trend = "→"
    if prev and tail:
        d = tail_mean - sum(prev) / len(prev)
        trend = "↓" if d < -1e-4 else ("↑" if d > 1e-4 else "→")
    spans = sorted(v["step_hi"] - v["step_lo"] for v in versions[:-1] if v["step_hi"] > v["step_lo"])
    steps_per_seg = _median(spans) if spans else None
    lr_phase = None
    if max_steps:
        lr_phase = ("warmup" if step < warmup else
                    "done" if step >= max_steps else
                    "decay" if (decay_start and step >= decay_start) else "stable")

    rec = {
        "run": os.path.basename(run_dir.rstrip("/")), "dir": run_dir, "versions": versions,
        "loss_series": loss_series, "cadence": cadence, "max_steps": max_steps,
        "decay_start": decay_start, "step": step, "pct": (100 * step / max_steps) if max_steps else None,
        "mtime": newest_mtime, "age_s": now - newest_mtime,
        "steps_per_seg": steps_per_seg, "steps_per_h": (steps_per_seg / SEGMENT_HOURS) if steps_per_seg else None,
        "loss_tail_mean": tail_mean, "loss_trend": trend, "lr_phase": lr_phase,
        "nonfinite": len(nf), "first_nonfinite_step": nf[0] if nf else None, "guard": guard, "flags": [],
    }

    # ---- metric-only flags (level, tag, message) --------------------------------------------
    if rec["nonfinite"] > 0 or guard["param"] > 0:
        rec["flags"].append((2, "nonfinite",
            f"NON-FINITE loss x{rec['nonfinite']} from step {rec['first_nonfinite_step']}; "
            f"param_dead={int(guard['param'])} grad_nan={int(guard['grad'])} -- model DEAD, restart from clean snapshot"))
    elif guard["grad"] > 0:
        rec["flags"].append((1, "grad_nan",
            f"grad-NaN self-heal x{int(guard['grad'])} (guard zeroed grads; transient -- watch the count)"))
    if rec["nonfinite"] == 0:
        div, at = detect_divergence(loss_series)
        if div:
            rec["flags"].append((1, "divergence", f"loss tail jump at step {at} (>{DIVERGE_K}sigma)"))
        if detect_plateau(loss_series, cadence, step, decay_start):
            rec["flags"].append((1, "plateau",
                f"loss flat over ~{PLATEAU_WIN_STEPS} steps at step {step} (<decay {decay_start}) -- WARN, investigate not fix"))
    tdrop, ratio = throughput_drop(versions)
    if tdrop:
        rec["flags"].append((1, "throughput", f"steps/segment dropped to {ratio:.0%} of median"))

    rec["severity"] = max((lv for lv, _, _ in rec["flags"]), default=0)
    return rec


def classify_slurm(rec, on_node, any_pending, stall_age_min=STALL_AGE_MIN):
    """Add the SLURM-dependent flags. Don't-cry-wolf rules baked in: a quiet csv is only STALLED when
    a segment is actually on a node; 0-running + 0-pending below max_steps is chain-exhausted."""
    age_min = rec.get("age_s", 0) / 60.0
    if on_node and age_min > stall_age_min:
        rec["flags"].append((2, "stall", f"STALLED: on-node but csv quiet {age_min:.0f} min"))
    if (not on_node) and (not any_pending) and rec.get("step", 0) < (rec.get("max_steps") or 10 ** 12):
        rec["flags"].append((2, "chain",
            "CHAIN EXHAUSTED: nothing running and nothing queued below max_steps -- run stopped silently"))
    rec["severity"] = max((lv for lv, _, _ in rec["flags"]), default=rec.get("severity", 0))
    return rec


# ----------------------------------------------------------------------------- discovery + SLURM
def discover_runs(runs_root, mmin=DISCOVER_MMIN, now=None):
    now = now or time.time()
    cutoff = now - mmin * 60
    newest = {}
    for p in glob.glob(os.path.join(runs_root, "*", "csv", "version_*", "metrics.csv")):
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        if mt >= cutoff:
            run_dir = os.path.dirname(os.path.dirname(os.path.dirname(p)))
            newest[run_dir] = max(newest.get(run_dir, 0), mt)
    return sorted(newest, key=lambda d: -newest[d])


def parse_squeue(out):
    jobs = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        jid, name, state, elapsed, left, nodes = (line.split("|") + [""] * 6)[:6]
        jobs.append({"jobid": jid, "name": name, "state": state,
                     "elapsed": elapsed, "left": left, "nodes": nodes})
    return jobs


def job_to_run(jobid, logs_dir):
    """Map a SLURM jobid to its run via the first-log-line `config=<run>` trick (job names are shared
    across arms, so the log line is the only disambiguator). General glob: <logs_dir>/*_<jobid>.out."""
    for path in glob.glob(os.path.join(logs_dir, f"*_{jobid}.out")):
        try:
            with open(path) as f:
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    m = re.search(r"config=(\S+)", line)
                    if m:
                        return m.group(1)
        except OSError:
            pass
    return None


def sacct_cmd(job, user, since="now-24hours"):
    """Correct sacct invocation: --name= (NOT -n, which is --noheader and makes the name positional)."""
    return ["sacct", "-u", user, "--name=" + job, "-S", since, "-X", "-P", "--noheader",
            "-o", "JobID,State,Elapsed,ExitCode"]


def _elapsed_secs(s):
    parts = s.split("-")
    days = int(parts[0]) if len(parts) == 2 else 0
    hh, mm, ss = (parts[-1].split(":") + ["0", "0", "0"])[:3]
    try:
        return days * 86400 + int(hh) * 3600 + int(mm) * 60 + int(float(ss))
    except ValueError:
        return None


def training_job_names(pending_counts, running_names=(), cached_names=()):
    """The job-names to run the burned-segment sacct check over. A training run is an afterany chain
    with MANY same-named PENDING segments queued ahead; an eval job (fe-gen/fold/report) is a
    singleton. So `count >= 2` cleanly selects training chains and drops per-target eval names — a
    generic discriminator that needs no project-specific name list. Union in names already tied to a
    run (currently running, or cached from a prior tick) so a run with only one queued segment isn't missed."""
    chains = {n for n, c in pending_counts.items() if c >= 2}
    return set(running_names) | set(cached_names) | chains


def slurm_state(user, logs_dir, cached_names):
    """Live squeue map (run -> on_node / job_name) + pending names + burned segments (last 24h),
    splitting exit 15:0 (preemption noise) from real burns. Never raises — reports errors instead."""
    st = {"running_run": {}, "run_name": {}, "pending_names": set(), "pending_counts": {},
          "n_pending": 0, "burned_preempt": 0, "burned_real": [], "error": None}
    running_names = set()
    try:
        q = subprocess.run(["squeue", "-h", "-u", user, "-o", "%i|%j|%T|%M|%L|%N"],
                           capture_output=True, text=True, timeout=60)
        for j in parse_squeue(q.stdout):
            if j["state"] == "RUNNING":
                run = job_to_run(j["jobid"], logs_dir)
                running_names.add(j["name"])
                if run:
                    st["running_run"][run] = j["jobid"]
                    st["run_name"][run] = j["name"]
            elif j["state"] == "PENDING":
                st["n_pending"] += 1
                st["pending_names"].add(j["name"])
                st["pending_counts"][j["name"]] = st["pending_counts"].get(j["name"], 0) + 1
        for name in sorted(training_job_names(st["pending_counts"], running_names, cached_names)):
            s = subprocess.run(sacct_cmd(name, user), capture_output=True, text=True, timeout=60)
            for line in s.stdout.strip().splitlines():
                parts = line.split("|")
                if len(parts) < 4:
                    continue
                jid, state, elapsed, exitcode = parts[:4]
                if state.startswith(("COMPLETED", "PENDING", "RUNNING")):
                    continue
                secs = _elapsed_secs(elapsed)
                if secs is not None and secs < BURN_SECONDS:
                    if exitcode.strip() == "15:0":
                        st["burned_preempt"] += 1                  # SIGTERM preemption == noise
                    else:
                        st["burned_real"].append({"jobid": jid, "state": state,
                                                  "elapsed": elapsed, "exit": exitcode, "name": name})
    except Exception as e:  # squeue/sacct absent or slow -- report, don't crash the health check
        st["error"] = repr(e)
    return st


# ----------------------------------------------------------------------------- state (since-delta)
def load_state(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_state(path, state):
    try:
        with open(path, "w") as fh:
            json.dump(state, fh, indent=1, default=str)
    except OSError:
        pass


# ----------------------------------------------------------------------------- driver
def build_report(runs_root, logs_dir, run_args, user, mmin, use_state, state_path, now=None):
    now = now or time.time()
    if run_args:
        run_dirs = [r if os.path.isabs(r) else os.path.join(runs_root, r) for r in run_args]
    else:
        run_dirs = discover_runs(runs_root, mmin, now)

    recs, missing = [], []
    for d in run_dirs:
        r = read_run(d, now=now)
        (recs if r else missing).append(r if r else {"run": os.path.basename(d.rstrip("/")), "dir": d})

    state = load_state(state_path) if use_state else {}
    job_names = set(v.get("job_name") for v in state.values() if isinstance(v, dict) and v.get("job_name"))
    slurm = slurm_state(user, logs_dir, job_names)

    for rec in recs:
        run = rec["run"]
        on_node = run in slurm["running_run"]
        name = slurm["run_name"].get(run) or (state.get(run, {}) or {}).get("job_name")
        any_pending = (name in slurm["pending_names"]) if name else (slurm["n_pending"] > 0)
        classify_slurm(rec, on_node, any_pending)
        rec["on_node"] = on_node
        rec["jobid"] = slurm["running_run"].get(run)
        rec["job_name"] = name
        prev = state.get(run) if isinstance(state.get(run), dict) else None
        rec["since"] = ({"d_step": rec["step"] - prev["step"],
                         "d_loss": round(rec["loss_tail_mean"] - prev["loss"], 4),
                         "d_hours": round((now - prev["ts"]) / 3600.0, 1)}
                        if prev and "step" in prev else None)

    if use_state:
        for rec in recs:
            state[rec["run"]] = {"step": rec["step"], "loss": rec["loss_tail_mean"], "ts": now,
                                 "job_name": rec.get("job_name") or (state.get(rec["run"], {}) or {}).get("job_name")}
        save_state(state_path, state)

    # A burned segment is lost wall-clock, not a dead model -> WARN (the skill escalates a wave); a
    # large exit-15:0 preempt count may be sick-node churn worth a .err look (>=8/24h).
    worst = max([r["severity"] for r in recs]
                + [1 if slurm["burned_real"] else 0,
                   1 if slurm["burned_preempt"] >= 8 else 0,
                   1 if missing else 0], default=0)
    return {"runs": recs, "missing": missing, "slurm": slurm, "now": now, "worst": worst}


def _fmt_human(rep):
    out = [f"=== training-watch @ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rep['now']))} "
           f"(overall {LEVEL[rep['worst']]}) ==="]
    sl = rep["slurm"]
    q = f"slurm: {len(sl['running_run'])} running, {sl['n_pending']} pending"
    if sl["burned_preempt"]:
        q += f", {sl['burned_preempt']} preempt-burns (noise)"
    if sl["error"]:
        q += f"  ERR={sl['error']}"
    out.append(q)
    if sl["burned_real"]:
        out.append(f"  !! {len(sl['burned_real'])} REAL burned segment(s) <{BURN_SECONDS}s (not preemption):")
        for b in sl["burned_real"][-8:]:
            out.append(f"     {b['name']} {b['jobid']} {b['state']} after {b['elapsed']} exit={b['exit']}")
    for m in rep["missing"]:
        out.append(f"\n--- {m['run']}: NO metrics.csv ({m['dir']})")
    for r in sorted(rep["runs"], key=lambda x: -x["severity"]):
        head = LEVEL[r["severity"]] if r["severity"] else ("OK (queued between segments)" if not r["on_node"] else "OK")
        flagtxt = " ; ".join(f"[{LEVEL[lv]}] {msg}" for lv, _, msg in r["flags"])
        out.append(f"\n--- {r['run']}: {head}" + (f"  !! {flagtxt}" if r["flags"] else ""))
        pct = f"{r['pct']:.1f}%" if r["pct"] is not None else "?%"
        ms = r["max_steps"] if r["max_steps"] else "?"
        since = ""
        if r.get("since"):
            s = r["since"]
            since = f"  since last: +{s['d_step']:.0f} steps, loss {s['d_loss']:+.3f}, {s['d_hours']}h ago"
        out.append(f"    step {r['step']:.0f}/{ms} ({pct}) [{r['lr_phase'] or '?'}]  "
                   f"loss {r['loss_tail_mean']:.4f}{r['loss_trend']}  cad {r['cadence']}  csv age {r['age_s']/60:.0f} min{since}")
        if r["steps_per_h"]:
            eta = ""
            if r["max_steps"]:
                rem_h = (r["max_steps"] - r["step"]) / r["steps_per_h"]
                eta = f"  ETA {rem_h:.0f}h ({rem_h/24:.1f}d)"
            out.append(f"    {r['steps_per_h']:.0f} steps/h, {r['steps_per_seg']:.0f} steps/segment{eta}")
        gd = r["guard"]
        if any(gd.values()):
            out.append(f"    guard counters: nonfinite steps={int(gd['steps'])} grad={int(gd['grad'])} param={int(gd['param'])}")
        out.append("    segments: " + ", ".join(f"v{v['version']}[{v['step_lo']:.0f}-{v['step_hi']:.0f}]" for v in r["versions"]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Run-agnostic training-health detector.")
    ap.add_argument("--runs", nargs="*", default=None, help="run names or abs dirs (default: auto-discover)")
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    ap.add_argument("--logs-dir", default=LOGS_DIR)
    ap.add_argument("--user", default=os.environ.get("USER", ""))
    ap.add_argument("--mmin", type=int, default=DISCOVER_MMIN)
    ap.add_argument("--state", default=STATE_PATH)
    ap.add_argument("--no-state", action="store_true", help="don't read/write the since-last-check state file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rep = build_report(args.runs_root, args.logs_dir, args.runs, args.user,
                       args.mmin, not args.no_state, args.state)
    if args.json:
        strip = {k: [{kk: vv for kk, vv in r.items() if kk != "loss_series"} for r in rep[k]]
                 for k in ("runs", "missing")}
        rep_json = dict(rep, **strip)
        rep_json["slurm"] = {k: (sorted(v) if isinstance(v, set) else v) for k, v in rep["slurm"].items()}
        print(json.dumps(rep_json, indent=2, default=str))
    else:
        print(_fmt_human(rep))
    return rep["worst"]


if __name__ == "__main__":
    raise SystemExit(main())
