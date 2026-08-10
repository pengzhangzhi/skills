---
name: babysit
description: Use when the user asks to babysit / watch a deep-learning training run on SLURM — "babysit this training job", "babysit <run>", "keep an eye on my training", "watch the ablation". DEFAULT = CONTINUOUS: it self-perpetuates (tick → handle → ScheduleWakeup → repeat), running unattended in a dedicated tmux Claude session. Healthy → SILENT (no routine pings); it Slacks ONLY on real issues + big milestones (a run breaks/needs you, a fix+recovery, a run finishes). Broken → autonomously root-causes (systematic-debugging + a hypothesis Workflow), fixes, resumes, and Slacks what it did. Auto-discovers active runs. "check once" for a single digest.
---

# babysit

Babysit a training run so it can run **unattended**: each tick, **catch you up** when healthy and
**autonomously debug-fix-resume** when not — pinging you on **Slack** so returning means either "still
healthy, +Δ steps" or "broke at X, fixed by Y, training again." Autonomy on the fix path is opt-in
(see guardrails) — the user grants it once for the run they hand you.

**Default = continuous** (see the last section): "babysit \<run\>" starts a self-perpetuating watch,
not a one-shot. Runs in its own long-lived tmux Claude session.

The deterministic detection lives in `train_watch.py` (stdlib-only, CPU, read-only, unit-tested). This
skill is the **judgment + reaction** on top of it. It assumes a PyTorch-Lightning-style run layout
(`<run>/csv/version_*/metrics.csv` + optional `resolved_config.yaml`); point it at your roots with
`$BABYSIT_RUNS_ROOT` / `$BABYSIT_LOGS_DIR` and adapt column names in `read_run()` if your logger differs.

## Step 1 — measure

```bash
python3 train_watch.py --json      # auto-discovers active runs; writes since-last-check state
```

`--json` gives `{worst, runs[], missing[], slurm}`. Each run carries `severity` (0 OK / 1 WARN / 2
CRIT), `flags` (`[level, tag, msg]`), `step/pct/max_steps`, `lr_phase`, `loss_tail_mean/loss_trend`,
`steps_per_h`, `guard` counters, `on_node`, `since` (Δ vs your last check). Add `--runs <a> <b>` to
scope, `--no-state` for a read-only peek. Run the plain (non-`--json`) form to show the human report.

## Step 2 — branch on severity

### `worst == 0` (all OK) → **catch-me-up digest**

One tight block per run — nothing more:

```
<run>: <lr_phase> step <X>/<max> (<P>%) · loss <L> <trend> · <steps/h> · ETA <h>h
       since last: +<Δstep> steps, loss <Δloss>  [+ any WARN note]
```

Lead with a one-line verdict ("4 runs, all healthy"). Do **not** spin up agents for a healthy run.

### any `severity >= 1` → **debug-fix-resume**

1. **Announce + invoke `systematic-debugging`.** No fix before a reproduced/understood cause.
2. **Warm-start from the catalog.** Read `failure-catalog.md`, match the `flags` to a mode, but
   **verify against the actual CSV/logs** — a documented pattern can be stale.
3. **Gather evidence** for the run: its latest SLURM `.out`/`.err` (`ls -t <LOGS_DIR>/*_<jobid>.out`,
   or the newest `*.out` whose line-1 has `config=<run>`), the failing CSV window
   (`<run>/csv/version_*/metrics.csv`), `<run>/resolved_config.yaml`, `sacct --name=<job> ...` history
   (never `-n` — that's `--noheader`), and recent git touching the training loop / launcher / data.
4. **Root-cause via a Workflow** (for anything non-obvious) — parallel hypothesis agents, one per
   class, each returning a verdict + evidence; synthesize the survivor:
   `data/mixer` · `numerics/grad` · `config/dead-knob` · `node/hardware` · `schedule/LR`.
   Scale it to the problem (a clearly-preemption stall doesn't need 5 agents).
5. **Apply the fix under guardrails** (below), **resume**, **verify recovery** (re-run the detector
   once the new segment lands — steps advancing + finite + `guard` counters flat), **ledger + notify**.

## Severity → first move (quick map; full detail in failure-catalog.md)

| flag | move |
|---|---|
| `nonfinite` (CRIT) | Model likely dead (a guard rarely self-heals a param-NaN). **Rename-aside** the run dir (preserve evidence), resume **from the last clean checkpoint/snapshot BEFORE the NaN, not the latest** (it may be dead). Catalog #1/#14. |
| `grad_nan` (WARN) | A guard zeroed grads = transient self-heal. Watch the count across ticks; escalate only if it's **rising** (→ systematic). |
| `stall` (CRIT) | On-node but wedged (hang). `scancel` **only that jobid**; the chain requeues. |
| `chain` (CRIT) | Chain exhausted (0 running + 0 pending, step<max). Re-seed the chain via your resubmit script. Verify no `<run>/DONE` sentinel first. |
| `divergence` (WARN) | Loss jumped up; may precede a NaN. Inspect the window + LR; don't restart blindly. |
| `plateau` (WARN) | Often **real convergence**, not a bug. **Do NOT auto-fix/restart** — note it, suggest an LR/budget review. |
| `throughput` (WARN) | Preemption / dataloader stall / slow node. Check `perf` + node id, not loss. |
| `burned_real` (WARN) | Real burn: check `.err` for a config exception (fix config) or a node error (exclude the node). |
| high `burned_preempt` | ≥8 short SIGTERM burns/24h = possible sick-node churn; check `.err` clustered by node. Otherwise preemption noise — ignore. |

## Resume / launch

Resuming is site-specific — use the run's own chain/resubmit mechanism (re-running the segment sbatch
with the same config usually auto-restores from the latest checkpoint). After a **systematic NaN**,
resume from a clean checkpoint/snapshot **before** the failure, not the latest (which may be dead), and
verify the resume path against the run's config before launching (don't guess).

## Guardrails (autonomy is granted per-run; these protect healthy work)

- **Name-guard.** Only `scancel` the specific jobid under diagnosis. Never touch a *healthy* run's
  jobs. Never `scancel` a self-perpetuating chain without disabling it first (else it re-fires).
- **Never delete checkpoints/snapshots.** Preserve the failing state (**rename-aside**, never
  overwrite) — after a NaN the only surviving checkpoints may be the dead ones.
- **Reproduce before editing code.** Code fixes go through a subagent that reproduces the failure
  first, lands on a branch, minimal diff, rationale in the commit message.
- **Idempotent + state-from-disk.** Re-running submits nothing already in flight (check `squeue`).
- **Don't "fix" a plateau.** WARNs that are expected behavior get reported, not acted on.
- **Ledger every action** to `<LOGS_DIR>/train_watch_ledger.md` (symptom → cause → fix → resubmitted
  jobid → recovery evidence), before and after.

## Continuous by default (+ Slack)

**"Babysit \<run\>" is a continuous watch, not a one-shot.** Each invocation:

1. Runs **one tick** — Step 1 (measure) then Step 2 (digest, or debug-fix-resume).
2. **Slacks** per the policy below.
3. **Schedules the next tick** with `ScheduleWakeup`, re-firing this same babysit request:
   `delaySeconds` ≈ **1200–1800** (20–30 min) while idle/healthy; tighten to **120–240** (2–4 min)
   while actively fixing or verifying a recovery. Pass the babysit request back as the wakeup `prompt`.

So it self-perpetuates until you say **"stop babysitting"**. Each tick is cheap and idempotent (state
from disk + `squeue`), so nothing long-lived races. Equivalent explicit form: `/loop babysit`. Say
**"check once"** (or `--once`) for a single digest with no rescheduling.

**Where it runs.** A dedicated, long-lived **Claude Code session in its own tmux** — the session's
lifetime is the watch's lifetime. Start it, say "babysit \<run\>", detach; it keeps ticking silently
and only Slacks you when something's worth telling.

**Slack (it messages you).** Post with the bundled helper —
`bash slack_notify.sh "<message>"` — which reads the destination from `notify.txt` (create it from
`notify.txt.example`) and a bot token from `$BABYSIT_SLACK_TOKEN`, resolves a `U…` member id to your
DM, and posts `chat.postMessage`. Token-based, so unattended `ScheduleWakeup` ticks work without
interactive OAuth. Policy — **only issues + big things; silence = healthy.**

The watch **ticks every ~30 min regardless** (so problems are caught fast), but **Slack ONLY on**:
- **A problem it hit / acted on / needs you for** — a run goes CRIT, an action was taken, or it's
  stuck: one line, what + why.
- **Recovery** — "fixed \<run\>: \<cause\> → \<fix\>, resubmitted \<jobid\>, steps advancing again."
- **A big milestone** — a run finished (`DONE`/reached `max_steps`), or all watched runs finished.

**No routine/progress pings.** Absence of a message means everything is fine. **Notify on state
CHANGE, not every tick** — use the ledger as cross-tick memory: Slack once when a condition appears,
again only when it changes (resolved, worsened, or new). Never re-Slack a standing condition each tick.

Always print the digest in-session too. If `notify.txt`/`$BABYSIT_SLACK_TOKEN` are unset, print a
one-line reminder and continue (don't fail the tick).
