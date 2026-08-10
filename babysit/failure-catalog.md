# Training failure catalog — symptom → cause → recovery → signal

Warm-start for the `babysit` autonomous debugger: a field guide to the ways a long, chained SLURM
training run fails, and how to tell them apart from the metrics/`squeue`/`sacct`/logs. When escalating,
match the observed signal to a mode here **before** starting a cold `systematic-debugging` pass — but
verify against the actual CSV/logs, don't assume. These are distilled from a real multi-day training
study; adapt thresholds and column names to your setup.

Exit-code decoder used throughout: **`15:0` = SIGTERM = preemption/requeue (usually expected noise)**,
`1:0` = a real error in your program, `NODE_FAIL` / a CUDA-init error = a bad node,
`MisconfigurationException` = a framework/config crash. `sacct` gotcha: use **`--name=<job>`**, never
`-n` (that's `--noheader` and silently makes the name positional → reports clean through every failure).

---

## Numeric / NaN

**1. Abrupt run-poisoning NaN (dead model keeps "training").**
Loss goes `nan/inf` abruptly with little or no run-up, then the job trains many more steps looking
healthy on the dashboard while a later checkpoint is entirely non-finite.
- Cause: often a single non-finite gradient element from a pathological batch or a transient hardware
  fault — frequently *not* a reproducible software bug. Global gradient clipping is an **amplifier**:
  one NaN grad element makes the global norm NaN, which scales every parameter to NaN in a single
  optimizer step → abrupt and total.
- Recovery: kill immediately; resume from the last **clean** checkpoint/snapshot BEFORE the NaN step
  (the latest checkpoint may already hold dead weights). Rename-aside the failing dir; never overwrite.
- Signal: `train/loss` = nan/inf (scan **every** row of **every** version); any guard counters
  (`train/nonfinite_{steps,grad,param}_steps`) > 0; grad-norm spikes then nan. Logs: your guard's
  "NON-FINITE" / "model is DEAD" prints, if you have them.

**2. Silent NaN under FSDP sharding (counter reads 0 while loss=nan).**
- Cause: rank-local, rank-0-only detection. FSDP FULL_SHARD localizes a non-finite element to the one
  rank that owns that shard, and a rank-0-only logger never sees a NaN owned by another rank.
- Recovery: make the verdict **global** — OR per-tensor non-finite flags on-device, then one
  `all_reduce(MAX)` so every rank makes the same decision; log counters with `sync_dist=True,
  reduce_fx="max"`. Triaging an old run: don't trust `nonfinite_steps==0` next to `loss==nan`; grep
  per-rank `.err`.

**Guard-aware severity.** A **transient single-batch** grad-NaN can **self-heal** (a guard zeros grads
on all ranks, momentum-only step, run continues) → a grad counter ticks up but loss stays finite =
**WARN, watch the count**. A **systematic** NaN or **already-dead params** (a param counter > 0, or
`train/loss` non-finite) does **NOT** self-heal → **CRITICAL, kill + restart from a clean checkpoint**.

---

## SLURM / chain / node

**3. A bad node burns chained job slots.**
A job fails in seconds having done zero training; because `afterany` releases the successor instantly,
SLURM re-allocates the **same** bad node and many segments die in minutes.
- Cause: a node that fails `torch._C._cuda_init()` (e.g. "CUDA unknown error") while the scheduler
  still shows it allocatable, so it keeps getting handed out. Amplified by `afterany` re-hitting it.
- Recovery: `--exclude=<nodes>` at submit, or `scontrol update JobId=<id> ExcNodeList=<nodes>` for
  queued jobs (reversible with `ExcNodeList=''`). Don't swallow the exit code in stage wrappers.
- Signal: `sacct --name=<job> -S now-24hours -X -P -o JobID,State,Elapsed,ExitCode`: State
  FAILED/NODE_FAIL, Elapsed < 300s, **clustered on one node**. `.err`: the CUDA-init error.

**4. Chain exhausted — run stops silently below max_steps.**
Nothing running, nothing queued for the job name; the log goes quiet but the run is NOT wedged (no
segment is on a node).
- Cause: a finite pre-submitted `afterany` stack ran out, an un-chained single job, or a
  self-perpetuating chain hit its max-segment ceiling.
- Recovery: re-seed the chain with your resubmit script. Verify no `<run>/DONE` sentinel (its presence
  is a legitimate finish). To STOP a self-perpetuating chain, disable it (e.g. `touch <run>/DONE`)
  FIRST, then scancel (scancel alone re-fires the `afterany` successor).
- Signal: `squeue -h -u $USER -n <name>` → 0 RUNNING **and** 0 PENDING; max step < max_steps.

**5. Wall-clock TIMEOUT not covered by `--requeue`.**
A run stops partway; the tracker may show "Finished" (a clean checkpoint was written) so it looks done.
- Cause: on a non-preemptible partition, a standalone job hits the walltime (TIMEOUT), and `--requeue`
  does not cover TIMEOUT there — so nothing resumes.
- Recovery: never run a long job as a single sbatch; use an `afterany` segment chain and checkpoint +
  resume every segment.
- Signal: `sacct` State=TIMEOUT; `.err`: `CANCELLED AT <t> DUE TO TIME LIMIT`; step plateaus with a
  clean checkpoint present and no successor queued.

**6. Preemption on a backfill/preemptible partition (expected).**
A segment is killed mid-run and requeued; a version dir ends abruptly, throughput/ETA slip.
- Recovery: nothing to fix — `--requeue` + a pre-timeout signal handler (`--signal=USR1@N`) flushes a
  checkpoint in the grace window; resume from it. For zero-tolerance runs, prefer non-preemptible.
- Signal: `sacct` State=PREEMPTED, or `.err` `CANCELLED AT <t>` **without** `DUE TO TIME LIMIT`;
  requeued (same JobID, new Start). **Do not debug this — it's noise.**

**7. STALLED — on-node but log quiet (a real hang).**
A segment is RUNNING (`squeue`) but the metrics file hasn't been appended in a long time — dataloader
deadlock, NCCL collective hang, or a node that went bad mid-segment.
- Recovery: `scancel` the wedged segment; the chain requeues (or re-seed).
- Signal: a `squeue` RUNNING job for the run **AND** newest `version_*/metrics.csv` mtime age > ~15
  min. Contrast: no on-node segment + high age = healthy "queued between segments", **not** a stall.

---

## Throughput / OOM / config

**8. Throughput trap or real drop.**
Wall-clock ~25% slower than `perf/sec_per_step` implies, or a genuine ~25–30% drop.
- Cause: `perf/sec_per_step` excludes the model-load + dataloader warm-up paid at the start of every
  segment (overstates throughput). A real drop = preemption requeue, dataloader stall, or a slow node.
- Recovery: measure **steps-per-segment from completed version dirs** (median, exclude the in-flight
  last one; skip the first post-resume row). For a real drop, check `perf` + node id, not just loss.
- Signal: shrinking `(step_hi − step_lo)` across successive version_* dirs.

**9. Validation-step OOM.**
OOMs at a val/eval step (not the train step); peak memory near the cap, timed to the eval interval.
- Cause: a val probe at a larger sequence length / batch than training; attention/triangle-mul memory
  scales super-linearly in length.
- Recovery: cap the val length/batch; reduce `max_len`.
- Signal: `.err` `torch.OutOfMemoryError` at a val step; `perf/peak_mem_gb` peaking at the eval cadence.

**10. Illegal checkpoint config crashes segments.**
Segments die fast at the first checkpoint write.
- Cause: e.g. Lightning's `ModelCheckpoint` with `monitor=None` accepts only `save_top_k ∈ {0,1,-1}`;
  other values raise `MisconfigurationException("No quantity for top_k to track")` at save time.
- Recovery: keep the value legal; `save_last=True` retains a resume checkpoint regardless.
- Signal: `sacct` FAILED at/after the first save interval; `.err` that exception; burns (Elapsed<300s)
  appearing at the checkpoint interval.

**11. Dead-config knob (no effect).**
A set knob does nothing — e.g. an eval interval is configured but no `val/*` columns ever appear.
- Cause: the knob isn't actually read by THIS launcher/loop.
- Recovery: before trusting a knob, grep the launcher/loop for the key. Don't infer from the config
  file alone.
- Signal: an expected metric family absent from the metrics file despite the config.

---

## Learning / data (WARN, not "fix")

**12. Plateau / mild overfitting — looks stuck, isn't broken.**
Loss slope ~0 over a long window while far from the step budget; held-out metric flat or drifting the
wrong way.
- Cause: real convergence, not a bug — a small model / easy corpus can saturate fast; the stable-phase
  LR may be too low.
- Recovery: **WARN — do NOT auto-fix or restart.** Consider the LR schedule / whether to spend the
  remaining budget; the best checkpoint may be earlier than the latest.
- Signal: `train/loss` slope ~0 with step << max_steps; NO non-finite counters. Distinguish from
  divergence (a loss tail jump beyond k·rolling-std).

**13. Data-stream replay-from-t0 on requeue → memorization.**
A sampling/memorization metric pins at its ceiling; corpus coverage caps no matter how many steps run.
- Cause: the checkpoint load drops the dataloader cursor, so every requeue replays the same
  examples/masks from t=0 — coverage capped by the longest single segment.
- Recovery: use `global_step` (restored identically on every rank) as a resume cursor to fast-forward
  the stream, and fold the step into any masking seed.
- Signal: a memorization metric stuck at its ceiling; not visible in `train/loss` — watch held-out /
  per-source signals.

---

## Evidence preservation (guardrail, not a failure)

**14. Post-NaN there's nothing to rewind to.**
- Cause: a retention policy (e.g. `save_top_k=1`) deletes the previous checkpoint each save; by the
  time a NaN is noticed, the surviving checkpoints are already dead.
- Recovery: keep periodic **snapshots** separate from the rolling checkpoint; a watcher MUST
  **preserve the failing state (rename-aside, never overwrite/delete)** so the cause stays inspectable.
  Logging grad-norm every step helps distinguish a one-shot poison from a divergence run-up.
- Signal: only the latest (dead) checkpoints on disk; nothing older than the failure.
