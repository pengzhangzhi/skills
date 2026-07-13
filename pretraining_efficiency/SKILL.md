---
name: pretraining-efficiency
description: Speed up large-model FSDP / multi-GPU pretraining or fine-tuning, especially on walltime-limited or preemptible clusters. Validated levers (bf16 all-gather, packing, microbatch, save-on-preemption, weak/strong scaling) with gains, gotchas, and what NOT to bother with.
---

# Large-model training efficiency

Use when speeding up FSDP / multi-GPU pretraining or fine-tuning. Every number below is from a real A/B benchmark on a 6B model.

**Rule 0 — profile first.** Measure exposed-comm %, MFU, padding %, data-wait %, checkpoint stall. Fix the *actual* bottleneck, not the assumed one. (For transformers the MLP+norms are usually the compute; attention is often <5% of FLOPs.)

## Levers, highest ROI first

**1. bf16 FSDP all-gather** — top lever when comm-bound.
Default `bf16-mixed` (Lightning & many frameworks) keeps params **fp32**, so FSDP all-gathers weights in fp32 = 2× the collective volume. Set an explicit `MixedPrecision(param_dtype=bf16, reduce_dtype=bf16)` and keep the **fp32 optimizer master** — loss-neutral, checkpoint-compatible.
→ **+10% tok/s @128 GPU, +18% when comm-bound, −12 GiB peak mem.** The freed memory also unlocks a bigger microbatch (#3).

**2. Sequence packing** — often the biggest lever; usually already done.
Offline-pack variable-length sequences into fixed-length contexts (best-fit-decreasing) with **block-diagonal attention** (no cross-sequence leakage). → ~99%+ token efficiency, near-zero padding. Do it once in preprocessing, not per step.

**3. Maximize microbatch** — after #1 frees memory.
Bigger micro-batch → bigger GEMMs → higher MFU. → **+5–6% / step.** It's a recipe change: re-derive the LR/schedule for the larger global batch (see #5).

**4. Save-on-preemption checkpointing** — goodput, not MFU. Huge on 4h-wall / preemptible clusters.
SLURM `--signal=USR1@<grace>` + a callback that saves a **fresh checkpoint at the current step** on the signal, then stops cleanly → **~0 recompute on requeue** (vs rolling back to the last periodic checkpoint, up to a full ckpt-interval lost). FSDP-safe: async-signal-safe handler only sets a flag; the *collective* save runs at the next batch boundary with an **all-rank flag-reduce** (any rank signaled → all save) so there's no rank-0-only deadlock.
→ recompute/wall 500→0 steps, **~+5–10% GPU-time**.

**5. Scale to more GPUs — weak vs strong.**
FSDP comm is **world-independent per rank**. **Weak-scaling** (fixed per-rank batch, global batch grows with world) stays ~linear — measured **93% at 8× GPUs** — *if* you stay compute-bound (per-rank tokens > comp:comm knee) *and* the global batch stays **below the critical batch size**. **Strong-scaling** (fixed global batch, per-rank shrinks) needs no batch/LR change but is sub-linear (~75%) as you get comm-bound. #1 lowers the knee → both scale better. Scale LR ≈ √(batch ratio); validate the bigger batch with **loss-per-token**, not just tok/s.

**6. Checkpoint storage & stall.**
Cap `save_top_k` (keep `last` + a few) — full-state checkpoints are tens of GB each and pile up fast. Async distributed checkpoint (torch `dcp.async_save` / `AsyncCheckpointIO`) overlaps the write with compute.

## Don't bother (saves you time)
- **Attention kernels when attention is <5% of FLOPs** — the MLP+norms dominate (~95%); fuse *those* (`torch.compile`, fused RMSNorm/SwiGLU). A perfect attention kernel barely moves MFU.
- **Fused cross-entropy for small vocab** — the logit-materialization win only matters at large vocab.
- **More dataloader workers when data-wait ≈ 0** — with whole-shard buffering + packing it usually is.
- **Trusting any lever unbenchmarked** — A/B each one (a few hundred steps: tok/s + loss-parity) before rolling it out.

## Track these (goodput, not just tok/s)
non-pad tok/s/GPU · MFU · padding % · data-wait % · exposed-comm % · peak mem · checkpoint stall · **loss-per-token** (the real convergence metric when batch size changes).
