
# Large-model training efficiency

## Core rule

**Profile before optimizing.** Track:

* useful non-padding tokens/s/GPU
* MFU
* padding and data-wait %
* exposed communication %
* peak memory
* checkpoint stalls
* loss per token

Fix the measured bottleneck. Benchmark every change for throughput **and** loss parity.

## Optimize in this order

### 1. Remove wasted tokens

Pack variable-length examples into fixed token blocks with correct attention boundaries. Prefer offline packing and a small set of stable shapes.

### 2. Use efficient precision

Use BF16 by default. Ensure parameter all-gathers and gradient reductions do not silently use FP32. Preserve FP32 optimizer state when needed for stability.

### 3. Maximize useful microbatch

Increase microbatch until GEMMs are efficient without forcing excessive activation checkpointing. Re-tune learning rate and schedule when the global batch changes.

### 4. Use the simplest parallelism that fits

Prefer data parallelism plus optimizer-state sharding. Add FSDP, tensor, pipeline, or context parallelism only when required by memory or sequence length. Keep frequent communication within the fastest topology.

### 5. Compile stable compute

Compile the repeated numerical core, not orchestration code.

Prefer:

```text
fixed packed shape
→ small static bucket set
→ selectively dynamic dimensions
→ fully dynamic execution
```

Prewarm supported shapes, cache compiler artifacts, and monitor recompiles and graph breaks.

### 6. Fuse dominant operations

Optimize the operators that dominate the profile: usually MLPs, normalization, attention, and communication. Do not optimize kernels that contribute negligible runtime.

### 7. Overlap communication and computation

Overlap gradient reduction, parameter gathering, data transfer, and checkpoint writes with useful work. Measure **exposed** communication rather than total communication.

### 8. Eliminate input stalls

Pre-tokenize and pre-pack data. Use sequential or memory-mapped shards, pinned memory, persistent workers, asynchronous transfer, and local caching only when data wait is measurable.

### 9. Optimize goodput

Use asynchronous distributed checkpoints and save immediately before preemption. Coordinate checkpointing across all ranks; signal handlers should only set a flag.

### 10. Scale deliberately

* **Weak scaling:** preserve per-rank work; verify the larger global batch remains statistically efficient.
* **Strong scaling:** expect diminishing returns as per-rank compute falls below the communication knee.

Evaluate scaling using time-to-target-loss, not tokens/s alone.

## Avoid

* arbitrary dynamic shapes
* unnecessary tensor or pipeline parallelism
* full activation checkpointing when memory already fits
* more data workers when data wait is near zero
* FP8 without hardware support and convergence validation
* optimizing attention, loss, or vocabulary kernels without profiling evidence
* trusting theoretical speedups without end-to-end A/B tests

## Acceptance test

Adopt an optimization only when it provides:

```text
higher useful-token goodput
+ stable memory and recovery
+ no convergence regression
+ enough runtime to amortize setup or compilation cost
```
