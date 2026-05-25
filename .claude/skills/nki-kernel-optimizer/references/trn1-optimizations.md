# Trn1 NKI Optimization Strategies

Tactical optimization strategies for trn1 (NeuronCore-v2) NKI kernels. Vendored from `/home/ubuntu/kchern/autocomp/autocomp/agent_builder/.built/trn1-nki1/optimization_menu.yaml`. Use as a checklist when a kernel is slow.

## General Loop & Data-Movement Strategies

- **Reduce data movement** — every HBM round-trip costs ~820 GB/s of contention. Fuse where possible.
- **Overlap data movement and compute** — DMA engines and compute engines run in parallel.
- **Cache reused data in local memory** instead of reloading from main memory.
- **Loop tiling** to fit working set in SBUF/PSUM.
- **Loop reordering and restructuring** to expose parallelism or improve locality.
- **Loop unrolling** when the loop body is small.
- **Fuse operations** to eliminate intermediate stores/loads.
- **Use lower precision** (BF16 over FP32) where numerics allow.
- **Double buffering** to overlap producer/consumer.
- **Software pipelining** to keep all engines busy.
- **Hoist redundant operations** out of loops.
- **Eliminate redundant computation**.
- **Simplify or remove unnecessary code**.
- **Try new parameter values** (tile sizes, unroll factors).
- **Rewrite the algorithm to reduce total work**.

## Trn1-Specific Tactical Strategies

### Layout / Partition

- **Map contraction axis to partition dimension (P-dim)** to satisfy Tensor Engine layout constraints without reshuffling.
- **Pad tiles to `pmax=128` with masking** to handle non-aligned dimensions while maximizing partition utilization.
- **Maximize free dimension to ≥ 128 elements per partition** to amortize ~100-cycle fixed per-instruction overhead.

### Loop & Scheduling

- **Use `affine_range` instead of `sequential_range`** for loops without true loop-carried dependencies — enables compiler parallelization.
- **Use `modulo allocation` (mod_alloc)** for systematic multi-buffering of physical tiles across loop iterations.
- **Declare buffers inside inner loops** to reduce tensor lifetimes and prevent unexpected spilling.

### TensorE / Matmul

- **Accumulate partial matmul results in PSUM via read-add-write** to avoid extra memory traffic for contraction-dimension tiling.
- **Use `nki.isa.nc_matmul` with pre-transposed inputs** to eliminate implicit transpose overhead from the high-level matmul API.
- **Assign the large-free-axis matrix as the stationary operand** to exploit fast LoadStationary (up to 4× faster data movement than MultiplyMoving).
- **Leverage TensorE for cross-partition reductions and data reshaping** using constant matrices when not matmul-bound.

### ScalarE / VectorE

- **Combine multiply-add with nonlinear activation into a single `nki.isa.activation` instruction** to halve ScalarE cycles.
- **Use hardware `bn_stats` / `bn_aggr` instructions** for single-pass mean and variance computation.
- **Prefer `tensor_scalar` broadcast operations** over explicit broadcast-then-`tensor_tensor` to save an instruction.
- **Use dedicated `tensor_tensor_scan` instructions** instead of explicit sequential loops, to avoid per-instruction static overhead.

### Free-Dim Indexing

- **Exploit free-dimension flexible indexing for transposes, splits, and pooling** via access patterns instead of explicit data shuffling.

### DMA & Memory

- **Ensure DMA transfers are ≥ 32 KiB** by maximizing both partition and free dimension sizes in load/store operations.
- **Replace DMA-based `load_transpose2d` with regular loads plus `nc_transpose`** when the kernel is memory-bound.
- **Use direct allocation APIs** to manually control SBUF/PSUM placement and avoid compiler-inserted spill/refill traffic.
- **Keep reused data resident in SBUF across loop iterations** to avoid redundant HBM reloads.
- **Coalesce small result tiles into a single contiguous buffer** before DMA store to reduce transfer count.
