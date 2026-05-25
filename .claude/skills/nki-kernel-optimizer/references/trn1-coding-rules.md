# Trn1 NKI Coding Rules

Hard rules for writing trn1 (NeuronCore-v2) NKI kernels. Vendored from `/home/ubuntu/kchern/autocomp/autocomp/agent_builder/.built/trn1-nki1/rules.yaml`.

## Semantics & Signature

- The rewritten program must be **semantically equivalent** to the original within a small numerical tolerance.
- Keep the same function name and signature as the original (helper functions can be renamed or deleted).

## Memory Model

- **All kernel input/output tensors must reside in HBM.** Data must be explicitly loaded from HBM to SBUF via `nl.load` before computation, and results stored back via `nl.store`. Matmul results in PSUM must first be copied to SBUF via `nl.copy` before storing to HBM.
- **Output tensors** must be declared with `buffer=nl.shared_hbm` via `nl.ndarray`, must be explicitly written via `nl.store`, and must be returned from the kernel.
- **Tensors defined inside `if/else/for` blocks cannot be used outside that scope** — declare in outer scope and assign into them.

## Tile Shape & Indexing

- **Partition dimension (P)** is always the first/leftmost dimension of any SBUF/PSUM tile. Hard maximum **128** elements (`pmax = 128`). All NKI compute APIs require the partition dim as the first dim.
- **Tiles in SBUF/PSUM must have ≥ 2 dimensions.** 1D tiles cause `Insufficient rank` errors. Use `(128, 1)` not `(128,)`. Buffers with shape `[N, 1]` or `[1, M]` must be indexed with both indices explicitly (e.g., `my_sbuf[0:N, 0]`).
- **Partition dimension indices** must be column vectors: `nl.arange(N)[:, None]`.
- **Free dimension indices** must be row vectors: `nl.arange(M)[None, :]`.
- **Do not mix** basic indexing (slices) and advanced indexing (`nl.arange`-based) in the same index tuple.

## Operators & Broadcasting

- **Partition dimension broadcasting is NOT supported** on Python operator overloads (`+`, `-`, `*`, `/`). Use `nl.add`, `nl.multiply`, etc.
- **Free-axis broadcasting** works implicitly.
- **Combining masks** uses `&` and `|`, not Python `and` / `or`. Python logical operators cannot be used on NKI tensors.
- **Control flow conditions** (`if` / `while`) cannot depend on `nl.arange` or runtime tensor values — use the `mask` parameter on APIs instead.

## Masking for Non-Aligned Dimensions

When tile indices may exceed actual tensor dimensions (tensor size not a multiple of tile size), a `mask` parameter must be passed to `nl.load`, `nl.store`, and compute APIs to prevent out-of-bounds access. Mask expressions must be affine expressions of `nl.arange`, `nl.affine_range`, or `nl.program_id`.

## Matmul (`nc_matmul`) Constraints

- `nc_matmul` computes **`stationary.T @ moving`**, reads both inputs from SBUF, and always writes **FP32 results to PSUM**.
- **The contraction axis must be in the partition dimension for both operands.**
- **Tile size limits:** stationary free axis ≤ 128, moving free axis ≤ 512, partition axis ≤ 128.
- If the contraction dimension exceeds 128, split it into chunks ≤ 128 and accumulate across multiple `nc_matmul` calls into the same PSUM buffer.
- **PSUM accumulation pattern (exact)** — must be:
  1. Initialize with `nl.zeros(..., buffer=nl.psum)`
  2. Use `nl.affine_range` for the loop
  3. Accumulate via `psum_buf += nl.matmul(...)`

  **Using `psum_buf[...] = psum_buf + nisa.nc_matmul(...)` will NOT trigger PSUM accumulation.** This is a silent correctness bug.

## Loop Constructs

- **`nl.affine_range`** only when there are no loop-carried dependencies between iterations. Associative reductions (e.g., matmul accumulation via `+=`) are NOT loop-carried dependencies.
- **`nl.sequential_range`** for true loop-carried dependencies.
- **Plain Python `range()`** is silently converted to `sequential_range`.

## Capacity Limits (Trn1 / NCv2)

- **SBUF**: 24 MiB total = 128 partitions × 176 KiB usable each. All simultaneously live tiles must fit.
- **PSUM**: 2 MiB total = 128 partitions × 16 KiB each = 8 banks × 512 FP32 elements per bank.
- **PSUM free dimension ≤ 512** per tile (one bank). `bn_stats` free dimension ≤ 512. Hard hardware limits.

## Direct (Manual) Allocation

- When using direct allocation, **ALL tensors must use direct allocation**. Mixing with automatic allocation (`buffer=nl.sbuf` / `nl.psum`) is forbidden.
- `nisa.nc_transpose` with TensorEngine and high-level APIs like `nl.softmax` are **not allowed in allocated kernels**.
