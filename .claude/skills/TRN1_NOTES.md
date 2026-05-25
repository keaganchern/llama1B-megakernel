# trn1 Adaptation Notes for Vendored Skills

These skills (`nki-kernel-optimizer`, `neuron-profile`, `trainium-model-translation`) were vendored from `../../nki-moe-megakernel/.claude/skills/` (which target trn2/trn3). This file lists the places trn1 differs so the trn2-focused reference docs don't quietly mislead.

For the **authoritative trn1 hardware spec, coding rules, and optimization checklist**, see:
- `nki-kernel-optimizer/references/trn1-architecture.md`
- `nki-kernel-optimizer/references/trn1-coding-rules.md`
- `nki-kernel-optimizer/references/trn1-optimizations.md`

All three were vendored from `/home/ubuntu/kchern/autocomp/autocomp/agent_builder/.built/trn1-nki1/` (an auto-generated trn1-NKI agent config) and confirm the numbers below.

## ⚠️ Critical gen3+ trap in nkilib/experimental/transformer/

**DO NOT call `transformer_tkg` or `attention_block_tkg` from `nkilib/experimental/transformer/` on trn1.** Both hard-fail at compile time:

```python
# nkilib/experimental/transformer/attention_block_tkg.py:635
kernel_assert(
    nisa.get_nc_version() >= nisa.nc_version.gen3,
    f"Kernel requires nc-version >= gen3, got {nisa.get_nc_version()}",
)
```

trn1 = NCv2 = gen2 < gen3. `transformer_tkg` calls `attention_block_tkg` for every layer, so the whole `experimental/transformer/` path is gen3-locked. There is **no off-the-shelf multi-layer megakernel in nkilib that works on trn1** — that's the only one.

### Gen2-compatible building blocks (use these instead)

`nkilib/core/` has lower-level kernels that explicitly support gen2:

| Building block | Entry point | What it does |
|---|---|---|
| QKV projection | `core/qkv/qkv_tkg.py::qkv_tkg` | RMSNorm(X) + QKV matmul, output to SBUF or HBM |
| RoPE | `core/embeddings/rope.py::RoPE` and `RoPE_sbuf` | Rotary embedding for Q and K |
| Attention compute | `core/attention/attention_tkg.py::attention_tkg` | Q@K + softmax + @V with KV cache. Supports `fuse_rope`. Adaptive LNC2 sharding built-in. |
| Output projection | `core/output_projection/output_projection_tkg.py::output_projection_tkg` | Attn-out matmul → hidden |
| Full MLP block | `core/mlp/mlp.py::mlp` | RMSNorm + gate/up/down + SiLU all fused |
| RMSNorm (plain) | None in `core/rmsnorm/` (quant-only). Use `mlp()`'s built-in, or `nki_rmsnorm_kernel` from our llama.py | |

To build a Llama megakernel on trn1, write our own multi-layer driver that calls these — the same shape of work `transformer_tkg` itself does, just with `core/attention/attention_tkg` instead of the gen3-only `experimental/transformer/attention_block_tkg`.

## Hardware: trn1.2xlarge = 1 Trainium chip = 2 NeuronCore-v2 cores

| | trn1 (NCv2, your hardware) | trn2 (NCv3, reference docs assume this) |
|---|---|---|
| NeuronCores per chip | 2 | 8 |
| Device HBM | **32 GiB, 820 GB/s, 2 stacks** | 96 GiB |
| Per-core SBUF | **24 MiB = 128 partitions × 192 KiB each (176 KiB usable, 16 KiB compiler-reserved)** | Different |
| Per-core PSUM | **2 MiB = 128 partitions × 16 KiB each = 8 banks × 512 FP32/bank** | Different |
| Partition dim (`nl.tile_size.pmax`) | 128 | 128 ✓ same |
| DMA engines per core | **16 (27 GiB/s each, 32 per device)** | Different + per-engine DGE |
| TensorE peak | **92 TFLOPS BF16/FP16/TF32/cFP8, 23 FP32 @ 2.8 GHz** | 158 FP8 / 79 BF16 TFLOPS @ 2.4 GHz |
| VectorE | **128 lanes @ 1.12 GHz, 2.3 FP32 TFLOPS; reduction op = ADD only** | Different |
| ScalarE | **128 lanes @ 1.4 GHz, 2.9 FP32 TFLOPS** | Different |
| GpSimd | **8 SIMD procs @ 1.4 GHz, 64 KB each, 3-cycle latency** | Different |
| Logical NeuronCore (LNC) options | LNC=1 only (or trivially LNC=2 = the whole chip) | LNC=1, 2, 4 |
| Native GQA TP support | **No** — NxDI falls back to `GQA.CONVERT_TO_MHA` (duplicates K/V per rank) | Yes |
| Per-engine integrated DMA (DGE) | No | Yes (NCv3 feature) |
| FP8 matmul double-throughput | No | Yes (`double_row` matmul) |
| MXFP4 / MXFP8 / `nc_matmul_mx` | No (trn3 only) | No (trn3 only) |
| `nki.jit(platform_target=...)` | Use `"trn1"` | Use `"trn2"` |
| `NEURON_PLATFORM_TARGET_OVERRIDE` | Set to `trn1` (or leave unset to auto-detect) | Set to `trn2` |

The vendored files have already had their `platform_target` strings and `NEURON_PLATFORM_TARGET_OVERRIDE` env vars patched to `trn1`.

## Trn1-only behavior worth knowing

These are confirmed in `trn1-architecture.md` and the autocomp rules:

- **TensorE tile limits**: stationary free axis ≤ 128, partition K ≤ 128, moving free axis ≤ 512. Contraction axis K must be in partition dim for both operands. Larger free axis should be stationary (LoadStationary up to 4× faster than MultiplyMoving).
- **MM initiation interval**: `max(N, 64)` TensorE cycles for BF16/FP16/TF32/cFP8; FP32 ~4× more.
- **Free-dim stride penalty**: stride < 16 B → peak 128 elem/cycle; stride ≥ 16 B → ~50% peak. ~60 cycle static overhead per tensor access.
- **Partition start alignment**: > 64 partitions must start at 0; > 32 at 0 or 64; ≤ 32 at 0/32/64/96.
- **Engine parallelism**: all 4 engines run concurrently, but VectorE + GpSimdE cannot access SBUF in parallel, and VectorE + ScalarE cannot access PSUM in parallel (compiler serializes).
- **PSUM accumulation pattern (exact)** — only this triggers PSUM accumulation:
  ```python
  psum_buf = nl.zeros((128, F), dtype=nl.float32, buffer=nl.psum)
  for i in nl.affine_range(...):
      psum_buf += nl.matmul(...)   # CORRECT — accumulates
  # WRONG (silent bug): psum_buf[...] = psum_buf + nisa.nc_matmul(...)
  ```
- **Arithmetic intensity threshold to saturate TensorE at BF16**: ~222 Flops/Byte.
- **Compute imbalance**: TensorE has ~92 TFLOPS vs VectorE 2.3 and ScalarE 2.9 — a ~30–40× gap. Minimize non-matmul work or overlap it with matmul via engine parallelism.
- **Use `&` / `|` to combine masks**, not Python `and` / `or`. Logical operators don't work on NKI tensors.

## What the trn2 reference docs say that does NOT apply to trn1

### `nki-kernel-optimizer/references/dma-patterns.md`
- `dma_transpose` and the integrated-DGE patterns assume NCv3. On trn1 you'd use `nc_transpose` more often. Most other patterns (`.ap()` strides, scalar broadcast, weight ring buffers) transfer.

### `nki-kernel-optimizer/references/sbuf-allocation.md`
- The numerical SBUF budgets in that file are written for NCv3. For trn1 the actual numbers are: 24 MiB total per core, 128 partitions × 176 KiB usable each. The *allocation rules* (all-or-nothing auto vs manual, `NCC_EGCA111` handling) still apply.

### Removed from this vendoring
- `fp8-mxfp-quantization.md` — trn3-only feature, removed.
- `trn3-architecture.md` — removed (not your hardware).
- `trn2-architecture.md` — **replaced** with `trn1-architecture.md` (the authoritative reference).
- The whole `nki-kernel-optimizer-trn3` skill — not vendored.

## Things that "just work" identically on trn1

- The whole NKI language (`nl.*`): `nl.load`, `nl.store`, `nl.matmul`, `nl.sum`, `nl.rsqrt`, `nl.multiply`, tile indexing, `nl.shared_hbm` vs `nl.sbuf`.
- Partition dimension of 128.
- The full `nki-syntax-quickref.md` cheatsheet (minus the commented-out MX/FP8 examples).
- The `performance-playbook.md` workflow (the principles are hardware-agnostic).
- The `common-pitfalls.md` list (compiler errors, dtype gotchas).
- The `neuron-profile` skill — profiler output format is the same.
- The `trainium-model-translation` skill — translation patterns and `block_testing_utils.py` are hardware-agnostic.
- The `benchmark.py` script (after the trn1 patch).

## On-demand references (not vendored due to size)

When the in-tree docs don't have what you need:
- `/home/ubuntu/kchern/autocomp/autocomp/agent_builder/.built/trn1-nki1/code_examples.md` (1498 lines) — full working trn1 kernels: rmsnorm, layernorm, matmul, fused-self-attn, fused-mamba, transpose2d, average_pool, spmd_tensor_addition, block-dim migration. Read directly when writing a similar kernel.
- `/home/ubuntu/kchern/autocomp/autocomp/agent_builder/.built/trn1-nki1/isa_docs.md` (6598 lines) — exhaustive trn1 `nl.*` / `nisa.*` API reference. Read when you need an API call that isn't in `nki-syntax-quickref.md`.

## Practical implications for this project (Llama-3.2-1B on trn1)

1. **GQA limitation**: Llama-3.2-1B has 32 Q heads and 8 KV heads. With `tp_degree=2` you saw the warning `TP degree (2) and KV heads (8) are not divisible. Overriding ... to GQA.CONVERT_TO_MHA`. That's the trn1-no-native-GQA-support kicking in. Each rank ends up holding all 8 KV heads (duplicated). Costs memory; doesn't affect correctness.
2. **Stay bf16**: don't try FP8/MX paths in any vendored docs — they won't compile on trn1.
3. **LNC=2 = the whole chip**: many trn2 docs talk about combining 2 of 8 cores into LNC=2. On trn1 you only have 2 cores total, so LNC=2 just means "use both cores" — there's no other configuration.
4. **Per-engine DGE optimizations don't help you**: any "use DGE for X" advice should be reframed as "use a main DMA engine for X."
5. **Llama-1B SBUF budget**: 16 layers, hidden=2048, KV head_dim=64. Per-core SBUF is 24 MiB / 176 KiB per partition. A single layer's worth of residual (`[128, 16]` partition tiling of the 2048-dim hidden state at bf16 = ~32 KiB/partition) leaves plenty of room — fitting all 16 layers' residuals SBUF-resident across a megakernel is plausible.
