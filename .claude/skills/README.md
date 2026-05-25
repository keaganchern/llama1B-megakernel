# Vendored Skills

These skills were vendored from `../../nki-moe-megakernel/.claude/skills/` on 2026-05-24, then patched for trn1.2xlarge. See `TRN1_NOTES.md` for the hardware differences that the trn2-targeted reference docs do not capture.

## What's here

| Skill | What it does | When to invoke |
|---|---|---|
| `nki-kernel-optimizer` | Writing, reviewing, optimizing, benchmarking NKI kernels | Any NKI kernel work; mentions of nki.lang/nki.isa/SBUF/PSUM/Trainium |
| `neuron-profile` | Analyzing Neuron profiler output (NTFF/NEFF files) | After running with profiling enabled, to inspect what actually ran on chip |
| `trainium-model-translation` | Porting models to Trainium; **includes `scripts/block_testing_utils.py`** which has `test_block_correctness()` for validating any NxDI block against a PyTorch reference | When integrating a new model or block; when validating that a swapped kernel matches the reference numerically |

## Quick start references

- **Want to write a new NKI kernel** → `nki-kernel-optimizer/references/nki-syntax-quickref.md` (cheatsheet) and `references/templates.md` (boilerplate)
- **Want to verify your kernel is correct** → `trainium-model-translation/scripts/block_testing_utils.py` — wrap your block, call `test_block_correctness()`, get a pass/fail with numerical diff
- **Want to benchmark a kernel** → `nki-kernel-optimizer/scripts/benchmark.py` + `references/benchmarking-api.md`
- **Want to know why a kernel is slow** → `neuron-profile/SKILL.md` + `references/reading_the_profiler.md`
- **Hit a compiler error you don't understand** → `nki-kernel-optimizer/references/common-pitfalls.md`

## What was deliberately NOT vendored

- `nki-kernel-optimizer-trn3` — targets NCv4 / MXFP4-8 hardware that doesn't exist on trn1.
- Inside `nki-kernel-optimizer`: `references/trn3-architecture.md`, `references/fp8-mxfp-quantization.md` — same reason.
