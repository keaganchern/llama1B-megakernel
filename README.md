# llama-megakernel

A multi-layer NKI megakernel for Llama-3.2-1B token-generation inference on AWS Trainium.

Goal: run all 16 decoder layers in a single `nki.jit` invocation, with the residual stream living in SBUF across layer boundaries and KV cache updates happening in place. Modeled after the Qwen3-MoE megakernel at `../nki-moe-megakernel/megakernels/qwen3_moe/`, minus the MoE complexity.

## Layout

```
llama.py                       # Baseline XLA NxDI subclass (works today)
llama_with_megakernel.py       # NKI-enabled subclass (stub; loaded via --enable-nki)
transformer_llama.py           # The multi-layer megakernel (stub)
nki_kernels/                   # Hand-written NKI subkernels (to be filled in)
main.py                        # CLI entry point: generate / validate / evaluate
test.py, prompts.txt, ...      # Eval helpers
```

## Environment

Tested with AWS Neuron SDK installed in `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/`.

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
```

## Running the baseline

Assumes Llama-3.2-1B weights are at `/home/ubuntu/kchern/models/llama-3.2-1b/`.

```bash
# Generate a completion
python main.py --mode generate \
  --model-path /home/ubuntu/kchern/models/llama-3.2-1b/ \
  --compiled-model-path /home/ubuntu/kchern/traced_model/llama-3.2-1b-megakernel-baseline/

# Benchmark + sanity-check (current baseline: ~545 ms p99, ~128 tok/s)
python main.py --mode evaluate_single \
  --model-path /home/ubuntu/kchern/models/llama-3.2-1b/ \
  --compiled-model-path /home/ubuntu/kchern/traced_model/llama-3.2-1b-megakernel-baseline/
```

First compile is ~10-20 min; cached after that.

## Running with the megakernel (not implemented yet)

```bash
python main.py --mode validate --enable-nki \
  --model-path /home/ubuntu/kchern/models/llama-3.2-1b/ \
  --compiled-model-path /home/ubuntu/kchern/traced_model/llama-3.2-1b-megakernel-nki/
```

When switching between baseline and megakernel, clear the compile cache:

```bash
rm -rf /home/ubuntu/kchern/traced_model/llama-3.2-1b-megakernel-nki /var/tmp/neuron-compile-cache/*
```

## References

- `nkilib/experimental/transformer/transformer_tkg.py` (in the installed Neuron SDK) — generic dense-transformer megakernel template that already matches Llama's architecture.
- `../nki-moe-megakernel/megakernels/qwen3_moe/qwen_with_megakernel.py` — the NxDI integration pattern (hijack layer 0, weakref to parent, weight gathering, KV-cache config flag).
