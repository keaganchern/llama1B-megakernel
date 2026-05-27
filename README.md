# llama-megakernel

A multi-layer NKI megakernel for Llama-3.2-1B token-generation inference on AWS Trainium 1 (NeuronCore-v2).

All 16 decoder layers run inside a single `nki.jit` invocation: the residual hidden state stays SBUF-resident across layer boundaries, per-layer weights stream in from cat-stacked HBM tensors via absolute-offset `.ap()` reads, and each layer returns fresh shared_hbm KV-cache outputs that NxDI's function-level `input_output_aliases` threads back to `past_key_values[i]` for the next decode step.

## Layout

```
llama.py                       # Baseline NxDI subclass (XLA-only, no NKI)
llama_with_megakernel.py       # NKI-enabled subclass — selected by --enable-nki
transformer_llama.py           # Multi-layer megakernel factory (1-layer + 2-layer + 16-layer)
nki_kernels/attention.py       # Fused attention block (RMSNorm + QKV + RoPE + softmax + O)
nki_kernels/mlp.py             # Fused dense MLP block (RMSNorm + SwiGLU + down)
main.py                        # CLI entry point: generate / validate / evaluate
```

## Environment

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn1
```

## Generating with the megakernel

```bash
python main.py --mode generate --enable-nki \
  --model-path /home/ubuntu/kchern/models/llama-3.2-1b/ \
  --compiled-model-path /home/ubuntu/kchern/traced_model/llama-3.2-1b-megakernel-nki/
```

To validate end-to-end against the XLA baseline:

```bash
python main.py --mode validate --enable-nki \
  --model-path /home/ubuntu/kchern/models/llama-3.2-1b/ \
  --compiled-model-path /home/ubuntu/kchern/traced_model/llama-3.2-1b-megakernel-nki/
```

When switching between baseline and megakernel, clear the compile cache:

```bash
rm -rf /home/ubuntu/kchern/traced_model/llama-3.2-1b-megakernel-nki \
       /var/tmp/neuron-compile-cache/*
```

First compile is ~10-20 min; cached after that.
