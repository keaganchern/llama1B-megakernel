# llama-megakernel

A multi-layer NKI megakernel for Llama-3.2-1B token-generation inference on AWS Trainium 1 (NeuronCore-v2).

All 16 decoder layers run inside a single `nki.jit` invocation: the residual hidden state stays SBUF-resident across layer boundaries, per-layer weights stream in from cat-stacked HBM tensors via absolute-offset `.ap()` reads, and each layer returns fresh shared_hbm KV-cache outputs that NxDI's function-level `input_output_aliases` threads back to `past_key_values[i]` for the next decode step.


## Results

End-to-end on Trainium 1 (trn1.2xlarge), 640 output tokens, batch_size=1, max_context_length=1024:

| Metric                | Baseline    | Megakernel  |
|-----------------------|-------------|-------------|
| Token gen latency p50 | 11.11 ms    | 11.75 ms    | 
| Token gen throughput  | 90.56 tok/s | 85.19 tok/s | 
| E2E throughput        | 76.01 tok/s | 73.44 tok/s | 



**Conditions:**
- **Baseline** = NxDI's stock Llama-3.2-1B decode path on trn1: XLA-traced through NxDI module subclasses (`NeuronAttentionBase`, `Column/RowParallelLinear`), `CustomRMSNorm` backed by the `AwsNeuronRmsNorm` XLA custom op, then compiled by `neuron-cc` with its full optimization pipeline. 
- **Megakernel** = our full multi-layer fused NKI kernel covering all 16 decoder layers in one launch (`--enable-nki=True`).


## Environment

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn1
```

## Generating tokens (qualitative check)

```bash
# Baseline
python main.py --mode generate \
  --model-path /home/ubuntu/kchern/models/llama-3.2-1b/ \
  --compiled-model-path /tmp/llama1b_baseline

# Megakernel
python main.py --mode generate --enable-nki \
  --model-path /home/ubuntu/kchern/models/llama-3.2-1b/ \
  --compiled-model-path /tmp/llama1b_nki
```

To validate megakernel output token-by-token against the baseline:

```bash
python main.py --mode validate --enable-nki \
  --model-path /home/ubuntu/kchern/models/llama-3.2-1b/ \
  --compiled-model-path /tmp/llama1b_nki
```

When switching between baseline and megakernel, clear the compile cache:

```bash
rm -rf /tmp/llama1b_baseline /tmp/llama1b_nki /var/tmp/neuron-compile-cache/*
```
