"""Multi-layer NKI megakernel for Llama-3.2-1B token generation.

This file owns the @nki.jit function that runs the entire 16-layer decoder
stack in one NKI invocation. It is called from llama_with_megakernel.py's
overridden decoder-layer-0 forward(), with weights gathered from the sibling
layers.

Reference template:
  /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/nkilib/experimental/transformer/transformer_tkg.py

The reference is a generic dense-transformer megakernel (no MoE, no Q/K norm,
no FP8 quant) that already matches Llama-3.2-1B's architecture. The minimum
viable version of this file is a thin wrapper that calls transformer_tkg with
Llama's dimensions and weight layout. Once that scaffold validates against the
XLA baseline, the leaf subkernels (RMSNorm, QKV+RoPE, MLP, attention compute)
can be progressively replaced with hand-written NKI implementations in
nki_kernels/.

TODO: implement transformer_llama_megakernel(X, W_qkvs, W_outs, W_gates,
W_ups, W_downs, W_gamma_qkvs, W_gamma_mlps, K_caches, V_caches, RoPE_cos,
RoPE_sin, mask_cache, mask_active, position_ids, num_layers, eps).
"""
