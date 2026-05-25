"""Multi-layer NKI megakernel for Llama-3.2-1B token generation on trn1.

STATUS: Passthrough stub. The megakernel currently DMAs the input hidden state
straight to the output without any compute. Tokens produced by the model will
be garbage. This is intentional for milestone 1: validate the NxDI integration
end-to-end (subclass loaded, weakref installed, layer-0 hijack fires, weights
collected, kernel compiles, kernel called, output flows back through NxDI)
before investing in real kernel work.

DESIGN: mirrors transformer_tkg's HBM-path branch (single core, no SBUF
residual yet) using gen2-compatible building blocks from nkilib/core/:

    core/qkv/qkv_tkg               -- RMSNorm + QKV projection
    core/embeddings/rope.RoPE      -- RoPE
    core/attention/attention_tkg   -- Q@K + softmax + @V + KV cache update
    core/output_projection/...     -- output projection
    core/mlp/mlp                   -- RMSNorm + gate/up/down + SiLU (fused)

DO NOT call nkilib/experimental/transformer/transformer_tkg or
attention_block_tkg from here -- both hard-fail at compile time with
kernel_assert(nc_version >= gen3). See .claude/skills/TRN1_NOTES.md.
"""

import nki
import nki.isa as nisa
import nki.language as nl


# Llama-3.2-1B architecture constants. Hardcoded for now -- this kernel is
# specific to this model and any other Llama variant would need its own.
LLAMA_1B_NUM_LAYERS = 16
LLAMA_1B_HIDDEN = 2048
LLAMA_1B_INTERMEDIATE = 8192
LLAMA_1B_HEAD_DIM = 64
LLAMA_1B_NUM_Q_HEADS = 32
LLAMA_1B_NUM_KV_HEADS = 8
LLAMA_1B_RMS_NORM_EPS = 1e-5


# Platform target is set via NEURON_PLATFORM_TARGET_OVERRIDE env var, not
# nki.jit(platform_target=...) (that kwarg was deprecated in newer SDKs).
# main.py / the run script must set NEURON_PLATFORM_TARGET_OVERRIDE=trn1.
@nki.jit
def transformer_llama_megakernel_passthrough(X):
    """Passthrough megakernel: hidden state in -> hidden state out, byte-identical.

    Used to validate NxDI integration plumbing without confounding NKI kernel
    bugs. Once layer-0 hijack + weight gather + KV cache wiring are confirmed
    to work, the body of this function gets replaced with the real multi-layer
    fused transformer.

    Args:
        X: Input hidden state of shape [B, S, H] in HBM (where H = 2048).
           At decode time, S == 1.

    Returns:
        Y: Output hidden state of shape [B, S, H] in HBM. Contents identical
           to X (this is a passthrough).
    """
    Y = nl.ndarray(X.shape, dtype=X.dtype, buffer=nl.shared_hbm, name="Y_passthrough")
    nisa.dma_copy(dst=Y, src=X)
    return Y
