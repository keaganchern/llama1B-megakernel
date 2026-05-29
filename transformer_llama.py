"""Multi-layer NKI megakernel for Llama-3.2-1B token generation on AWS Trainium.

Runs all L decoder layers in a single nki.jit invocation. The hidden state
stays SBUF-resident across layers; only the initial HBM→SBUF load and the
final SBUF→HBM store cross the boundary. Per-layer weights stream in via
absolute-offset .ap() reads off cat-stacked HBM tensors. Each layer produces
fresh K_next_i / V_next_i shared_hbm outputs holding the input cache with the
freshly projected K / V scattered at row position_ids; NxDI's function-level
input_output_aliases threads them back to past_key_values[i].

The top-level kernel function is built by `_build_multilayer_kernel(num_layers)`
because NKI's frontend requires each KV-cache tensor to appear as its own
top-level arg — tuples / lists are not classified as HBM-binding inputs.
"""

import linecache

import nki
import nki.isa as nisa
import nki.language as nl

from nki_kernels.attention import (
    attn_block_sbuf,
    H, D, NUM_Q_HEADS, NUM_KV_HEADS, NUM_H_TILES, PMAX, S_MAX,
)
from nki_kernels.mlp import mlp_block_sbuf, NUM_I_TILES


# Llama-3.2-1B constants (re-exported).
LLAMA_1B_NUM_LAYERS = 16
LLAMA_1B_HIDDEN = H
LLAMA_1B_INTERMEDIATE = NUM_I_TILES * PMAX
LLAMA_1B_HEAD_DIM = D
LLAMA_1B_NUM_Q_HEADS = NUM_Q_HEADS
LLAMA_1B_NUM_KV_HEADS = NUM_KV_HEADS
LLAMA_1B_RMS_NORM_EPS = 1e-5


# One full Llama decoder layer applied to the SBUF-resident hidden state.
def _layer_body_sbuf(
    h_in_sb,            # [PMAX, NUM_H_TILES] bf16 SBUF
    Wq_all, Wk_all, Wv_all, Wo_all,       # cat-stacked attention weights
    gpre_all,                              # cat-stacked pre-attn gamma [L*H]
    W_gate_up_all, W_down_all,             # cat-stacked MLP weights (gate+up fused)
    gpost_all,                             # cat-stacked post-attn gamma [L*H]
    K_cache_i, V_cache_i,                  # this layer's KV cache (updated in place)
    cos, sin, position_ids,
    layer_idx,                             # Python int, selects per-layer slice
):
    """Returns a fresh SBUF tile holding the post-layer hidden state.

    The fresh K / V are scattered in place into K_cache_i / V_cache_i.
    """
    bf16 = nl.bfloat16

    attn_out_sb = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    attn_block_sbuf(
        hidden_sb=h_in_sb,
        Wq_pt=Wq_all, Wk_pt=Wk_all, Wv_pt=Wv_all, Wo=Wo_all,
        gamma_pre_attn=gpre_all,
        K_cache=K_cache_i, V_cache=V_cache_i,
        cos=cos, sin=sin,
        position_ids=position_ids,
        out_sb=attn_out_sb,
        layer_idx=layer_idx,
    )

    # Residual #1
    h1 = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(h1, h_in_sb, attn_out_sb, op=nl.add)

    # MLP
    mlp_out_sb = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    mlp_block_sbuf(
        h_sbuf=h1,
        W_gate_up_pt=W_gate_up_all, W_down_pt=W_down_all,
        gamma_post_attn=gpost_all,
        out_sb=mlp_out_sb,
        layer_idx=layer_idx,
    )

    # Residual #2
    h2 = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(h2, h1, mlp_out_sb, op=nl.add)
    return h2


# Multi-layer body. Takes Python tuples of L KV-cache tensors; the exec'd
# top-level wrapper builds those tuples from individual nki.jit args.
def _multilayer_body(
    X,
    Wq_all,                   # [L*n_q*PMAX, NUM_H_TILES*D]            cat-stacked
    Wk_all,                   # [L*n_kv*PMAX, NUM_H_TILES*D]
    Wv_all,                   # [L*n_kv*PMAX, NUM_H_TILES*D]
    Wo_all,                   # [L*n_q*D, H]
    gpre_all,                 # [L*H] cat-stacked pre-attn gamma
    Wgu_all,                  # [L*NUM_I_TILES*PMAX, 2*NUM_H_TILES*PMAX]  gate+up fused
    Wd_all,                   # [L*NUM_H_TILES*PMAX, NUM_I_TILES*PMAX]
    gpost_all,                # [L*H] cat-stacked post-attn gamma
    K_caches,                 # tuple of L HBM tensors [1, n_kv, S_MAX, D]
    V_caches,                 # tuple of L HBM tensors [1, n_kv, S_MAX, D]
    cos, sin, position_ids,
    num_layers,
):
    bf16 = nl.bfloat16

    # Initial HBM → SBUF load; the hidden state stays SBUF-resident across all
    # L layers and only crosses back to HBM at the final store.
    h_sb = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf, name="mkN_h_initial")
    nisa.dma_copy(
        dst=h_sb,
        src=X.reshape((H,)).ap(
            pattern=[[1, PMAX], [PMAX, NUM_H_TILES]], offset=0,
        ),
    )

    # Per-layer loop. Python `range` so the body is unrolled at trace time;
    # the compiler then overlaps DMAs of layer i+1's weights with compute on
    # layer i. Each layer scatters its fresh K / V IN PLACE into K_caches[i] /
    # V_caches[i]; we return those same cache tensors as pass-through outputs so
    # the scatter DMAs aren't DCE'd and the caller aliases them back to
    # past_key_values[i].
    for i in range(num_layers):
        h_sb = _layer_body_sbuf(
            h_sb,
            Wq_all, Wk_all, Wv_all, Wo_all,
            gpre_all,
            Wgu_all, Wd_all,
            gpost_all,
            K_caches[i], V_caches[i],
            cos, sin, position_ids,
            i,
        )

    # Final SBUF → HBM store.
    Y = nl.ndarray((1, 1, H), dtype=bf16, buffer=nl.shared_hbm, name="mkN_Y")
    nisa.dma_copy(
        dst=Y.reshape((H,)).ap(
            pattern=[[1, PMAX], [PMAX, NUM_H_TILES]], offset=0,
        ),
        src=h_sb,
    )
    # (Y, K_0..L-1, V_0..L-1) — the in-place-updated input caches.
    return (Y,) + tuple(K_caches) + tuple(V_caches)


@nki.jit
def transformer_llama_megakernel_passthrough(X):
    Y = nl.ndarray(X.shape, dtype=X.dtype, buffer=nl.shared_hbm, name="Y_passthrough")
    nisa.dma_copy(dst=Y, src=X)
    return Y


@nki.jit
def transformer_llama_megakernel_1layer(
    X,
    Wq_pt, Wk_pt, Wv_pt, Wo,
    gamma_pre_attn,
    W_gate_up_pt, W_down_pt,
    gamma_post_attn,
    K_cache, V_cache,
    cos, sin, position_ids,
):
    """Single-layer megakernel. Returns (Y, K_cache, V_cache) — KV updated in place."""
    bf16 = nl.bfloat16

    h_sbuf = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf, name="mk1_h_sbuf")
    nisa.dma_copy(
        dst=h_sbuf,
        src=X.reshape((H,)).ap(
            pattern=[[1, PMAX], [PMAX, NUM_H_TILES]], offset=0,
        ),
    )
    h_out = _layer_body_sbuf(
        h_sbuf,
        Wq_pt, Wk_pt, Wv_pt, Wo,
        gamma_pre_attn,
        W_gate_up_pt, W_down_pt,
        gamma_post_attn,
        K_cache, V_cache,
        cos, sin, position_ids,
        0,
    )
    Y = nl.ndarray((1, 1, H), dtype=bf16, buffer=nl.shared_hbm, name="mk1_Y")
    nisa.dma_copy(
        dst=Y.reshape((H,)).ap(
            pattern=[[1, PMAX], [PMAX, NUM_H_TILES]], offset=0,
        ),
        src=h_out,
    )
    return Y, K_cache, V_cache


# Code-gen a top-level kernel wrapper with explicit per-layer K_/V_ args
# (NKI requires individual tensor args for aliased HBM buffers).
def _build_multilayer_kernel(num_layers: int):
    k_args = [f"K_{i:02d}" for i in range(num_layers)]
    v_args = [f"V_{i:02d}" for i in range(num_layers)]
    sig = ",\n    ".join(
        ["X",
         "Wq_all", "Wk_all", "Wv_all", "Wo_all",
         "gpre_all",
         "Wgu_all", "Wd_all",
         "gpost_all"]
        + k_args + v_args
        + ["cos", "sin", "position_ids"]
    )
    fn_name = f"transformer_llama_megakernel_{num_layers}L"
    k_tuple = ", ".join(k_args)
    v_tuple = ", ".join(v_args)
    src = (
        f"def {fn_name}(\n"
        f"    {sig},\n"
        f"):\n"
        f"    K_caches = ({k_tuple},)\n"
        f"    V_caches = ({v_tuple},)\n"
        f"    return _multilayer_body(\n"
        f"        X, Wq_all, Wk_all, Wv_all, Wo_all, gpre_all,\n"
        f"        Wgu_all, Wd_all, gpost_all,\n"
        f"        K_caches, V_caches, cos, sin, position_ids,\n"
        f"        num_layers={num_layers},\n"
        f"    )\n"
    )

    fname = f"<generated-llama-megakernel-{num_layers}L>"
    # @nki.jit's parser calls inspect.getsource on the function, which reads
    # from linecache — populate it so the generated function is parseable.
    linecache.cache[fname] = (
        len(src),
        None,
        src.splitlines(keepends=True),
        fname,
    )
    code = compile(src, fname, "exec")
    ns = {"_multilayer_body": _multilayer_body}
    exec(code, ns)
    return nki.jit(ns[fn_name])


transformer_llama_megakernel_2layers = _build_multilayer_kernel(2)
transformer_llama_megakernel_16layers = _build_multilayer_kernel(16)
