"""Fused dense MLP block for Llama-3.2-1B token generation on AWS Trainium.

Computes  out = W_down( SiLU(W_gate(h_normed)) * W_up(h_normed) )  with
h_normed = RMSNorm(hidden, gamma_post_attn) for one decoder layer.

Layer weights are CAT-stacked across all 16 layers along the row dim; the
per-layer slice is selected by a `layer_idx` Python int that gets folded into
absolute DMA offsets (no compound slicing).

Tile-transposed weight layout (caller pre-shuffles):
  W_gate, W_up [I, H] → reshape(NUM_I_TILES, PMAX, NUM_H_TILES, PMAX)
                       .permute(0, 3, 2, 1)
                       .reshape(NUM_I_TILES*PMAX, NUM_H_TILES*PMAX)
  Then W_gate and W_up are cat'd along the free dim into a single
  W_gate_up [NUM_I_TILES*PMAX, 2*NUM_H_TILES*PMAX] so one DMA per m-tile
  loads both stationaries in one contiguous read.

  W_down       [H, I] → reshape(NUM_H_TILES, PMAX, NUM_I_TILES, PMAX)
                       .permute(0, 3, 2, 1)
                       .reshape(NUM_H_TILES*PMAX, NUM_I_TILES*PMAX)
"""

import nki
import nki.isa as nisa
import nki.language as nl


# Llama-3.2-1B / Trainium-1 constants.
PMAX = 128
H = 2048
I_DIM = 8192                                # intermediate dim
NUM_H_TILES = H // PMAX                     # 16
NUM_I_TILES = I_DIM // PMAX                 # 64
EPS = 1e-5


def down_weight_contig_layout(wd_tile_transposed):
    """Reorganize the tile-transposed down weight for static-DMA loading.

    Input  : [NUM_H_TILES*PMAX, NUM_I_TILES*PMAX]  (= [2048, 8192], the output of
             _tile_transpose_mlp(W_down, NUM_H_TILES, NUM_I_TILES)).
    Output : [NUM_H_TILES*2*PMAX, NUM_I_TILES*PMAX//2]  (= [4096, 4096]).

    Each per-ht [PMAX, 8192] tile (16KB/partition) is too wide for a static DMA
    descriptor, and splitting its free dim in place leaves strided halves. This
    stores the two [PMAX, 4096] halves as separate CONTIGUOUS row-blocks (block
    (ht, half) at rows [(ht*2+half)*PMAX : +PMAX]) so each kernel half-load is a
    contiguous static DMA (partition stride == free extent == 4096), matching the
    gate/up tiles that already convert. Data-identical: the kernel reassembles the
    original [PMAX, 8192] wd_t in SBUF, so the matmul is unchanged. Operates with
    tensor methods only (no torch import needed)."""
    half = NUM_I_TILES * PMAX // 2
    return (wd_tile_transposed
            .reshape(NUM_H_TILES, PMAX, 2, half)
            .permute(0, 2, 1, 3)
            .reshape(NUM_H_TILES * 2 * PMAX, half)
            .contiguous())


def mlp_block_sbuf(
    h_sbuf,            # [PMAX, NUM_H_TILES] bf16 SBUF — pre-norm input
    W_gate_up_pt,      # [L*NUM_I_TILES*PMAX, 2*NUM_H_TILES*PMAX] bf16 HBM — gate+up fused
    W_down_pt,         # [L*NUM_H_TILES*PMAX, NUM_I_TILES*PMAX] bf16 HBM, tile-transposed
    gamma_post_attn,   # [L*H] bf16 HBM (1-D)
    out_sb,            # [PMAX, NUM_H_TILES] bf16 SBUF — destination
    layer_idx=0,       # Python int, selects per-layer slice of stacked weights
):
    """SBUF-in / SBUF-out fused MLP block. Caller adds out_sb to the residual."""
    f32 = nl.float32
    bf16 = nl.bfloat16

    # 1. RMSNorm constants + gamma load (per-layer slice of [L*H] cat-stacked).
    rms_zero_bias = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.memset(rms_zero_bias, value=0.0)
    rms_ones = nl.ndarray((PMAX, PMAX), dtype=f32, buffer=nl.sbuf)
    nisa.memset(rms_ones, value=1.0)
    rms_eps_sb = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.memset(rms_eps_sb, value=EPS)

    # 2. Post-attention RMSNorm. post_attention_layernorm gamma is FOLDED into
    # the gate/up weights at load (fused RMSNorm — see
    # NeuronLlamaForCausalLMMK.convert_hf_to_neuron_state_dict), so the kernel
    # applies only the 1/rms scale — no per-layer gamma load or gamma multiply:
    #   h_normed = h_sbuf / rms(h_sbuf), cast to bf16.
    h_f32 = nl.ndarray((PMAX, NUM_H_TILES), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(h_f32, h_sbuf)
    h_sq = nl.ndarray((PMAX, NUM_H_TILES), dtype=f32, buffer=nl.sbuf)
    nisa.activation(h_sq, op=nl.square, data=h_f32, bias=rms_zero_bias)
    h_sq_partial = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_reduce(h_sq_partial, op=nl.add, data=h_sq, axis=(1,))
    h_sq_total_psum = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(h_sq_total_psum, stationary=rms_ones, moving=h_sq_partial)
    rms_inv = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.activation(rms_inv, op=nl.rsqrt, data=h_sq_total_psum, scale=(1.0 / H), bias=rms_eps_sb)
    h_normed = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(h_normed, data=h_f32, op0=nl.multiply, operand0=rms_inv)

    # 3. Gate + Up + SwiGLU: one I-output tile at a time, streaming weights.
    # For each m ∈ [0, NUM_I_TILES):
    #   mlp_inter_sb[:, m] = SiLU(h_normed @ W_gate[m]) * (h_normed @ W_up[m])
    # Row offsets use a single absolute index folding layer_idx + m. W_gate
    # and W_up are fused along the free dim of the HBM tensor: cols
    # [0:NUM_H_TILES*PMAX] hold W_gate, cols [NUM_H_TILES*PMAX:2*NUM_H_TILES*PMAX]
    # hold W_up. One DMA per m-tile loads both stationaries in a single
    # 2x-wider contiguous read.
    mlp_inter_sb = nl.ndarray((PMAX, NUM_I_TILES), dtype=bf16, buffer=nl.sbuf)

    _wgu_cols = NUM_H_TILES * PMAX           # per-weight free-dim width
    _wgu_row_stride = 2 * _wgu_cols          # fused row stride (gate cols + up cols)
    _wgu_layer_rows = NUM_I_TILES * PMAX     # rows per layer

    for m in nl.affine_range(NUM_I_TILES):
        # Single fused DMA loads [PMAX, 2*NUM_H_TILES*PMAX] = both gate and up.
        # gate stationary lives at cols [0:_wgu_cols], up at [_wgu_cols:2*_wgu_cols].
        wgu_t = nl.ndarray((PMAX, 2 * _wgu_cols), dtype=bf16, buffer=nl.sbuf)
        # Static DMA (dge_mode.none): the default swdge fragments this contiguous
        # per-tile weight read into ~6KB per-partition packets (+ 4B descriptor
        # packets), capping weight MBU at ~55%. Offsets here are all compile-time
        # constants, so a static descriptor coalesces the transfer. (AWS mlp_tkg
        # does the same.)
        nisa.dma_copy(
            dst=wgu_t,
            src=W_gate_up_pt.ap(
                pattern=[[_wgu_row_stride, PMAX], [1, _wgu_row_stride]],
                offset=(layer_idx * _wgu_layer_rows + m * PMAX) * _wgu_row_stride,
            ),
            dge_mode=nisa.dge_mode.none,
        )

        gate_psum = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.psum)
        nisa.memset(gate_psum, value=0.0)
        for h_t in nl.affine_range(NUM_H_TILES):
            nisa.nc_matmul(
                gate_psum,
                stationary=wgu_t[0:PMAX, h_t * PMAX:(h_t + 1) * PMAX],
                moving=h_normed[0:PMAX, h_t:h_t + 1],
            )

        up_psum = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.psum)
        nisa.memset(up_psum, value=0.0)
        for h_t in nl.affine_range(NUM_H_TILES):
            nisa.nc_matmul(
                up_psum,
                stationary=wgu_t[0:PMAX, _wgu_cols + h_t * PMAX:_wgu_cols + (h_t + 1) * PMAX],
                moving=h_normed[0:PMAX, h_t:h_t + 1],
            )

        gate_silu = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.sbuf)
        nisa.activation(gate_silu, op=nl.silu, data=gate_psum)

        # Materialise up_psum into SBUF — tensor_tensor needs at most one PSUM operand.
        up_sb = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(up_sb, up_psum)

        nisa.tensor_tensor(
            mlp_inter_sb[0:PMAX, m:m + 1], gate_silu, up_sb, op=nl.multiply,
        )

    # 4. Down projection. For each H-output tile ht:
    #   out_sb[:, ht] = sum_{i_t} mlp_inter_sb[:, i_t] @ W_down[ht, i_t]
    # W_down_pt is in the contiguous-split layout (down_weight_contig_layout): the
    # two [PMAX, _wd_half] halves of each ht-tile are separate contiguous row-blocks.
    # Load each half into its OWN full [PMAX, _wd_half=4096] buffer (8KB/partition) —
    # this is the size that converts to static DMA, like gate/up. A [PMAX, 8192]
    # (16KB/partition) buffer falls back to dynamic regardless of access pattern, so
    # we never materialise the full-width tile; the matmul accumulates both halves'
    # 32 i-tiles into one PSUM. half h covers i-tiles [h*32 : h*32+32].
    _wd_half = NUM_I_TILES * PMAX // 2          # 4096
    _wd_half_tiles = NUM_I_TILES // 2           # 32
    for ht in nl.affine_range(NUM_H_TILES):
        down_psum = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.psum)
        nisa.memset(down_psum, value=0.0)
        for half in range(2):
            wd_h = nl.ndarray((PMAX, _wd_half), dtype=bf16, buffer=nl.sbuf)
            _row = (layer_idx * NUM_H_TILES * 2 + ht * 2 + half) * PMAX
            nisa.dma_copy(
                dst=wd_h,
                src=W_down_pt.ap(
                    pattern=[[_wd_half, PMAX], [1, _wd_half]],
                    offset=_row * _wd_half,
                ),
                dge_mode=nisa.dge_mode.none,
            )
            for i_local in nl.affine_range(_wd_half_tiles):
                nisa.nc_matmul(
                    down_psum,
                    stationary=wd_h[0:PMAX, i_local * PMAX:(i_local + 1) * PMAX],
                    moving=mlp_inter_sb[0:PMAX,
                                        half * _wd_half_tiles + i_local:half * _wd_half_tiles + i_local + 1],
                )

        nisa.tensor_copy(out_sb[0:PMAX, ht:ht + 1], down_psum)


@nki.jit
def mlp_kernel(
    hidden_hbm,           # [1, 1, H=2048] bf16 HBM
    W_gate_up_pt,         # [I_TILES*PMAX=8192, 2*H_TILES*PMAX=4096] bf16 HBM, gate+up fused
    W_down_pt,            # [H_TILES*PMAX=2048, I_TILES*PMAX=8192] bf16 HBM
    gamma_post_attn,      # [H=2048] bf16 HBM
):
    """Standalone @nki.jit wrapper around `mlp_block_sbuf` for unit testing."""
    bf16 = nl.bfloat16

    hidden_sb = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden_hbm.reshape((H,)).ap(
            pattern=[[1, PMAX], [PMAX, NUM_H_TILES]], offset=0,
        ),
    )

    out_sb = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    mlp_block_sbuf(
        h_sbuf=hidden_sb,
        W_gate_up_pt=W_gate_up_pt, W_down_pt=W_down_pt,
        gamma_post_attn=gamma_post_attn,
        out_sb=out_sb,
    )

    out_hbm = nl.ndarray((1, 1, H), dtype=bf16, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=out_hbm.reshape((H,)).ap(
            pattern=[[1, PMAX], [PMAX, NUM_H_TILES]], offset=0,
        ),
        src=out_sb,
    )
    return out_hbm
