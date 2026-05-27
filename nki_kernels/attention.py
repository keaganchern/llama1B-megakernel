"""Fused attention block for Llama-3.2-1B token generation on AWS Trainium.

Computes Wo(attn(RoPE(QKV(RMSNorm(hidden))))) for one decoder layer with the
freshly projected K/V scattered into the KV cache at row position_ids.

Layer weights are CAT-stacked across all 16 layers along the row dim; the
per-layer slice is selected by a `layer_idx` Python int that gets folded into
absolute DMA offsets (no compound slicing).
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl


# Llama-3.2-1B / Trainium-1 constants.
PMAX = 128
H = 2048
D = 64
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
GQA = NUM_Q_HEADS // NUM_KV_HEADS                  # 4 query heads per kv-head
NUM_H_TILES = H // PMAX                            # 16
HALF_D = D // 2                                    # RoPE split point
S_MAX = 1024                                       # KV cache sequence length
S_TILE = PMAX                                      # flash-attention tile size = 128
NUM_S_TILES = S_MAX // S_TILE                      # 8 cache tiles
EPS = 1e-5
INV_SQRT_D = float(1.0 / math.sqrt(D))             # softmax scale
MASK_NEG_INF = -3.4028234663852886e38              # torch.finfo(fp32).min


def attn_block_sbuf(
    hidden_sb,            # [PMAX, NUM_H_TILES] bf16 SBUF — pre-norm input
    Wq_pt,                # [L*n_q*PMAX, NUM_H_TILES*D] bf16 HBM, tile-transposed
    Wk_pt,                # [L*n_kv*PMAX, NUM_H_TILES*D] bf16 HBM, tile-transposed
    Wv_pt,                # [L*n_kv*PMAX, NUM_H_TILES*D] bf16 HBM, tile-transposed
    Wo,                   # [L*n_q*D, H] bf16 HBM (= nn.Linear.weight.T)
    gamma_pre_attn,       # [L*H] bf16 HBM (1-D)
    K_cache,              # [1, n_kv, S_MAX, D] bf16 HBM — read only
    V_cache,              # [1, n_kv, S_MAX, D] bf16 HBM — read only
    cos,                  # [1, 1, D] bf16 HBM, indexed at position_ids
    sin,                  # [1, 1, D] bf16 HBM
    position_ids,         # [1, 1] int32 HBM
    out_sb,               # [PMAX, NUM_H_TILES] bf16 SBUF — destination
    K_next,               # [1, n_kv, S_MAX, D] bf16 shared_hbm — output
    V_next,               # [1, n_kv, S_MAX, D] bf16 shared_hbm — output
    layer_idx=0,          # Python int, selects per-layer slice of stacked weights
):
    """SBUF-in / SBUF-out fused attention block.

    Step 9 runs flash-attention with online softmax over NUM_S_TILES cache
    tiles of S_TILE=PMAX=128 positions each, plus one active-slot iteration
    for the freshly projected K/V at virtual position S_MAX. The cache mask
    is built per-tile from (position_ids, t*S_TILE); the active slot is
    always unmasked.

    K_next / V_next are written as a full copy of the input cache plus the
    freshly projected K / V scattered at row position_ids; the caller aliases
    them back to past_key_values. Tried in-place aliasing of K_cache instead —
    on trn1 the compiler inserts a protective save/restore round-trip that
    makes HBM traffic worse, not better.
    """
    f32 = nl.float32
    bf16 = nl.bfloat16

    # 1. RMSNorm constants.
    rms_zero_bias = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.memset(rms_zero_bias, value=0.0)
    rms_ones = nl.ndarray((PMAX, PMAX), dtype=f32, buffer=nl.sbuf)
    nisa.memset(rms_ones, value=1.0)
    rms_eps_sb = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.memset(rms_eps_sb, value=EPS)

    # Load this layer's input_layernorm gamma from the [L*H] cat-stacked
    # 1-D tensor.
    gamma_sb = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=gamma_sb,
        src=gamma_pre_attn.reshape((gamma_pre_attn.shape[0],)).ap(
            pattern=[[1, PMAX], [PMAX, NUM_H_TILES]], offset=layer_idx * H,
        ),
    )

    # 2. Pre-attention RMSNorm: h_all = RMSNorm(hidden) * gamma.
    h_f32 = nl.ndarray((PMAX, NUM_H_TILES), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(h_f32, hidden_sb)
    h_sq = nl.ndarray((PMAX, NUM_H_TILES), dtype=f32, buffer=nl.sbuf)
    nisa.activation(h_sq, op=nl.square, data=h_f32, bias=rms_zero_bias)
    h_sq_partial = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_reduce(h_sq_partial, op=nl.add, data=h_sq, axis=(1,))
    h_sq_total_psum = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(h_sq_total_psum, stationary=rms_ones, moving=h_sq_partial)
    rms_inv = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.activation(rms_inv, op=nl.rsqrt, data=h_sq_total_psum, scale=(1.0 / H), bias=rms_eps_sb)
    h_scaled = nl.ndarray((PMAX, NUM_H_TILES), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(h_scaled, data=h_f32, op0=nl.multiply, operand0=rms_inv)
    h_all = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(h_all, h_scaled, gamma_sb, op=nl.multiply)

    # 3. Load Wk / Wv for THIS layer's kv-heads. Each row offset is computed
    # from a single absolute index `(layer_idx*n_kv + kv)*PMAX` to avoid
    # compound HBM slicing.
    wk_sb_per_kv = [None] * NUM_KV_HEADS
    wv_sb_per_kv = [None] * NUM_KV_HEADS
    _wk_row_stride = NUM_H_TILES * D
    for kv in range(NUM_KV_HEADS):
        _row0 = (layer_idx * NUM_KV_HEADS + kv) * PMAX
        wk_t = nl.ndarray((PMAX, NUM_H_TILES * D), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=wk_t,
            src=Wk_pt.ap(
                pattern=[[_wk_row_stride, PMAX], [1, _wk_row_stride]],
                offset=_row0 * _wk_row_stride,
            ),
        )
        wk_sb_per_kv[kv] = wk_t
        wv_t = nl.ndarray((PMAX, NUM_H_TILES * D), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=wv_t,
            src=Wv_pt.ap(
                pattern=[[_wk_row_stride, PMAX], [1, _wk_row_stride]],
                offset=_row0 * _wk_row_stride,
            ),
        )
        wv_sb_per_kv[kv] = wv_t

    # 4. Load cos / sin and position_ids.
    cos_bf16 = nl.ndarray((D, 1), dtype=bf16, buffer=nl.sbuf)
    sin_bf16 = nl.ndarray((D, 1), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_bf16, src=cos.reshape((D, 1)))
    nisa.dma_copy(dst=sin_bf16, src=sin.reshape((D, 1)))
    # fp32 copies needed because tensor_scalar's operand0 must be fp32 when
    # broadcasting cos / sin across the GQA free dim in the Q RoPE.
    cos_f32 = nl.ndarray((D, 1), dtype=f32, buffer=nl.sbuf)
    sin_f32 = nl.ndarray((D, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(cos_f32, cos_bf16)
    nisa.tensor_copy(sin_f32, sin_bf16)

    pos_i32 = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pos_i32, src=position_ids.reshape((1, 1)))
    pos_u32 = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(pos_u32, pos_i32)
    pos_f32 = nl.ndarray((1, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(pos_f32, pos_i32)

    # 5. K + V projection per kv-head; K then gets RoPE.
    k_active_per_kv = [None] * NUM_KV_HEADS    # [D, 1] bf16 post-RoPE
    v_active_per_kv = [None] * NUM_KV_HEADS    # [D, 1] bf16

    for kv in range(NUM_KV_HEADS):
        # K projection: stationary [PMAX K, D M_free] -> PSUM [D, 1].
        k_psum = nl.ndarray((D, 1), dtype=f32, buffer=nl.psum)
        nisa.memset(k_psum, value=0.0)
        for h_t in nl.affine_range(NUM_H_TILES):
            nisa.nc_matmul(
                k_psum,
                stationary=wk_sb_per_kv[kv][0:PMAX, h_t * D:(h_t + 1) * D],
                moving=h_all[0:PMAX, h_t:h_t + 1],
            )
        k_vec_bf16 = nl.ndarray((D, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(k_vec_bf16, k_psum)

        # K RoPE — split-half rotation:  k_rope = k*cos + rotate_half(k)*sin
        # rotate_half(k) = concat(-k[D/2:], k[:D/2])
        neg_k_upper = nl.ndarray((HALF_D, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(neg_k_upper, data=k_vec_bf16[HALF_D:D, 0:1], op0=nl.multiply, operand0=-1.0)
        rot_k = nl.ndarray((D, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(rot_k[0:HALF_D, 0:1], neg_k_upper)
        nisa.tensor_copy(rot_k[HALF_D:D, 0:1], k_vec_bf16[0:HALF_D, 0:1])

        k_cos = nl.ndarray((D, 1), dtype=bf16, buffer=nl.sbuf)
        k_sin_part = nl.ndarray((D, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(k_cos, k_vec_bf16, cos_bf16, op=nl.multiply)
        nisa.tensor_tensor(k_sin_part, rot_k, sin_bf16, op=nl.multiply)
        k_rope = nl.ndarray((D, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(k_rope, k_cos, k_sin_part, op=nl.add)
        k_active_per_kv[kv] = k_rope

        # V projection (no RoPE).
        v_psum = nl.ndarray((D, 1), dtype=f32, buffer=nl.psum)
        nisa.memset(v_psum, value=0.0)
        for h_t in nl.affine_range(NUM_H_TILES):
            nisa.nc_matmul(
                v_psum,
                stationary=wv_sb_per_kv[kv][0:PMAX, h_t * D:(h_t + 1) * D],
                moving=h_all[0:PMAX, h_t:h_t + 1],
            )
        v_vec_bf16 = nl.ndarray((D, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(v_vec_bf16, v_psum)
        v_active_per_kv[kv] = v_vec_bf16

    # 6. Q projection + RoPE, per kv-group of GQA q-heads → [D, GQA] bf16.
    wq_sb_per_q = [None] * NUM_Q_HEADS
    _wq_row_stride = NUM_H_TILES * D
    for q_h in range(NUM_Q_HEADS):
        _row0 = (layer_idx * NUM_Q_HEADS + q_h) * PMAX
        wq_t = nl.ndarray((PMAX, NUM_H_TILES * D), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=wq_t,
            src=Wq_pt.ap(
                pattern=[[_wq_row_stride, PMAX], [1, _wq_row_stride]],
                offset=_row0 * _wq_row_stride,
            ),
        )
        wq_sb_per_q[q_h] = wq_t

    q_rope_per_kv = [None] * NUM_KV_HEADS   # [D, GQA] bf16

    for kv in range(NUM_KV_HEADS):
        q_psums = []
        for g in range(GQA):
            q_h = kv * GQA + g
            q_p = nl.ndarray((D, 1), dtype=f32, buffer=nl.psum)
            nisa.memset(q_p, value=0.0)
            q_psums.append(q_p)

        for h_t in nl.affine_range(NUM_H_TILES):
            for g in range(GQA):
                q_h = kv * GQA + g
                nisa.nc_matmul(
                    q_psums[g],
                    stationary=wq_sb_per_q[q_h][0:PMAX, h_t * D:(h_t + 1) * D],
                    moving=h_all[0:PMAX, h_t:h_t + 1],
                )

        # Pack GQA q-heads into [D, GQA] bf16.
        q_packed = nl.ndarray((D, GQA), dtype=bf16, buffer=nl.sbuf)
        for g in range(GQA):
            nisa.tensor_copy(q_packed[0:D, g:g + 1], q_psums[g])

        # Q RoPE — split-half rotation; cos / sin broadcast across the GQA free
        # dim via tensor_scalar (operand0 is a per-partition vector).
        neg_q_upper = nl.ndarray((HALF_D, GQA), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(
            neg_q_upper, data=q_packed[HALF_D:D, 0:GQA], op0=nl.multiply, operand0=-1.0,
        )
        rot_q = nl.ndarray((D, GQA), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(rot_q[0:HALF_D, 0:GQA], neg_q_upper)
        nisa.tensor_copy(rot_q[HALF_D:D, 0:GQA], q_packed[0:HALF_D, 0:GQA])

        q_cos = nl.ndarray((D, GQA), dtype=bf16, buffer=nl.sbuf)
        q_sin_part = nl.ndarray((D, GQA), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(q_cos, data=q_packed, op0=nl.multiply, operand0=cos_f32)
        nisa.tensor_scalar(q_sin_part, data=rot_q, op0=nl.multiply, operand0=sin_f32)
        q_rope = nl.ndarray((D, GQA), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(q_rope, q_cos, q_sin_part, op=nl.add)
        q_rope_per_kv[kv] = q_rope

    # 7. Transpose freshly projected K and V per kv-head from [D, 1] to [1, D].
    # These are used in step 9 (virtual-append slot 64) and in step 12 (cache
    # scatter into K_next / V_next at row position_ids). nc_transpose on NCv2
    # requires the destination PSUM to be fp32.
    k_active_T_per_kv = [None] * NUM_KV_HEADS
    v_active_T_per_kv = [None] * NUM_KV_HEADS

    for kv in range(NUM_KV_HEADS):
        k_T_psum = nl.ndarray((1, D), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(k_T_psum, k_active_per_kv[kv])
        k_T_sb = nl.ndarray((1, D), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(k_T_sb, k_T_psum)
        k_active_T_per_kv[kv] = k_T_sb

        v_T_psum = nl.ndarray((1, D), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(v_T_psum, v_active_per_kv[kv])
        v_T_sb = nl.ndarray((1, D), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(v_T_sb, v_T_psum)
        v_active_T_per_kv[kv] = v_T_sb

    K_cache_flat = K_cache.reshape((NUM_KV_HEADS * S_MAX, D))
    V_cache_flat = V_cache.reshape((NUM_KV_HEADS * S_MAX, D))

    # 8. Precompute per-tile mask building blocks.
    #   iota_st[i] = i for i in 0..S_TILE-1 (S_TILE partitions, 1 free).
    #   neg_pos    = -pos (scalar).
    # The per-tile mask is built inside the loop as:
    #   delta[i] = iota[i] + 1 - pos + t*S_TILE
    #   mask[i]  = MASK_NEG_INF * min(relu(delta), 1)  (= -inf iff t*S_TILE + i >= pos)
    iota_st = nl.ndarray((S_TILE, 1), dtype=f32, buffer=nl.sbuf)
    nisa.iota(iota_st, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    neg_pos = nl.ndarray((1, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(neg_pos, data=pos_f32, op0=nl.multiply, operand0=-1.0)

    # 9. Per-kv-head flash-attention with online softmax over S_TILE=PMAX chunks.
    # State per kv-head (in [GQA, *] layout so rescale [GQA, 1] is a per-partition op):
    #   running_max   [GQA, 1]  init -inf
    #   running_denom [GQA, 1]  init 0
    #   running_out   [GQA, D]  init 0
    # For each cache tile t in 0..NUM_S_TILES:
    #   compute score, mask, online-update max/denom/out.
    # Then handle the active slot (S_MAX, always unmasked) as a single-position update.
    # Final: attn_out = running_out / running_denom, transposed to [D, GQA] for step 10.
    attn_out_per_kv = [None] * NUM_KV_HEADS

    for kv in range(NUM_KV_HEADS):
        running_max = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(running_max, value=MASK_NEG_INF)
        running_denom = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(running_denom, value=0.0)
        running_out = nl.ndarray((GQA, D), dtype=f32, buffer=nl.sbuf)
        nisa.memset(running_out, value=0.0)

        # Python `range` to enable first-tile fast-path branching at compile time.
        for t in range(NUM_S_TILES):
            # ----- Load K tile [D, S_TILE] via dma_transpose -----
            k_tile = nl.ndarray((D, S_TILE), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_transpose(
                dst=k_tile,
                src=K_cache_flat[kv * S_MAX + t * S_TILE:kv * S_MAX + (t + 1) * S_TILE, :],
            )

            # ----- score = k_tile.T @ q_rope → PSUM [S_TILE, GQA] -----
            score_psum = nl.ndarray((S_TILE, GQA), dtype=f32, buffer=nl.psum)
            nisa.memset(score_psum, value=0.0)
            nisa.nc_matmul(score_psum, stationary=k_tile, moving=q_rope_per_kv[kv])

            # ----- Build mask [S_TILE, 1], scale + mask in [S_TILE, GQA] -----
            tile_off_scalar = nl.ndarray((1, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(tile_off_scalar, data=neg_pos, op0=nl.add,
                               operand0=float(t * S_TILE + 1))
            tile_off_psum = nl.ndarray((S_TILE, 1), dtype=f32, buffer=nl.psum)
            nisa.nc_transpose(tile_off_psum, tile_off_scalar.ap([[1, 1], [0, S_TILE]], offset=0))
            tile_off_bcast = nl.ndarray((S_TILE, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(tile_off_bcast, tile_off_psum)
            delta = nl.ndarray((S_TILE, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(delta, iota_st, tile_off_bcast, op=nl.add)
            relu_delta = nl.ndarray((S_TILE, 1), dtype=f32, buffer=nl.sbuf)
            nisa.activation(relu_delta, op=nl.relu, data=delta)
            clamped = nl.ndarray((S_TILE, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(clamped, data=relu_delta, op0=nl.minimum, operand0=1.0)
            mask_tile = nl.ndarray((S_TILE, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(mask_tile, data=clamped, op0=nl.multiply, operand0=MASK_NEG_INF)

            score_scaled = nl.ndarray((S_TILE, GQA), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(score_scaled, data=score_psum, op0=nl.multiply, operand0=INV_SQRT_D)
            score_masked = nl.ndarray((S_TILE, GQA), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(score_masked, data=score_scaled, op0=nl.add, operand0=mask_tile)

            # ----- Transpose to [GQA, S_TILE] layout; all softmax math stays here -----
            # so neg_max becomes a per-partition bias for fused exp(score - max).
            score_T_psum = nl.ndarray((GQA, S_TILE), dtype=f32, buffer=nl.psum)
            nisa.nc_transpose(score_T_psum, score_masked)
            score_T = nl.ndarray((GQA, S_TILE), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(score_T, score_T_psum)
            tile_max = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_reduce(tile_max, op=nl.maximum, data=score_T, axis=(1,))

            if t == 0:
                # First-tile fast-path: running_max = tile_max, no rescale.
                # tile_exp_T = exp(score_T - tile_max).
                neg_tile_max = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(neg_tile_max, data=tile_max, op0=nl.multiply, operand0=-1.0)
                tile_exp_T = nl.ndarray((GQA, S_TILE), dtype=f32, buffer=nl.sbuf)
                nisa.activation(tile_exp_T, op=nl.exp, data=score_T, bias=neg_tile_max)
                tile_denom = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_reduce(tile_denom, op=nl.add, data=tile_exp_T, axis=(1,))

                # Establish state directly from this tile (no scaling, running_* is 0).
                nisa.tensor_copy(running_max, tile_max)
                nisa.tensor_copy(running_denom, tile_denom)
                # running_out is set after the V matmul below.
                rescale = None
            else:
                # Subsequent tiles: rescale prior state by exp(running_max - new_max).
                new_max = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(new_max, running_max, tile_max, op=nl.maximum)
                # Fused: rescale = exp(running_max - new_max) = exp(scale * new_max + bias=running_max).
                rescale = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
                nisa.activation(rescale, op=nl.exp, data=new_max,
                                bias=running_max, scale=-1.0)
                neg_new_max = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(neg_new_max, data=new_max, op0=nl.multiply, operand0=-1.0)
                # Fused: tile_exp_T = exp(score_T - new_max), per-partition bias.
                tile_exp_T = nl.ndarray((GQA, S_TILE), dtype=f32, buffer=nl.sbuf)
                nisa.activation(tile_exp_T, op=nl.exp, data=score_T, bias=neg_new_max)
                tile_denom = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_reduce(tile_denom, op=nl.add, data=tile_exp_T, axis=(1,))

                # running_denom = running_denom * rescale + tile_denom.
                rd_scaled = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(rd_scaled, running_denom, rescale, op=nl.multiply)
                nisa.tensor_tensor(running_denom, rd_scaled, tile_denom, op=nl.add)
                # running_max := new_max
                nisa.tensor_copy(running_max, new_max)

            # ----- Load V tile [S_TILE, D] from HBM -----
            v_tile = nl.ndarray((S_TILE, D), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=v_tile,
                src=V_cache_flat[kv * S_MAX + t * S_TILE:kv * S_MAX + (t + 1) * S_TILE, :],
            )

            # ----- Transpose tile_exp_T [GQA, S_TILE] → bf16 [S_TILE, GQA] for matmul -----
            tile_exp_pt_psum = nl.ndarray((S_TILE, GQA), dtype=f32, buffer=nl.psum)
            nisa.nc_transpose(tile_exp_pt_psum, tile_exp_T)
            tile_exp_bf16 = nl.ndarray((S_TILE, GQA), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(tile_exp_bf16, tile_exp_pt_psum)

            # ----- tile_out [GQA, D] = tile_exp.T @ v_tile -----
            tile_out_psum = nl.ndarray((GQA, D), dtype=f32, buffer=nl.psum)
            nisa.memset(tile_out_psum, value=0.0)
            nisa.nc_matmul(tile_out_psum, stationary=tile_exp_bf16, moving=v_tile)

            if t == 0:
                # running_out = tile_out (no prior state to rescale).
                nisa.tensor_copy(running_out, tile_out_psum)
            else:
                # running_out = running_out * rescale + tile_out.
                ro_scaled = nl.ndarray((GQA, D), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(ro_scaled, data=running_out, op0=nl.multiply, operand0=rescale)
                nisa.tensor_tensor(running_out, ro_scaled, tile_out_psum, op=nl.add)

        # ----- Active slot: one more position at virtual slot S_MAX, always unmasked -----
        # Same online-softmax pattern with S=1.
        # score_active [1, GQA] = k_active.T @ q_rope
        score_active_psum = nl.ndarray((1, GQA), dtype=f32, buffer=nl.psum)
        nisa.memset(score_active_psum, value=0.0)
        nisa.nc_matmul(score_active_psum,
                       stationary=k_active_per_kv[kv],
                       moving=q_rope_per_kv[kv])
        score_active_scaled = nl.ndarray((1, GQA), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(score_active_scaled, data=score_active_psum,
                           op0=nl.multiply, operand0=INV_SQRT_D)

        # tile_max_act [GQA, 1] = transpose([1, GQA])
        tile_max_act_psum = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(tile_max_act_psum, score_active_scaled)
        tile_max_act = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(tile_max_act, tile_max_act_psum)

        new_max_act = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(new_max_act, running_max, tile_max_act, op=nl.maximum)
        # Fused: rescale_act = exp(running_max - new_max_act).
        rescale_act = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.activation(rescale_act, op=nl.exp, data=new_max_act,
                        bias=running_max, scale=-1.0)
        # Fused: tile_exp_act = exp(tile_max_act - new_max_act).
        tile_exp_act = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.activation(tile_exp_act, op=nl.exp, data=new_max_act,
                        bias=tile_max_act, scale=-1.0)

        # running_denom = running_denom * rescale_act + tile_exp_act
        rd_scaled_act = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(rd_scaled_act, running_denom, rescale_act, op=nl.multiply)
        nisa.tensor_tensor(running_denom, rd_scaled_act, tile_exp_act, op=nl.add)

        # tile_out_act [GQA, D] = tile_exp_act @ v_active.T (rank-1 outer product via matmul with K=1)
        # stationary [1, GQA]: tile_exp_act.T
        tile_exp_act_T_psum = nl.ndarray((1, GQA), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(tile_exp_act_T_psum, tile_exp_act)
        tile_exp_act_T_sb = nl.ndarray((1, GQA), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(tile_exp_act_T_sb, tile_exp_act_T_psum)
        # moving [1, D]: v_active_T_per_kv[kv]
        tile_out_act_psum = nl.ndarray((GQA, D), dtype=f32, buffer=nl.psum)
        nisa.memset(tile_out_act_psum, value=0.0)
        nisa.nc_matmul(tile_out_act_psum,
                       stationary=tile_exp_act_T_sb,
                       moving=v_active_T_per_kv[kv])

        # running_out = running_out * rescale_act + tile_out_act
        ro_scaled_act = nl.ndarray((GQA, D), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(ro_scaled_act, data=running_out, op0=nl.multiply, operand0=rescale_act)
        nisa.tensor_tensor(running_out, ro_scaled_act, tile_out_act_psum, op=nl.add)

        # ----- Final divide: attn_out_qd = running_out / running_denom -----
        inv_denom = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(inv_denom, running_denom)
        attn_out_qd = nl.ndarray((GQA, D), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(attn_out_qd, data=running_out, op0=nl.multiply, operand0=inv_denom)

        # ----- Transpose [GQA, D] → [D, GQA] for step 10 -----
        attn_kv_psum = nl.ndarray((D, GQA), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(attn_kv_psum, attn_out_qd)
        attn_kv = nl.ndarray((D, GQA), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(attn_kv, attn_kv_psum)
        attn_out_per_kv[kv] = attn_kv

    # 10. Pack per-kv-head outputs into a [D, NUM_Q_HEADS] tile.
    attn_out_all = nl.ndarray((D, NUM_Q_HEADS), dtype=bf16, buffer=nl.sbuf)
    for kv in range(NUM_KV_HEADS):
        nisa.tensor_copy(
            attn_out_all[0:D, kv * GQA:(kv + 1) * GQA],
            attn_out_per_kv[kv],
        )

    # 11. O projection: out = attn_out_all @ Wo.
    # Wo is [n_q*D, H]; per q-head h_q the slice Wo[h_q*D:(h_q+1)*D, :] is its
    # (D, H) block. Tile across n_q heads in D-wide K chunks and across H in
    # PMAX-wide N chunks, accumulating into one PSUM per H-tile.
    wo_sb_per_q = [None] * NUM_Q_HEADS
    _wo_row_stride = H
    for h_q in range(NUM_Q_HEADS):
        _row0 = (layer_idx * NUM_Q_HEADS + h_q) * D
        wo_t = nl.ndarray((D, H), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=wo_t,
            src=Wo.ap(
                pattern=[[_wo_row_stride, D], [1, _wo_row_stride]],
                offset=_row0 * _wo_row_stride,
            ),
        )
        wo_sb_per_q[h_q] = wo_t

    res_psums = []
    for ht in range(NUM_H_TILES):
        rp = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.psum)
        nisa.memset(rp, value=0.0)
        res_psums.append(rp)

    for h_q in range(NUM_Q_HEADS):
        for ht in range(NUM_H_TILES):
            nisa.nc_matmul(
                res_psums[ht],
                stationary=attn_out_all[0:D, h_q:h_q + 1],
                moving=wo_sb_per_q[h_q][0:D, ht * PMAX:(ht + 1) * PMAX],
            )

    # Transpose each PSUM [1, PMAX] → SBUF column [PMAX, 1] in out_sb.
    for ht in range(NUM_H_TILES):
        row_sb = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(row_sb, res_psums[ht])
        col_T_psum = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(col_T_psum, row_sb)
        nisa.tensor_copy(out_sb[0:PMAX, ht:ht + 1], col_T_psum)

    # 12. Build K_next / V_next: full copy of the input cache, then scatter
    # the freshly projected K / V per kv-head at row (kv*S_MAX + position_ids).
    # The caller aliases K_next / V_next back to past_key_values for the next step.
    K_next_flat = K_next.reshape((NUM_KV_HEADS * S_MAX, D))
    V_next_flat = V_next.reshape((NUM_KV_HEADS * S_MAX, D))
    nisa.dma_copy(dst=K_next_flat, src=K_cache_flat)
    nisa.dma_copy(dst=V_next_flat, src=V_cache_flat)

    # Coalesced source: pack all NUM_KV_HEADS active rows into a single SBUF tile
    # [NUM_KV_HEADS partition, D free] and issue ONE dma_copy per K / V with a
    # multi-row .ap() pattern striding through kv-head sections at S_MAX * D.
    k_active_pack_perD = nl.ndarray((D, NUM_KV_HEADS), dtype=bf16, buffer=nl.sbuf)
    v_active_pack_perD = nl.ndarray((D, NUM_KV_HEADS), dtype=bf16, buffer=nl.sbuf)
    for kv in range(NUM_KV_HEADS):
        nisa.tensor_copy(k_active_pack_perD[0:D, kv:kv + 1], k_active_per_kv[kv])
        nisa.tensor_copy(v_active_pack_perD[0:D, kv:kv + 1], v_active_per_kv[kv])

    k_active_pack_T_psum = nl.ndarray((NUM_KV_HEADS, D), dtype=f32, buffer=nl.psum)
    v_active_pack_T_psum = nl.ndarray((NUM_KV_HEADS, D), dtype=f32, buffer=nl.psum)
    nisa.nc_transpose(k_active_pack_T_psum, k_active_pack_perD)
    nisa.nc_transpose(v_active_pack_T_psum, v_active_pack_perD)
    k_active_pack_T = nl.ndarray((NUM_KV_HEADS, D), dtype=bf16, buffer=nl.sbuf)
    v_active_pack_T = nl.ndarray((NUM_KV_HEADS, D), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(k_active_pack_T, k_active_pack_T_psum)
    nisa.tensor_copy(v_active_pack_T, v_active_pack_T_psum)

    # scalar_offset=pos shifts each kv-head's row by pos*D within its section
    # (indirect_dim=0 → indirect_stride = D for K_next_flat = [n_kv*S_MAX, D]).
    nisa.dma_copy(
        dst=K_next_flat.ap(
            pattern=[[S_MAX * D, NUM_KV_HEADS], [1, D]],
            offset=0,
            scalar_offset=pos_u32,
            indirect_dim=0,
        ),
        src=k_active_pack_T,
    )
    nisa.dma_copy(
        dst=V_next_flat.ap(
            pattern=[[S_MAX * D, NUM_KV_HEADS], [1, D]],
            offset=0,
            scalar_offset=pos_u32,
            indirect_dim=0,
        ),
        src=v_active_pack_T,
    )


@nki.jit
def attn_kernel(
    hidden_hbm,           # [1, 1, H=2048] bf16 HBM
    Wq_pt,                # [n_q*PMAX=4096, n_h_tiles*D=1024] bf16 HBM
    Wk_pt,                # [n_kv*PMAX=1024, n_h_tiles*D=1024] bf16 HBM
    Wv_pt,                # [n_kv*PMAX=1024, n_h_tiles*D=1024] bf16 HBM
    Wo,                   # [n_q*D=2048, H=2048] bf16 HBM
    gamma_pre_attn,       # [H=2048] bf16 HBM
    K_cache,              # [1, n_kv=8, S_MAX=1024, D=64] bf16 HBM, read-only input
    V_cache,              # [1, n_kv=8, S_MAX=1024, D=64] bf16 HBM, read-only input
    cos,                  # [1, 1, D=64] bf16 HBM
    sin,                  # [1, 1, D=64] bf16 HBM
    position_ids,         # [1, 1] int32 HBM
):
    """Standalone @nki.jit wrapper around `attn_block_sbuf` for unit testing.

    Returns (out_hbm, K_next, V_next) where K_next / V_next are fresh
    shared_hbm tensors holding the input cache with the active K / V scattered
    at row position_ids.
    """
    bf16 = nl.bfloat16

    hidden_sb = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden_hbm.reshape((H,)).ap(
            pattern=[[1, PMAX], [PMAX, NUM_H_TILES]], offset=0,
        ),
    )

    K_next = nl.ndarray((1, NUM_KV_HEADS, S_MAX, D), dtype=bf16, buffer=nl.shared_hbm)
    V_next = nl.ndarray((1, NUM_KV_HEADS, S_MAX, D), dtype=bf16, buffer=nl.shared_hbm)
    out_sb = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    attn_block_sbuf(
        hidden_sb=hidden_sb,
        Wq_pt=Wq_pt, Wk_pt=Wk_pt, Wv_pt=Wv_pt, Wo=Wo,
        gamma_pre_attn=gamma_pre_attn,
        K_cache=K_cache, V_cache=V_cache,
        cos=cos, sin=sin,
        position_ids=position_ids,
        out_sb=out_sb,
        K_next=K_next, V_next=V_next,
    )

    out_hbm = nl.ndarray((1, 1, H), dtype=bf16, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=out_hbm.reshape((H,)).ap(
            pattern=[[1, PMAX], [PMAX, NUM_H_TILES]], offset=0,
        ),
        src=out_sb,
    )
    return out_hbm, K_next, V_next
