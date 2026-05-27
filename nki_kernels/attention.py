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
S_MAX = 64                                         # KV cache sequence length
S_ATT = S_MAX + 1                                  # cache (64) + active (1)
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

    Step 9's softmax runs over S_ATT = S_MAX + 1 slots: 64 cached positions
    plus one virtual slot for this step's freshly projected K/V. The causal
    mask leaves slot s unmasked iff s <= position_ids, naturally including
    the active slot.

    K_next / V_next are written as a full copy of the input cache plus the
    freshly projected K/V scattered at row position_ids; the caller aliases
    them back to the KV-cache parameters.
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

    # 8. Causal mask [S_ATT, 1]: cache slots (0..S_MAX-1) are -inf iff s >= pos
    # (cache row s holds the prior step's K/V; it is valid only for s < pos).
    # The active slot S_MAX is always 0 (unmasked).
    causal_mask = nl.ndarray((S_ATT, 1), dtype=f32, buffer=nl.sbuf)
    nisa.memset(causal_mask, value=0.0)

    # Build the cache portion: delta = (s + 1) - pos > 0 iff s >= pos.
    s_idx_f32 = nl.ndarray((S_MAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.iota(s_idx_f32, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    s_plus_1 = nl.ndarray((S_MAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(s_plus_1, data=s_idx_f32, op0=nl.add, operand0=1.0)
    neg_pos = nl.ndarray((1, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(neg_pos, data=pos_f32, op0=nl.multiply, operand0=-1.0)
    neg_pos_psum = nl.ndarray((S_MAX, 1), dtype=f32, buffer=nl.psum)
    nisa.nc_transpose(neg_pos_psum, neg_pos.ap([[1, 1], [0, S_MAX]], offset=0))
    neg_pos_bcast = nl.ndarray((S_MAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(neg_pos_bcast, neg_pos_psum)
    delta = nl.ndarray((S_MAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(delta, s_plus_1, neg_pos_bcast, op=nl.add)
    relu_delta = nl.ndarray((S_MAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.activation(relu_delta, op=nl.relu, data=delta)
    clamped = nl.ndarray((S_MAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(clamped, data=relu_delta, op0=nl.minimum, operand0=1.0)
    cache_mask_64 = nl.ndarray((S_MAX, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(cache_mask_64, data=clamped, op0=nl.multiply, operand0=MASK_NEG_INF)
    nisa.tensor_copy(causal_mask[0:S_MAX, 0:1], cache_mask_64)

    # 9. Per-kv-head attention with virtual-append layout (S_ATT = S_MAX + 1):
    #   slots 0..S_MAX-1: cache rows from HBM
    #   slot S_MAX:       active K/V from SBUF
    # Score = k_aug.T @ Q, softmax over S_ATT, attn = v_aug.T @ softmax.
    attn_out_per_kv = [None] * NUM_KV_HEADS

    for kv in range(NUM_KV_HEADS):
        # 9a. K_cache[kv] -> SBUF [D, S_MAX] via dma_transpose (D = contraction).
        k_cache_sb = nl.ndarray((D, S_MAX), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_transpose(
            dst=k_cache_sb,
            src=K_cache_flat[kv * S_MAX:(kv + 1) * S_MAX, :],
        )
        # 9b. k_aug = [k_cache | k_active] along the free dim.
        k_aug = nl.ndarray((D, S_ATT), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(k_aug[0:D, 0:S_MAX], k_cache_sb)
        nisa.tensor_copy(k_aug[0:D, S_MAX:S_ATT], k_active_per_kv[kv])

        # 9c. scores = k_aug.T @ Q  → PSUM [S_ATT, GQA].
        score_psum = nl.ndarray((S_ATT, GQA), dtype=f32, buffer=nl.psum)
        nisa.memset(score_psum, value=0.0)
        nisa.nc_matmul(score_psum, stationary=k_aug, moving=q_rope_per_kv[kv])
        score_scaled = nl.ndarray((S_ATT, GQA), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(score_scaled, data=score_psum, op0=nl.multiply, operand0=INV_SQRT_D)
        score_masked = nl.ndarray((S_ATT, GQA), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(score_masked, data=score_scaled, op0=nl.add, operand0=causal_mask)

        # 9d. Softmax across S_ATT (partition) dim.
        score_T_psum = nl.ndarray((GQA, S_ATT), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(score_T_psum, score_masked)
        score_T_sb = nl.ndarray((GQA, S_ATT), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(score_T_sb, score_T_psum)
        smax_max = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(smax_max, op=nl.maximum, data=score_T_sb, axis=(1,))
        neg_max = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(neg_max, data=smax_max, op0=nl.multiply, operand0=-1.0)
        neg_max_psum = nl.ndarray((S_ATT, GQA), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(neg_max_psum, neg_max.ap([[1, GQA], [0, S_ATT]], offset=0))
        neg_max_bcast = nl.ndarray((S_ATT, GQA), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(neg_max_bcast, neg_max_psum)
        score_shifted = nl.ndarray((S_ATT, GQA), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(score_shifted, score_masked, neg_max_bcast, op=nl.add)
        score_exp = nl.ndarray((S_ATT, GQA), dtype=f32, buffer=nl.sbuf)
        nisa.activation(score_exp, op=nl.exp, data=score_shifted)
        score_exp_T_psum = nl.ndarray((GQA, S_ATT), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(score_exp_T_psum, score_exp)
        score_exp_T_sb = nl.ndarray((GQA, S_ATT), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(score_exp_T_sb, score_exp_T_psum)
        exp_sum = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(exp_sum, op=nl.add, data=score_exp_T_sb, axis=(1,))
        inv_sum = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(inv_sum, exp_sum)
        inv_sum_psum = nl.ndarray((S_ATT, GQA), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(inv_sum_psum, inv_sum.ap([[1, GQA], [0, S_ATT]], offset=0))
        inv_sum_bcast = nl.ndarray((S_ATT, GQA), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(inv_sum_bcast, inv_sum_psum)
        softmax_fp32 = nl.ndarray((S_ATT, GQA), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(softmax_fp32, score_exp, inv_sum_bcast, op=nl.multiply)
        softmax_bf16 = nl.ndarray((S_ATT, GQA), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(softmax_bf16, softmax_fp32)

        # 9e. v_aug = [v_cache; v_active_T] along the partition dim.
        # The active row is a single-partition write at base S_MAX = 64.
        v_aug = nl.ndarray((S_ATT, D), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=v_aug[0:S_MAX, 0:D],
            src=V_cache_flat[kv * S_MAX:(kv + 1) * S_MAX, :],
        )
        nisa.tensor_copy(v_aug[S_MAX:S_ATT, 0:D], v_active_T_per_kv[kv])

        # 9f. attn = v_aug.T @ softmax (K-axis = S_ATT in partition) → PSUM [D, GQA].
        attn_psum = nl.ndarray((D, GQA), dtype=f32, buffer=nl.psum)
        nisa.memset(attn_psum, value=0.0)
        nisa.nc_matmul(attn_psum, stationary=v_aug, moving=softmax_bf16)
        attn_kv = nl.ndarray((D, GQA), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(attn_kv, attn_psum)
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

    # 12. Build K_next / V_next: full copy of the input cache, then scatter the
    # freshly projected K / V at row (kv * S_MAX + position_ids). The caller
    # aliases K_next / V_next back to past_key_values for the next step.
    K_next_flat = K_next.reshape((NUM_KV_HEADS * S_MAX, D))
    V_next_flat = V_next.reshape((NUM_KV_HEADS * S_MAX, D))
    nisa.dma_copy(dst=K_next_flat, src=K_cache_flat)
    nisa.dma_copy(dst=V_next_flat, src=V_cache_flat)

    for kv in range(NUM_KV_HEADS):
        nisa.dma_copy(
            dst=K_next_flat.ap(
                pattern=[[D, 1], [1, D]],
                offset=kv * S_MAX * D,
                scalar_offset=pos_u32,
                indirect_dim=0,
            ),
            src=k_active_T_per_kv[kv],
        )
        nisa.dma_copy(
            dst=V_next_flat.ap(
                pattern=[[D, 1], [1, D]],
                offset=kv * S_MAX * D,
                scalar_offset=pos_u32,
                indirect_dim=0,
            ),
            src=v_active_T_per_kv[kv],
        )


@nki.jit
def attn_kernel(
    hidden_hbm,           # [1, 1, H=2048] bf16 HBM
    Wq_pt,                # [n_q*PMAX=4096, n_h_tiles*D=1024] bf16 HBM
    Wk_pt,                # [n_kv*PMAX=1024, n_h_tiles*D=1024] bf16 HBM
    Wv_pt,                # [n_kv*PMAX=1024, n_h_tiles*D=1024] bf16 HBM
    Wo,                   # [n_q*D=2048, H=2048] bf16 HBM
    gamma_pre_attn,       # [H=2048] bf16 HBM
    K_cache,              # [1, n_kv=8, S_MAX=64, D=64] bf16 HBM, read-only input
    V_cache,              # [1, n_kv=8, S_MAX=64, D=64] bf16 HBM, read-only input
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

    # Load hidden HBM [1, 1, H] → SBUF [PMAX, NUM_H_TILES].
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
