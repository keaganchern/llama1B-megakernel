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
    K_cache,              # [1, n_kv, S_MAX, D] bf16 HBM — read, then in-place K scatter
    V_cache,              # [1, n_kv, S_MAX, D] bf16 HBM — read, then in-place V scatter
    cos,                  # [1, 1, D] bf16 HBM, indexed at position_ids
    sin,                  # [1, 1, D] bf16 HBM
    position_ids,         # [1, 1] int32 HBM
    out_sb,               # [PMAX, NUM_H_TILES] bf16 SBUF — destination
    layer_idx=0,          # Python int, selects per-layer slice of stacked weights
):
    """SBUF-in / SBUF-out fused attention block.

    Step 9 runs single-pass softmax attention over NUM_S_TILES cache tiles of
    S_TILE=PMAX=128 positions each, plus the freshly projected K/V active slot.
    The cache mask is built per-tile from (position_ids, t*S_TILE); the active
    slot is always unmasked.

    The freshly projected K / V are scattered IN PLACE into K_cache / V_cache at
    row position_ids (step 12). The attention reads (positions < pos) all
    precede that write (row pos), so the in-place update is safe; the caller
    returns K_cache / V_cache as pass-through outputs so the scatter DMAs aren't
    DCE'd and NxDI aliases them back to past_key_values. (The old fresh-output
    K_next / V_next path wholesale-copied the input cache first — ~64 MB/token
    of avoidable HBM traffic; in-place was a ruled-out divergence suspect, the
    real bug was compound HBM slicing, now fixed via absolute-offset .ap().)
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

    # 2. Pre-attention RMSNorm. input_layernorm gamma is FOLDED into Wq/Wk/Wv
    # at load time (fused RMSNorm — see
    # NeuronLlamaForCausalLMMK.convert_hf_to_neuron_state_dict), so the kernel
    # applies only the 1/rms scale — no per-layer gamma load or gamma multiply:
    #   h_all = hidden / rms(hidden), cast to bf16.
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
    h_all = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(h_all, data=h_f32, op0=nl.multiply, operand0=rms_inv)

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
            dge_mode=nisa.dge_mode.none,   # static DMA — coalesce weight read (see mlp.py note)
        )
        wk_sb_per_kv[kv] = wk_t
        wv_t = nl.ndarray((PMAX, NUM_H_TILES * D), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=wv_t,
            src=Wv_pt.ap(
                pattern=[[_wk_row_stride, PMAX], [1, _wk_row_stride]],
                offset=_row0 * _wk_row_stride,
            ),
            dge_mode=nisa.dge_mode.none,   # static DMA — coalesce weight read (see mlp.py note)
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
            dge_mode=nisa.dge_mode.none,   # static DMA — coalesce weight read (see mlp.py note)
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

    # 8b. Precompute the per-tile causal masks ONCE. They depend only on the
    # tile index and position_ids, NOT on the kv-head, so building them inside
    # each kv-head's score loop rebuilt them 8x (NUM_KV_HEADS) redundantly.
    mask_tiles = [None] * NUM_S_TILES
    for t in range(NUM_S_TILES):
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
        mask_t = nl.ndarray((S_TILE, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(mask_t, data=clamped, op0=nl.multiply, operand0=MASK_NEG_INF)
        mask_tiles[t] = mask_t

    # 9. Per-kv-head single-pass softmax attention (decode: one query token).
    # For one query token the whole score row is tiny, so the flash-attention
    # online/streaming softmax (per-tile running max/denom/out + rescale) is
    # pure overhead — it dominated Vector-engine time. Instead: assemble the
    # full masked score row [GQA, S_MAX] over the cache tiles, add the
    # always-unmasked active slot, take ONE softmax, then accumulate exp @ V
    # over the V tiles + active slot (no rescaling) and normalize at the end.
    attn_out_per_kv = [None] * NUM_KV_HEADS

    for kv in range(NUM_KV_HEADS):
        # ----- Score phase: score_all[GQA, S_MAX], scaled + causally masked -----
        score_all = nl.ndarray((GQA, S_MAX), dtype=f32, buffer=nl.sbuf)
        for t in range(NUM_S_TILES):
            k_tile = nl.ndarray((D, S_TILE), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_transpose(
                dst=k_tile,
                src=K_cache_flat[kv * S_MAX + t * S_TILE:kv * S_MAX + (t + 1) * S_TILE, :],
            )
            score_psum = nl.ndarray((S_TILE, GQA), dtype=f32, buffer=nl.psum)
            nisa.memset(score_psum, value=0.0)
            nisa.nc_matmul(score_psum, stationary=k_tile, moving=q_rope_per_kv[kv])

            score_scaled = nl.ndarray((S_TILE, GQA), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(score_scaled, data=score_psum, op0=nl.multiply, operand0=INV_SQRT_D)
            score_masked = nl.ndarray((S_TILE, GQA), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(score_masked, data=score_scaled, op0=nl.add, operand0=mask_tiles[t])

            # Transpose [S_TILE, GQA] → [GQA, S_TILE] into the score_all columns.
            score_T_psum = nl.ndarray((GQA, S_TILE), dtype=f32, buffer=nl.psum)
            nisa.nc_transpose(score_T_psum, score_masked)
            nisa.tensor_copy(score_all[0:GQA, t * S_TILE:(t + 1) * S_TILE], score_T_psum)

        # ----- Active slot score [GQA, 1] (the fresh K/V, always unmasked) -----
        score_active_psum = nl.ndarray((1, GQA), dtype=f32, buffer=nl.psum)
        nisa.memset(score_active_psum, value=0.0)
        nisa.nc_matmul(score_active_psum, stationary=k_active_per_kv[kv], moving=q_rope_per_kv[kv])
        score_active_scaled = nl.ndarray((1, GQA), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(score_active_scaled, data=score_active_psum, op0=nl.multiply, operand0=INV_SQRT_D)
        sa_T_psum = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(sa_T_psum, score_active_scaled)
        score_active = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(score_active, sa_T_psum)

        # ----- One softmax over [score_all | score_active] -----
        row_max = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(row_max, op=nl.maximum, data=score_all, axis=(1,))
        max_all = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(max_all, row_max, score_active, op=nl.maximum)
        neg_max = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(neg_max, data=max_all, op0=nl.multiply, operand0=-1.0)

        # exp(score - max). Masked cache positions are ~-inf → exp = 0.
        exp_all = nl.ndarray((GQA, S_MAX), dtype=f32, buffer=nl.sbuf)
        nisa.activation(exp_all, op=nl.exp, data=score_all, bias=neg_max)
        exp_active = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.activation(exp_active, op=nl.exp, data=score_active, bias=neg_max)

        denom = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(denom, op=nl.add, data=exp_all, axis=(1,))
        nisa.tensor_tensor(denom, denom, exp_active, op=nl.add)
        inv_denom = nl.ndarray((GQA, 1), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(inv_denom, denom)

        # ----- Output phase: out[GQA, D] = exp_all @ V_all + exp_active * v_active -----
        # All tile matmuls + the active slot accumulate into one PSUM (no rescale).
        out_psum = nl.ndarray((GQA, D), dtype=f32, buffer=nl.psum)
        nisa.memset(out_psum, value=0.0)
        for t in range(NUM_S_TILES):
            v_tile = nl.ndarray((S_TILE, D), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=v_tile,
                src=V_cache_flat[kv * S_MAX + t * S_TILE:kv * S_MAX + (t + 1) * S_TILE, :],
            )
            # Transpose exp_all[:, tile] [GQA, S_TILE] → bf16 [S_TILE, GQA] for matmul.
            exp_pt_psum = nl.ndarray((S_TILE, GQA), dtype=f32, buffer=nl.psum)
            nisa.nc_transpose(exp_pt_psum, exp_all[0:GQA, t * S_TILE:(t + 1) * S_TILE])
            exp_bf16 = nl.ndarray((S_TILE, GQA), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(exp_bf16, exp_pt_psum)
            nisa.nc_matmul(out_psum, stationary=exp_bf16, moving=v_tile)

        # Active slot: exp_active[GQA,1] ⊗ v_active[1,D].
        exp_active_T_psum = nl.ndarray((1, GQA), dtype=f32, buffer=nl.psum)
        nisa.nc_transpose(exp_active_T_psum, exp_active)
        exp_active_T = nl.ndarray((1, GQA), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(exp_active_T, exp_active_T_psum)
        nisa.nc_matmul(out_psum, stationary=exp_active_T, moving=v_active_T_per_kv[kv])

        # Normalize: attn_out_qd[GQA, D] = out_psum * (1 / denom).
        attn_out_qd = nl.ndarray((GQA, D), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(attn_out_qd, data=out_psum, op0=nl.multiply, operand0=inv_denom)

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
    # (D, H) block. We make the 128-wide Wo tile the STATIONARY and the 1-wide
    # attn column the MOVING (the reverse of the obvious mapping). On trn1 the
    # matmul cost is max(N, 64) cycles where N = moving free axis: with N=1 each
    # matmul runs in ~64 cycles and emits a [PMAX, 1] output column directly,
    # vs ~128 cycles (N=128) plus a final [1,PMAX]->[PMAX,1] transpose the other
    # way. Halves O-proj matmul time and removes NUM_H_TILES transposes.
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
            dge_mode=nisa.dge_mode.none,   # static DMA — coalesce weight read (see mlp.py note)
        )
        wo_sb_per_q[h_q] = wo_t

    res_psums = []
    for ht in range(NUM_H_TILES):
        rp = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.psum)
        nisa.memset(rp, value=0.0)
        res_psums.append(rp)

    for h_q in range(NUM_Q_HEADS):
        for ht in range(NUM_H_TILES):
            nisa.nc_matmul(
                res_psums[ht],
                stationary=wo_sb_per_q[h_q][0:D, ht * PMAX:(ht + 1) * PMAX],
                moving=attn_out_all[0:D, h_q:h_q + 1],
            )

    # res_psums[ht] is already the [PMAX, 1] output column for H-tile ht.
    for ht in range(NUM_H_TILES):
        nisa.tensor_copy(out_sb[0:PMAX, ht:ht + 1], res_psums[ht])

    # 12. Scatter the freshly projected K / V per kv-head IN PLACE into
    # K_cache / V_cache at row (kv*S_MAX + position_ids). No wholesale copy —
    # the attention reads (step 9, positions < pos) precede this write (row pos)
    # in program order, so the update is safe; the caller returns K_cache /
    # V_cache so these scatters aren't DCE'd and NxDI aliases them back.
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
    # (indirect_dim=0 → indirect_stride = D for K_cache_flat = [n_kv*S_MAX, D]).
    nisa.dma_copy(
        dst=K_cache_flat.ap(
            pattern=[[S_MAX * D, NUM_KV_HEADS], [1, D]],
            offset=0,
            scalar_offset=pos_u32,
            indirect_dim=0,
        ),
        src=k_active_pack_T,
    )
    nisa.dma_copy(
        dst=V_cache_flat.ap(
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

    Returns (out_hbm, K_cache, V_cache); the active K / V are scattered in place
    into K_cache / V_cache at row position_ids, and the caches are returned as
    pass-through outputs.
    """
    bf16 = nl.bfloat16

    hidden_sb = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden_hbm.reshape((H,)).ap(
            pattern=[[1, PMAX], [PMAX, NUM_H_TILES]], offset=0,
        ),
    )

    out_sb = nl.ndarray((PMAX, NUM_H_TILES), dtype=bf16, buffer=nl.sbuf)
    attn_block_sbuf(
        hidden_sb=hidden_sb,
        Wq_pt=Wq_pt, Wk_pt=Wk_pt, Wv_pt=Wv_pt, Wo=Wo,
        gamma_pre_attn=gamma_pre_attn,
        K_cache=K_cache, V_cache=V_cache,
        cos=cos, sin=sin,
        position_ids=position_ids,
        out_sb=out_sb,
    )

    out_hbm = nl.ndarray((1, 1, H), dtype=bf16, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=out_hbm.reshape((H,)).ap(
            pattern=[[1, PMAX], [PMAX, NUM_H_TILES]], offset=0,
        ),
        src=out_sb,
    )
    return out_hbm, K_cache, V_cache
