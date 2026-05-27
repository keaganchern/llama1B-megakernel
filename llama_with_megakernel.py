"""NKI-megakernel-enabled Llama-3.2-1B subclass.

When --enable-nki is passed on the CLI, main.py imports NeuronLlamaForCausalLM
from this module. Decoder-layer-0's forward dispatches the multi-layer
megakernel for all 16 layers in one shot during token generation; layers
1..15 become no-op passthroughs returning the stashed result.
"""

import weakref

import torch
import torch.nn as nn

from llama import *  # noqa: F401, F403
from llama import (
    NeuronConfigNKI,
    NeuronLlamaDecoderLayer,
    NeuronLlamaForCausalLM,
    NeuronLlamaModel,
    get_updated_configs,
)

from transformer_llama import (
    transformer_llama_megakernel_16layers,
    LLAMA_1B_NUM_LAYERS,
)
from nki_kernels.attention import (
    PMAX, D, NUM_Q_HEADS, NUM_KV_HEADS, NUM_H_TILES,
)
from nki_kernels.mlp import NUM_I_TILES

# Layer-boundary markers used by NxDI to identify the megakernel as a single
# "layer block" for alias handling through the custom-call.
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)


def _tile_transpose(W, n_out_tiles, head_dim):
    """Pre-shuffle [n_out_tiles*head_dim, NUM_H_TILES*PMAX] weight into the
    tile-transposed layout expected by attention.py:
      result[h*PMAX + p, h_t*head_dim + f] = W[h*head_dim + f, h_t*PMAX + p]
    """
    return (W.reshape(n_out_tiles, head_dim, NUM_H_TILES, PMAX)
              .permute(0, 3, 2, 1)
              .reshape(n_out_tiles * PMAX, NUM_H_TILES * head_dim)
              .contiguous())


def _tile_transpose_mlp(W, n_out_tiles, n_in_tiles):
    """Same shuffle as `_tile_transpose` but with PMAX as the fan-out tile size
    (MLP fans out PMAX-wide output tiles, not D-wide head dims).
    """
    return (W.reshape(n_out_tiles, PMAX, n_in_tiles, PMAX)
              .permute(0, 3, 2, 1)
              .reshape(n_out_tiles * PMAX, n_in_tiles * PMAX)
              .contiguous())


class NeuronLlamaDecoderLayerMK(NeuronLlamaDecoderLayer):
    """Decoder layer that hijacks layer-0's forward to dispatch the megakernel.

    On token generation, layer 0 fires the multi-layer megakernel for all 16
    layers at once; layers 1..15 return the stashed result. On prefill /
    context-encoding, behavior is unchanged from baseline.
    """

    def __init__(self, config, layer_idx: int):
        # NeuronConfigLlamaMK sets attn_block_tkg_nki_kernel_enabled=True so the
        # model-level update_kv_per_layer gate fires, but with _enabled=True
        # NeuronAttentionBase.__init__ would also build an NKI o_proj that has
        # an n_q ≤ 17 constraint (Llama-1B has 32 Q heads). Flip the flag to
        # False on the instance during super().__init__(), then restore.
        _saved_block_flag = config.neuron_config.attn_block_tkg_nki_kernel_enabled
        config.neuron_config.attn_block_tkg_nki_kernel_enabled = False
        try:
            super().__init__(config)
        finally:
            config.neuron_config.attn_block_tkg_nki_kernel_enabled = _saved_block_flag

        self.layer_idx = layer_idx
        # Set by NeuronLlamaModelMK.init_model() after self.layers exists.
        # Weakref so the parent reference isn't an nn.Module child.
        self._parent_model_ref = None

    @property
    def _parent_model(self):
        return self._parent_model_ref() if self._parent_model_ref is not None else None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        adapter_ids=None,
        rotary_position_ids=None,
        residual=None,
        **kwargs,
    ):
        is_token_gen = past_key_value is not None

        if is_token_gen and self.layer_idx == 0:
            hidden_states = ModuleMarkerStartWrapper()(hidden_states)
            assert self._parent_model is not None, (
                "Layer 0 needs _parent_model_ref set by NeuronLlamaModelMK.init_model()"
            )

            parent = self._parent_model
            L = LLAMA_1B_NUM_LAYERS
            assert len(parent.layers) == L

            # Cat-stack per-layer weights along the row dim. Each NKI kernel
            # selects its layer's slice via a single absolute .ap() offset.
            Wq_stack = torch.cat([
                _tile_transpose(parent.layers[i].self_attn.qkv_proj.q_proj.weight,
                                NUM_Q_HEADS, D)
                for i in range(L)
            ], dim=0)
            Wk_stack = torch.cat([
                _tile_transpose(parent.layers[i].self_attn.qkv_proj.k_proj.weight,
                                NUM_KV_HEADS, D)
                for i in range(L)
            ], dim=0)
            Wv_stack = torch.cat([
                _tile_transpose(parent.layers[i].self_attn.qkv_proj.v_proj.weight,
                                NUM_KV_HEADS, D)
                for i in range(L)
            ], dim=0)
            Wo_stack = torch.cat([
                parent.layers[i].self_attn.o_proj.o_proj.weight.t().contiguous()
                for i in range(L)
            ], dim=0)
            gpre_stack = torch.cat([
                parent.layers[i].input_layernorm.weight for i in range(L)
            ], dim=0)
            gpost_stack = torch.cat([
                parent.layers[i].post_attention_layernorm.weight for i in range(L)
            ], dim=0)

            # Fuse W_gate + W_up along the free dim (cols) so one DMA per
            # m-tile loads both stationaries contiguously. Per-layer shape:
            # [NUM_I_TILES*PMAX, 2*NUM_H_TILES*PMAX] = [8192, 4096].
            Wgu_stack = torch.cat([
                torch.cat([
                    _tile_transpose_mlp(parent.layers[i].mlp.gate_proj.weight,
                                        NUM_I_TILES, NUM_H_TILES),
                    _tile_transpose_mlp(parent.layers[i].mlp.up_proj.weight,
                                        NUM_I_TILES, NUM_H_TILES),
                ], dim=1)
                for i in range(L)
            ], dim=0)
            Wd_stack = torch.cat([
                _tile_transpose_mlp(parent.layers[i].mlp.down_proj.weight,
                                    NUM_H_TILES, NUM_I_TILES)
                for i in range(L)
            ], dim=0)

            # KV caches stay as per-layer Parameters (NxDI mutates them across
            # decode steps). The cache manager keeps them in a flat
            # ParameterList: [K_0, V_0, K_1, V_1, ...].
            kv_mgr = parent.kv_mgr
            K_caches = [kv_mgr.past_key_values[2 * i]     for i in range(L)]
            V_caches = [kv_mgr.past_key_values[2 * i + 1] for i in range(L)]

            sa0 = parent.layers[0].self_attn
            cos_cache, sin_cache = sa0.rotary_emb(hidden_states, position_ids)
            pos_i32 = position_ids.to(torch.int32)

            # The megakernel is hardcoded for TP=1, 32 Q heads, 8 KV heads,
            # H=2048, S_MAX=1024. Assert config agreement.
            nc = self._parent_model.config.neuron_config
            assert nc.tp_degree == 1, (
                f"megakernel is TP=1 only, got tp_degree={nc.tp_degree}"
            )
            assert tuple(Wq_stack.shape) == (L * 32 * 128, 16 * 64), tuple(Wq_stack.shape)
            assert tuple(Wk_stack.shape) == (L *  8 * 128, 16 * 64), tuple(Wk_stack.shape)
            assert tuple(Wv_stack.shape) == (L *  8 * 128, 16 * 64), tuple(Wv_stack.shape)
            assert tuple(Wo_stack.shape) == (L * 32 *  64, 2048),    tuple(Wo_stack.shape)
            assert tuple(gpre_stack.shape) == (L * 2048,),           tuple(gpre_stack.shape)
            assert tuple(gpost_stack.shape) == (L * 2048,),          tuple(gpost_stack.shape)
            assert tuple(Wgu_stack.shape) == (L * 8192, 2 * 2048),   tuple(Wgu_stack.shape)
            assert tuple(Wd_stack.shape) == (L * 2048, 8192),        tuple(Wd_stack.shape)
            for i, k in enumerate(K_caches):
                assert tuple(k.shape) == (1, 8, 1024, 64), f"K_caches[{i}].shape={tuple(k.shape)}"
            for i, v in enumerate(V_caches):
                assert tuple(v.shape) == (1, 8, 1024, 64), f"V_caches[{i}].shape={tuple(v.shape)}"

            # Returns (Y, K_0, ..., K_{L-1}, V_0, ..., V_{L-1}).
            mk_results = transformer_llama_megakernel_16layers(
                hidden_states,
                Wq_stack, Wk_stack, Wv_stack, Wo_stack,
                gpre_stack,
                Wgu_stack, Wd_stack,
                gpost_stack,
                *K_caches, *V_caches,
                cos_cache, sin_cache, pos_i32,
            )
            Y = mk_results[0]
            K_outs = list(mk_results[1     : 1 + L])
            V_outs = list(mk_results[1 + L : 1 + 2 * L])
            present_kv = [(K_outs[i], V_outs[i]) for i in range(L)]

            Y = ModuleMarkerEndWrapper()(Y)

            self._parent_model._mk_outputs = {
                "hidden_states": Y,
                "present_kv": present_kv,
            }
            return (Y, present_kv[0], cos_cache, sin_cache, None)

        if is_token_gen and self.layer_idx > 0:
            # Each layer returns its OWN present_kv so NxDI aliases it back to
            # the correct kv_mgr.past_key_values slot.
            stashed = self._parent_model._mk_outputs
            return (stashed["hidden_states"], stashed["present_kv"][self.layer_idx],
                    None, None, None)

        # Prefill / context-encoding: baseline path.
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rotary_position_ids=rotary_position_ids,
            residual=residual,
            **kwargs,
        )


class NeuronLlamaModelMK(NeuronLlamaModel):
    """Llama model that builds MK decoder layers and installs parent weakrefs."""

    def init_model(self, config):
        super().init_model(config)

        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList([
            NeuronLlamaDecoderLayerMK(conf, layer_idx=i)
            for i, conf in enumerate(updated_configs)
        ])

        for layer in self.layers:
            layer._parent_model_ref = weakref.ref(self)


class NeuronConfigLlamaMK(NeuronConfigNKI):
    """NeuronConfig variant that tells NxDI the megakernel updates KV in-place.

    Setting attn_block_tkg_nki_kernel_cache_update=True makes the model-level
    update_kv_per_layer=True, which SKIPS kv_mgr.update_cache (the Python-side
    torch.scatter that would otherwise double-write on top of the megakernel's
    own scatter). NxDI's assertion gate requires attn_block_tkg_nki_kernel_enabled
    alongside it; we set that via the deprecated alias
    `attn_tkg_nki_kernel_enabled` which is auto-promoted AFTER NxDI's
    qkv_kernel_enabled assertion (Llama-1B's 32 Q heads violate that kernel's
    n_q ≤ 17 constraint, so we can't set the modern flag directly).
    """

    def __init__(self, **kwargs):
        kwargs["attn_tkg_nki_kernel_enabled"] = True
        kwargs["attn_block_tkg_nki_kernel_cache_update"] = True
        super().__init__(**kwargs)


class NeuronLlamaForCausalLMMK(NeuronLlamaForCausalLM):
    """NKI-megakernel-enabled CausalLM. Selected by main.py when --enable-nki."""

    _model_cls = NeuronLlamaModelMK

    @classmethod
    def get_neuron_config_cls(cls):
        return NeuronConfigLlamaMK


# Symbol main.py imports when --enable-nki is set.
NeuronLlamaForCausalLM = NeuronLlamaForCausalLMMK
