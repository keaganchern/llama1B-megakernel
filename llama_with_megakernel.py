"""NKI-megakernel-enabled Llama-3.2-1B subclass.

When --enable-nki is passed on the CLI, main.py imports NeuronLlamaForCausalLM
from this module (instead of the baseline llama.py). The classes here override
the baseline so that decoder-layer-0, on token generation, dispatches the
multi-layer megakernel defined in transformer_llama.py for all 16 layers in
one shot. Layers 1..15 become no-op pass-throughs returning the stashed result.

Architecture mirrors qwen_with_megakernel.py (the MoE reference at
/home/ubuntu/kchern/nki-moe-megakernel/megakernels/qwen3_moe/) but stripped
down for dense Llama on trn1.

STATUS: Milestone 1 -- passthrough megakernel wired up end-to-end. Token
output will be garbage (no actual transformer compute happens), but this
validates that:
    1. NeuronConfigNKI is selected when --enable-nki is set
    2. Our DecoderLayer subclass loads in place of the baseline
    3. weakref-to-parent is installed correctly
    4. Layer-0's forward() fires the megakernel
    5. Layers 1..15 are correctly skipped (the megakernel did their work)
    6. NxDI accepts our return tuple shape

Once milestone 1 is green, we evolve the megakernel from passthrough to real
multi-layer fused transformer.
"""

import weakref
from typing import Type

import torch.nn as nn

# Pull in every baseline symbol so `from llama_with_megakernel import X`
# works for any X that lives in llama.py. The MK subclasses below override
# just the pieces that need megakernel-specific behavior.
from llama import *  # noqa: F401, F403

# Explicit re-imports for clarity (these are the classes we override).
from llama import (
    LlamaInferenceConfig,
    NeuronConfig,
    NeuronConfigNKI,
    NeuronLlamaDecoderLayer,
    NeuronLlamaForCausalLM,
    NeuronLlamaModel,
    get_updated_configs,
)

from transformer_llama import transformer_llama_megakernel_passthrough


# === Config ===
# For milestone 1 we just inherit NeuronConfigNKI as-is. The TP=1 LNC=1 decision
# is enforced at config-build time in main.py / on the command line, not here.
# A separate NeuronConfigLlamaMK will get added later if/when we need megakernel-
# specific flags (e.g., a "skip per-layer KV scatter" gate, mirroring qwen3's
# attn_block_tkg_nki_kernel_cache_update).


# === Decoder layer (the hijack point) ===
class NeuronLlamaDecoderLayerMK(NeuronLlamaDecoderLayer):
    """Decoder layer with hijack-layer-0 megakernel dispatch.

    On token generation (TKG), layer 0 calls the multi-layer megakernel
    ONCE for all 16 layers. Layers 1..15 are pass-throughs that return the
    stashed result.

    On prefill / context-encoding (CTE), behavior is unchanged from baseline.
    The megakernel only kicks in for autoregressive decode.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx
        # Set by NeuronLlamaModelMK.init_model() AFTER self.layers exists.
        # We use weakref so that the layer's parent reference doesn't register
        # as an nn.Module child (which would create a reference cycle and
        # break .train() / .eval() recursion).
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

        # ----- TKG + layer 0: fire the megakernel -----
        if is_token_gen and self.layer_idx == 0:
            assert self._parent_model is not None, (
                "Layer 0 needs _parent_model_ref set by "
                "NeuronLlamaModelMK.init_model()"
            )

            # PASSTHROUGH STUB. Once the plumbing is validated, this will be
            # replaced with:
            #   Y, K_outs, V_outs = real_megakernel(
            #       hidden_states,
            #       *self._gather_weights_from_parent(),
            #       *self._gather_kv_caches(past_key_value),
            #       ...
            #   )
            Y = transformer_llama_megakernel_passthrough(hidden_states)

            # Stash the result so layers 1..15 can pull it out. With a real
            # megakernel we'd also stash per-layer K/V outputs here.
            self._parent_model._mk_outputs = {"hidden_states": Y}

            # Return tuple matches baseline NeuronLlamaDecoderLayer.forward():
            #   (hidden_states, present_kv, cos_cache, sin_cache, residual)
            # For passthrough, present_kv = past_key_value (unchanged); cos/sin/
            # residual are None because we don't compute them. NxDI's outer loop
            # tolerates None for the cos/sin/residual slots.
            return (Y, past_key_value, None, None, None)

        # ----- TKG + layer 1..15: pass-through, megakernel already did work -----
        if is_token_gen and self.layer_idx > 0:
            stashed = self._parent_model._mk_outputs
            return (stashed["hidden_states"], past_key_value, None, None, None)

        # ----- CTE / prefill: use the baseline path -----
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


# === Model (the layer factory + weakref installer) ===
class NeuronLlamaModelMK(NeuronLlamaModel):
    """Llama model that builds MK decoder layers and installs parent weakrefs."""

    def init_model(self, config):
        # Run the baseline init to set up embed_tokens, lm_head, norm, etc.
        # We'll discard and replace `self.layers` below; the redundant first
        # build is one-time at model load and not on the per-token critical path.
        super().init_model(config)

        # Replace baseline NeuronLlamaDecoderLayer instances with our MK subclass.
        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList([
            NeuronLlamaDecoderLayerMK(conf, layer_idx=i)
            for i, conf in enumerate(updated_configs)
        ])

        # Install weakref-to-parent on every layer AFTER the ModuleList exists,
        # so that layer 0 can reach into siblings to gather their weights.
        # Done in a second pass because the layer's __init__ can't see the parent
        # (the parent is still being constructed at that point).
        for layer in self.layers:
            layer._parent_model_ref = weakref.ref(self)


# === CausalLM entry point ===
class NeuronLlamaForCausalLMMK(NeuronLlamaForCausalLM):
    """NKI-megakernel-enabled CausalLM. Selected by main.py when --enable-nki."""

    _model_cls = NeuronLlamaModelMK


# === Public symbol that main.py expects ===
# main.py does `from llama_with_megakernel import NeuronLlamaForCausalLM` when
# --enable-nki is set. The `from llama import *` above re-exported the baseline
# class under that name; this rebinding swaps in our MK version.
NeuronLlamaForCausalLM = NeuronLlamaForCausalLMMK
