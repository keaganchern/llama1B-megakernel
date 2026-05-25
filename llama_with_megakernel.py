"""NKI-enabled Llama-3.2-1B NxDI subclass.

This is the model class that gets loaded when --enable-nki is passed. It mirrors
the structure of `llama.py` (the XLA baseline) but overrides the decoder layer
so that layer 0, on token-generation, dispatches the multi-layer megakernel
defined in transformer_llama.py. Layers 1..L-1 become no-op pass-throughs.

The integration pattern is taken from
  /home/ubuntu/kchern/nki-moe-megakernel/megakernels/qwen3_moe/qwen_with_megakernel.py
Key bits to mirror here:
  - Subclass NeuronLlamaDecoderLayer; override forward()
  - In __init__, install weakref.ref(self._parent_model) on each layer
  - Layer 0: gather W_qkv/W_out/W_gate/W_up/W_down + KV caches via the parent
    ref, call transformer_llama_megakernel(...), store output, return
  - Layers 1..L-1: return the hidden state layer 0 already computed
  - Add a NeuronConfig flag (e.g. attn_block_tkg_nki_kernel_cache_update) so
    NxDI skips its own KV-cache scatter and trusts the megakernel's in-place
    writes
  - Override LlamaInferenceConfig.get_neuron_config_cls() to return the
    NKI-aware config

TODO: implement once the baseline llama.py + main.py wiring is solid.
"""

# Importing from the sibling baseline file lets us reuse all the SDK-aligned
# pieces (LlamaInferenceConfig, NeuronLlamaForCausalLM, NeuronLlamaModel,
# NeuronLlamaDecoderLayer, NeuronLlamaAttention, NeuronLlamaMLP). The
# subclasses below override just what's needed for the megakernel path.
from llama import *  # noqa: F401,F403

# When this module is fully implemented, main.py imports
# NeuronLlamaForCausalLM from here; right now we just re-export the baseline so
# --enable-nki at least doesn't crash on import.
