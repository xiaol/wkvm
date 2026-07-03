"""RWKV-7 model integration (fla-format checkpoints).

M1 uses the reference ``fla`` module (`RWKV7ForCausalLM`) as the compute
graph and takes ownership of the *state* only: every forward is driven with
an explicitly seeded `fla` Cache whose per-layer entries are gathered from
the engine's arena tensors and scattered back afterwards. This gives exact
kernel parity for free (chunked scan `chunk_rwkv7` for prefill, fused
recurrent for decode — the same dispatch the reference uses) while keeping
request state where the roadmap requires it: in dense per-family GPU
tensors indexed by arena slot ids, never in per-request python objects.

Per-layer recurrent state carried by RWKV-7 (verified against
``fla/layers/rwkv7.py`` and ``fla/models/rwkv7/modeling_rwkv7.py``):

- ``recurrent_state``: the wkv matrix state, ``[H, head_dim, head_v_dim]``,
  float32 (the fla kernels always emit/consume fp32 final states).
- ``conv_state``: token-shift cache of the attention mixer, ``[hidden]``,
  model dtype.
- ``ffn_state``: token-shift cache of the channel mixer, ``[hidden]``,
  model dtype.

``v_first`` is recomputed from layer 0 within each forward and never
crosses a step boundary, so it is not state.

Zero state is exactly "fresh sequence" for all three components: a zero wkv
matrix is the empty prefix sum, and a zero token-shift cache reproduces the
ZeroPad2d shift-in at position 0. Slot admission therefore just zeroes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wkvm.core.config import ModelStateSpec, StateFamilySpec

WKV_DTYPE = torch.float32


@dataclass(frozen=True)
class RWKV7StateLayout:
    """Shapes/dtypes of the per-slot state, derived from a loaded config.

    Family names match ``wkvm.core.config.rwkv7_state_spec`` ("wkv",
    "shift"); byte counts here are exact (wkv is fp32, not model dtype).
    """

    n_layer: int
    hidden_size: int
    num_heads: int
    head_dim: int
    head_v_dim: int
    dtype: torch.dtype  # model dtype; used for the shift family

    @classmethod
    def from_config(cls, config, dtype: torch.dtype) -> RWKV7StateLayout:
        value_dims = set(config.value_dim)
        if len(value_dims) != 1:
            raise NotImplementedError(f"heterogeneous value_dim: {config.value_dim}")
        if config.attn is not None:
            raise NotImplementedError("hybrid attn layers are M4, not M1")
        return cls(
            n_layer=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            head_v_dim=value_dims.pop() // config.num_heads,
            dtype=dtype,
        )

    # Per-slot shapes, layer-major: the bank stores [n_layer, slots+1, ...]
    # so a per-layer batch gather is one contiguous index_select.
    @property
    def wkv_shape(self) -> tuple[int, ...]:
        return (self.num_heads, self.head_dim, self.head_v_dim)

    @property
    def shift_shape(self) -> tuple[int, ...]:
        # 2 token-shift caches per layer: [0] attn conv_state, [1] ffn_state.
        return (self.hidden_size,)

    def state_spec(self) -> ModelStateSpec:
        wkv_elems = self.n_layer * self.num_heads * self.head_dim * self.head_v_dim
        shift_elems = self.n_layer * 2 * self.hidden_size
        return ModelStateSpec(
            families=(
                StateFamilySpec(
                    name="wkv",
                    bytes_per_slot=wkv_elems * WKV_DTYPE.itemsize,
                    layer_ids=tuple(range(self.n_layer)),
                ),
                StateFamilySpec(
                    name="shift",
                    bytes_per_slot=shift_elems * self.dtype.itemsize,
                    layer_ids=tuple(range(self.n_layer)),
                ),
            )
        )


def load_rwkv7(
    model_path: str,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load an fla-format RWKV-7 checkpoint for inference.

    Returns ``(model, layout)``. The model is frozen and in eval mode —
    eval matters beyond dropout: the fla layer dispatches chunk vs fused
    recurrent kernels on ``self.training or seq_len >= 64``.
    """
    from fla.models.rwkv7 import RWKV7ForCausalLM

    model = RWKV7ForCausalLM.from_pretrained(model_path, dtype=dtype)
    model = model.to(device).eval().requires_grad_(False)
    layout = RWKV7StateLayout.from_config(model.config, dtype=dtype)
    return model, layout
