"""Qwen3.5 hybrid (Gated DeltaNet + full attention) integration — M4.

Same ownership split as ``models/rwkv7.py``: the HF reference modules are the
compute graph (``Qwen3_5DecoderLayer`` and friends, pure-torch kernels), and
wkvm takes ownership of the *state* only. Every forward is driven with a
wkvm-owned staging cache whose per-layer entries are gathered from arena
tensors and scattered back afterwards (``runner/hybrid_state.py``).

Per-request state carried by Qwen3.5 (verified against
``transformers/models/qwen3_5/modeling_qwen3_5.py`` and the
``LinearAttentionLayer`` cache in ``transformers/cache_utils.py``):

- ``linear_attention`` layers (Gated DeltaNet; 24 of 32 in Qwen3.5-9B):
  * ``gdn_state``: recurrent matrix state ``[H_v, K, V]``, float32 — the
    reference kernels emit and consume fp32 states regardless of dtype.
  * ``gdn_conv``: causal-conv1d window ``[conv_dim, kernel]`` in model dtype,
    ``conv_dim = 2 * key_dim + value_dim``.
- ``full_attention`` layers (8 of 32): the **guest** family ``guest_kv`` — a
  *paged* family: K and V pools of ``page_tokens``-token pages
  (``[kv_heads, page_tokens, head_dim]`` per page per layer), keys stored
  post-RoPE. A request reserves the pages for its whole lifetime
  (``prompt + max_new_tokens``) at admission, so admission stays a count
  (free slots and free pages) and nothing is preempted mid-flight. This is
  the deliberately dumb guest allocator of ROADMAP M4 (``docs/HYBRID_ENGINE_PLAN.md``).

Zero state is exactly "fresh sequence" for every family: a zero recurrent
state is what the kernels use for ``initial_state=None``; a zero conv window
reproduces the left zero padding of the cache-less path; an empty guest
history has length 0. Slot admission therefore just zeroes. A *tuned* initial
state (an RNN-StateTuning adapter) is the same slot with a non-zero
``gdn_state`` at length 0 — a state that is not ``f(token-prefix)`` for any
prefix, which is the capability prefix-keyed caches cannot represent
(docs/ANGLE.md §5).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from wkvm.core.config import ModelStateSpec, StateFamilySpec

GDN_STATE_DTYPE = torch.float32

GDN_STATE_FAMILY = "gdn_state"
GDN_CONV_FAMILY = "gdn_conv"
GUEST_KV_FAMILY = "guest_kv"

LINEAR = "linear_attention"
FULL = "full_attention"


@dataclass(frozen=True)
class Qwen35HybridLayout:
    """Shapes/dtypes of the per-slot state, derived from a loaded config.

    Family names: ``gdn_state``, ``gdn_conv`` (linear layers, one slot each)
    and ``guest_kv`` (full-attention layers, paged). Byte counts are exact.
    """

    n_layer: int
    layer_types: tuple[str, ...]
    hidden_size: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    dtype: torch.dtype
    page_tokens: int = 256
    # Guest memory mode for the full-attention layers:
    #   "paged": exact attention over a paged pool (M4; context bounded by the pool)
    #   "ring":  sink + sliding window per slot (M5.1; constant memory, tokens
    #            older than ``ring_tokens`` are evicted — approximate beyond the window)
    guest_mode: str = "paged"
    sink_tokens: int = 16
    ring_tokens: int = 1024
    # routed mode (docs/HYBRID_ENGINE_PLAN.md H9): pending threshold, span slots,
    # representative budget per slot, span cutting, retention, routing.
    routed_pending: int = 512
    routed_slots: int = 64
    routed_reps: int = 48
    routed_max_span: int = 48
    routed_fallback_span: int = 32
    routed_dup_floor: float = 0.10
    routed_new_slot_sim: float = 0.60
    break_token_ids: tuple[int, ...] = ()  # tokens that end a span (sentence punctuation, newline)

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.n_layer:
            raise ValueError("layer_types length must match n_layer")
        bad = set(self.layer_types) - {LINEAR, FULL}
        if bad:
            raise NotImplementedError(f"unsupported layer types: {sorted(bad)}")
        if self.page_tokens < 1:
            raise ValueError("page_tokens must be >= 1")
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError("num_v_heads must be a multiple of num_k_heads")
        if self.guest_mode not in ("paged", "ring", "routed"):
            raise ValueError("guest_mode must be 'paged', 'ring' or 'routed'")
        if self.guest_mode in ("ring", "routed") and (self.sink_tokens < 0 or self.ring_tokens < 1):
            raise ValueError("ring/routed mode needs sink_tokens >= 0 and ring_tokens >= 1")
        if self.guest_mode == "routed" and self.routed_pending > self.ring_tokens:
            raise ValueError("routed_pending must not exceed ring_tokens (prefill sub-chunks are bounded by it)")

    def routed_params(self):
        from wkvm.runner.hybrid_routed import RoutedParams

        return RoutedParams(
            sink=self.sink_tokens, ring=self.ring_tokens, pending=self.routed_pending, slots=self.routed_slots,
            reps=self.routed_reps, max_span=self.routed_max_span, fallback_span=self.routed_fallback_span,
            dup_floor=self.routed_dup_floor, new_slot_sim=self.routed_new_slot_sim,
        )

    @property
    def window_tokens(self) -> int:
        """Columns of the guest window: sink + ring (ring mode) or the whole
        routed column space (routed mode)."""
        if self.guest_mode == "routed":
            return self.routed_params().columns
        return self.sink_tokens + self.ring_tokens

    @classmethod
    def from_config(
        cls, config, dtype: torch.dtype, page_tokens: int = 256, **guest
    ) -> "Qwen35HybridLayout":
        cfg = getattr(config, "text_config", config)
        return cls(
            n_layer=cfg.num_hidden_layers,
            layer_types=tuple(cfg.layer_types),
            hidden_size=cfg.hidden_size,
            num_k_heads=cfg.linear_num_key_heads,
            num_v_heads=cfg.linear_num_value_heads,
            head_k_dim=cfg.linear_key_head_dim,
            head_v_dim=cfg.linear_value_head_dim,
            conv_kernel=cfg.linear_conv_kernel_dim,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=getattr(cfg, "head_dim", None)
            or cfg.hidden_size // cfg.num_attention_heads,
            vocab_size=cfg.vocab_size,
            dtype=dtype,
            page_tokens=page_tokens,
            **guest,
        )

    # -- layer families --------------------------------------------------------

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == LINEAR)

    @property
    def attn_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == FULL)

    @property
    def n_gdn(self) -> int:
        return len(self.gdn_layers)

    @property
    def n_attn(self) -> int:
        return len(self.attn_layers)

    # -- per-slot shapes (layer-major banks: [n_family_layers, slots+1, ...]) ----

    @property
    def conv_dim(self) -> int:
        return 2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim

    @property
    def gdn_state_shape(self) -> tuple[int, ...]:
        return (self.num_v_heads, self.head_k_dim, self.head_v_dim)

    @property
    def gdn_conv_shape(self) -> tuple[int, ...]:
        return (self.conv_dim, self.conv_kernel)

    @property
    def guest_page_shape(self) -> tuple[int, ...]:
        """One page of K (or V) for one layer: ``[kv_heads, page_tokens, head_dim]``."""
        return (self.num_kv_heads, self.page_tokens, self.head_dim)

    @property
    def guest_bytes_per_token(self) -> int:
        """K+V bytes one token costs across all guest layers."""
        return self.n_attn * 2 * self.num_kv_heads * self.head_dim * self.dtype.itemsize

    @property
    def bytes_per_page(self) -> int:
        return self.page_tokens * self.guest_bytes_per_token

    def state_spec(self) -> ModelStateSpec:
        families: list[StateFamilySpec] = []
        if self.n_gdn:
            state_elems = self.n_gdn * self.num_v_heads * self.head_k_dim * self.head_v_dim
            conv_elems = self.n_gdn * self.conv_dim * self.conv_kernel
            families.append(
                StateFamilySpec(
                    name=GDN_STATE_FAMILY,
                    bytes_per_slot=state_elems * GDN_STATE_DTYPE.itemsize,
                    layer_ids=self.gdn_layers,
                )
            )
            families.append(
                StateFamilySpec(
                    name=GDN_CONV_FAMILY,
                    bytes_per_slot=conv_elems * self.dtype.itemsize,
                    layer_ids=self.gdn_layers,
                )
            )
        if self.n_attn and self.guest_mode == "paged":
            families.append(
                StateFamilySpec(
                    name=GUEST_KV_FAMILY,
                    bytes_per_slot=self.bytes_per_page,  # per PAGE for a paged family
                    layer_ids=self.attn_layers,
                    page_tokens=self.page_tokens,
                )
            )
        elif self.n_attn:  # ring / routed: a fixed column space per slot, no pages
            families.append(
                StateFamilySpec(
                    name=GUEST_KV_FAMILY,
                    bytes_per_slot=self.window_tokens * self.guest_bytes_per_token,
                    layer_ids=self.attn_layers,
                )
            )
        if not families:
            raise ValueError("layout produced no state families")
        return ModelStateSpec(families=tuple(families))

    @property
    def bytes_per_slot(self) -> int:
        """Fixed per-request bytes (the recurrent families); guest pages are
        extra, ``bytes_per_page`` per reserved page."""
        return self.state_spec().bytes_per_request

    # -- engine factories (Engine never branches on model family) -----------------

    def make_bank(self, num_slots: int, device, num_pages: int = 0):
        from wkvm.runner.hybrid_state import Qwen35StateBank

        return Qwen35StateBank(self, num_slots=num_slots, device=device, num_pages=num_pages)

    def make_runner(self, model, bank, prefill_chunk: int):
        from wkvm.runner.hybrid_runner import Qwen35HybridRunner

        return Qwen35HybridRunner(model, bank, prefill_chunk=prefill_chunk)

    # -- tuned initial states ------------------------------------------------------

    def load_state_adapter(
        self, path: str | Path, *, strict: bool = True
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """Read an RNN-StateTuning adapter (``adapter_model.safetensors`` with
        ``layers.{i}.recurrent_state`` tensors) into bank-import form.

        Returns ``({"gdn_state": [n_gdn, H_v, K, V] fp32}, metadata)``. The
        adapter's layer indices are model layer ids; they are mapped onto the
        bank's linear-layer order. Per-layer-embedding (PLE) tensors are model
        weights, not state, and are rejected. With ``strict`` every linear
        layer must be present; otherwise missing layers stay zero (= untuned).
        """
        return load_state_adapter(self, path, strict=strict)


def load_state_adapter(
    layout: Qwen35HybridLayout, path: str | Path, *, strict: bool = True
) -> tuple[dict[str, torch.Tensor], dict]:
    from safetensors.torch import load_file

    path = Path(path)
    weights_path = path if path.suffix == ".safetensors" else path / "adapter_model.safetensors"
    config_path = weights_path.with_name("adapter_config.json")
    if not weights_path.is_file():
        raise FileNotFoundError(f"no adapter weights at {weights_path}")
    weights = load_file(str(weights_path), device="cpu")

    ple = [k for k in weights if k.startswith("ple.")]
    if ple:
        raise ValueError(
            f"{weights_path}: contains per-layer-embedding tensors ({len(ple)}); "
            "PLE is model weight, not state, and cannot be imported as a state handle"
        )
    prefix, suffix = "layers.", ".recurrent_state"
    found: dict[int, torch.Tensor] = {}
    for key, value in weights.items():
        if not (key.startswith(prefix) and key.endswith(suffix)):
            raise KeyError(f"{weights_path}: unexpected tensor {key!r}")
        layer_idx = int(key[len(prefix) : -len(suffix)])
        if layer_idx not in layout.gdn_layers:
            raise ValueError(f"{key}: layer {layer_idx} is not a linear_attention layer")
        if tuple(value.shape) != layout.gdn_state_shape:
            raise ValueError(
                f"{key}: shape {tuple(value.shape)} != layout {layout.gdn_state_shape}"
            )
        found[layer_idx] = value
    missing = [i for i in layout.gdn_layers if i not in found]
    if strict and missing:
        raise KeyError(f"{weights_path}: missing recurrent_state for layers {missing}")

    state = torch.zeros((layout.n_gdn, *layout.gdn_state_shape), dtype=GDN_STATE_DTYPE)
    for j, layer_idx in enumerate(layout.gdn_layers):
        if layer_idx in found:
            state[j].copy_(found[layer_idx].to(GDN_STATE_DTYPE))

    meta = {
        "source": str(weights_path),
        "sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "tuned_layers": sorted(found),
        "missing_layers": missing,
    }
    if config_path.is_file():
        cfg = json.loads(config_path.read_text())
        meta["adapter_config"] = {
            k: cfg[k] for k in ("backend", "base_model", "method", "format_version") if k in cfg
        }
    return {GDN_STATE_FAMILY: state}, meta


def _gqa_attention_forward(module, query, key, value, attention_mask, dropout: float = 0.0, scaling=None, **kwargs):
    """HF attention interface ``"wkvm_gqa"``: for a single query token with a
    boolean mask, attend GQA-natively — the KV heads are broadcast over
    their query group in the matmul instead of ``repeat_kv`` materialising
    ``groups`` copies of K and V (with 5k+ columns per row that copy is
    what a decode step costs). Anything else delegates to HF's sdpa path."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if query.shape[2] != 1 or attention_mask is None or attention_mask.dtype != torch.bool:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
    b, heads, _, hd = query.shape
    kvh = key.shape[1]
    q = query.reshape(b, kvh, heads // kvh, hd)  # heads grouped contiguously per KV head, as repeat_kv does
    scores = torch.matmul(q, key.transpose(-1, -2)).float() * (scaling if scaling is not None else hd ** -0.5)
    scores = scores.masked_fill(~attention_mask[:, 0, 0, :][:, None, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(value.dtype)
    out = torch.matmul(probs, value)  # [b, kvh, groups, hd]
    return out.reshape(b, 1, heads, hd), None


def _register_gqa_attention() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if "wkvm_gqa" not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS.register("wkvm_gqa", _gqa_attention_forward)


class Qwen35Decoder(torch.nn.Module):
    """The text stack of a Qwen3.5 checkpoint as wkvm drives it.

    Holds references to the HF submodules (embeddings, decoder layers, final
    norm, rotary embedding, lm_head) and runs the layer loop itself, so that
    masks, positions and the cache object are wkvm's — HF's
    ``Qwen3_5TextModel.forward`` (which would build masks against a
    ``DynamicCache``) is never called.
    """

    def __init__(self, embed_tokens, layers, norm, rotary_emb, lm_head, config) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.layers = layers
        self.norm = norm
        self.rotary_emb = rotary_emb
        self.lm_head = lm_head
        self.config = config
        self.layer_types = tuple(config.layer_types)

    def set_decode_attention(self, gqa: bool) -> None:
        """``gqa=True`` routes single-token decode through the GQA-native
        attention (see ``_gqa_attention_forward``); ``False`` keeps HF's
        sdpa path everywhere (bit-exact with HF generation)."""
        if gqa:
            _register_gqa_attention()
        self.config._attn_implementation = "wkvm_gqa" if gqa else "sdpa"

    @property
    def decode_attention(self) -> str:
        return "gqa" if self.config._attn_implementation == "wkvm_gqa" else "sdpa"

    @classmethod
    def from_hf(cls, hf_model) -> "Qwen35Decoder":
        inner = hf_model.model
        text = getattr(inner, "language_model", inner)
        return cls(
            embed_tokens=text.embed_tokens,
            layers=text.layers,
            norm=text.norm,
            rotary_emb=text.rotary_emb,
            lm_head=hf_model.lm_head,
            config=text.config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cache,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """``input_ids`` [B, T]; ``positions`` [B, T] absolute token positions;
        ``attention_mask`` bool [B, 1, T, KV] (True = attend) or None for
        the mask-free cases. Returns last-position logits [B, vocab]."""
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden, positions)
        for i, layer in enumerate(self.layers):
            mask = attention_mask if self.layer_types[i] == FULL else None
            hidden = layer(
                hidden,
                position_embeddings=(cos, sin),
                attention_mask=mask,
                position_ids=positions,
                past_key_values=cache,
            )
        hidden = self.norm(hidden[:, -1:])
        return self.lm_head(hidden)[:, -1]


def load_qwen35(
    model_path: str,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    page_tokens: int = 256,
    attn_implementation: str = "sdpa",
    drop_vision: bool = True,
    guest_mode: str = "paged",
    sink_tokens: int = 16,
    ring_tokens: int = 1024,
    decode_attention: str = "auto",
    **routed,
):
    """Load a Qwen3.5 checkpoint (text-only or conditional-generation layout)
    for inference. Returns ``(decoder, layout)``; the decoder is frozen, in
    eval mode (the GDN layer dispatches on ``seq_len == 1`` for the recurrent
    kernel, not on ``training``, but eval also disables dropout).

    Kernel path (fla Triton vs pure torch) is decided here, before the
    modeling module is imported — see ``wkvm/runner/kernels.py``.
    """
    from wkvm.runner.kernels import select_kernels

    select_kernels()
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path)
    kwargs = dict(dtype=dtype, attn_implementation=attn_implementation)
    if config.model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration

        hf = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **kwargs)
        if drop_vision and hasattr(hf.model, "visual"):
            del hf.model.visual  # ~0.5B params the text path never touches
    elif config.model_type == "qwen3_5_text":
        from transformers import Qwen3_5ForCausalLM

        hf = Qwen3_5ForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        raise ValueError(f"not a Qwen3.5 checkpoint: model_type={config.model_type!r}")
    hf = hf.to(device).eval().requires_grad_(False)
    decoder = Qwen35Decoder.from_hf(hf)
    decoder._hf_model = hf  # keep the owner alive; submodules are shared
    if decode_attention not in ("auto", "gqa", "sdpa"):
        raise ValueError(f"decode_attention must be auto|gqa|sdpa, got {decode_attention!r}")
    # Approximate guests (ring/routed) take the GQA-native decode attention;
    # paged stays on HF's sdpa so the exactness gates against HF hold.
    decoder.set_decode_attention(decode_attention == "gqa" or (decode_attention == "auto" and guest_mode != "paged"))
    if guest_mode == "routed" and not routed.get("break_token_ids"):
        routed["break_token_ids"] = break_token_ids(model_path)
    layout = Qwen35HybridLayout.from_config(
        decoder.config, dtype=dtype, page_tokens=page_tokens,
        guest_mode=guest_mode, sink_tokens=sink_tokens, ring_tokens=ring_tokens, **routed,
    )
    return decoder, layout


BREAK_CHARS = ".!?;:。！？；：\n"


def break_token_ids(model_path: str) -> tuple[int, ...]:
    """Token ids whose text ends a sentence-like span (routed mode splits
    evicted text into spans at these tokens)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    ids = []
    for i in range(len(tok)):
        text = tok.decode([i])
        if text and text.rstrip(" ") and text.rstrip(" ")[-1] in BREAK_CHARS:
            ids.append(i)
    return tuple(ids)
