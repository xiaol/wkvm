"""Native Gemma4 text layer math used to retire HF forward calls incrementally."""

from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

_TOKEN_POOL_ATTENTION_BACKEND = None
_NATIVE_GEMMA_FUSED_OPS = None
_NATIVE_GEMMA_FUSED_OPS_UNAVAILABLE = False
_NATIVE_GEMMA_FUSED_RMS_NORM_ENABLED = None
_KV_SHARING_FAST_PREFILL_MIN_TAIL_QUERY_LENGTH = 128


_NATIVE_FORWARD_TIMING_STATS: dict[str, float | int] = {
    "layer_forward_calls": 0,
    "layer_forward_wall_s": 0.0,
    "layer_input_norm_wall_s": 0.0,
    "layer_self_attention_wall_s": 0.0,
    "layer_post_attention_norm_wall_s": 0.0,
    "layer_pre_feedforward_norm_wall_s": 0.0,
    "layer_mlp_wall_s": 0.0,
    "layer_post_feedforward_norm_wall_s": 0.0,
    "layer_ple_wall_s": 0.0,
    "self_attention_qkv_proj_wall_s": 0.0,
    "self_attention_q_norm_rope_wall_s": 0.0,
    "self_attention_kv_norm_rope_wall_s": 0.0,
    "self_attention_shared_kv_wall_s": 0.0,
    "self_attention_metadata_wall_s": 0.0,
    "self_attention_cache_update_wall_s": 0.0,
    "self_attention_attention_wall_s": 0.0,
    "self_attention_output_proj_wall_s": 0.0,
    "mlp_gate_up_proj_wall_s": 0.0,
    "mlp_activation_down_proj_wall_s": 0.0,
    "text_embedding_wall_s": 0.0,
    "text_per_layer_input_wall_s": 0.0,
    "text_mask_wall_s": 0.0,
    "text_rotary_wall_s": 0.0,
    "text_layers_wall_s": 0.0,
    "text_final_norm_wall_s": 0.0,
    "lm_head_wall_s": 0.0,
    "lm_head_softcap_wall_s": 0.0,
    "token_pool_attention_calls": 0,
    "token_pool_attention_rows": 0,
    "token_pool_attention_wall_s": 0.0,
    "token_pool_attention_triton_calls": 0,
    "token_pool_attention_triton_wall_s": 0.0,
    "token_pool_attention_reference_calls": 0,
    "token_pool_attention_reference_wall_s": 0.0,
    "token_pool_attention_triton_attempts": 0,
    "token_pool_attention_triton_attempt_wall_s": 0.0,
    "token_pool_kv_write_calls": 0,
    "token_pool_kv_write_tokens": 0,
    "token_pool_kv_write_wall_s": 0.0,
    "dense_gqa_prefill_calls": 0,
    "dense_gqa_prefill_fallbacks": 0,
    "dense_gqa_decode_calls": 0,
    "dense_gqa_decode_fallbacks": 0,
}


def _coerce_env_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return False


def _env_bool(name: str) -> bool | None:
    return _coerce_env_bool(os.environ.get(name))


def _native_forward_timing_enabled() -> bool:
    return _env_bool("WKVM_NATIVE_FORWARD_TIMING") is True


def _slot_count(slot_ids: Any) -> int:
    shape = getattr(slot_ids, "shape", None)
    if shape is not None:
        numel = getattr(slot_ids, "numel", None)
        if numel is not None:
            return int(numel())
        count = 1
        for dim in shape:
            count *= int(dim)
        return int(count)
    if isinstance(slot_ids, (list, tuple)):
        return len(slot_ids)
    return 1


def native_forward_timing_stats() -> dict[str, Any]:
    stats = dict(_NATIVE_FORWARD_TIMING_STATS)
    stats["enabled"] = _native_forward_timing_enabled()
    return stats


def reset_native_forward_timing_stats() -> None:
    for key, value in list(_NATIVE_FORWARD_TIMING_STATS.items()):
        _NATIVE_FORWARD_TIMING_STATS[key] = 0.0 if isinstance(value, float) else 0


def _record_native_timing(name: str, elapsed: float) -> None:
    _NATIVE_FORWARD_TIMING_STATS[name] = (
        float(_NATIVE_FORWARD_TIMING_STATS.get(name, 0.0)) + float(elapsed)
    )


def _record_native_count(name: str, value: int = 1) -> None:
    _NATIVE_FORWARD_TIMING_STATS[name] = int(
        _NATIVE_FORWARD_TIMING_STATS.get(name, 0)
    ) + int(value)


def _record_token_pool_attention_timing(
    kind: str,
    *,
    rows: int,
    elapsed: float,
) -> None:
    _record_native_count("token_pool_attention_calls")
    _record_native_count("token_pool_attention_rows", rows)
    _record_native_timing("token_pool_attention_wall_s", elapsed)
    if kind == "triton":
        _record_native_count("token_pool_attention_triton_calls")
        _record_native_timing("token_pool_attention_triton_wall_s", elapsed)
    elif kind == "reference":
        _record_native_count("token_pool_attention_reference_calls")
        _record_native_timing("token_pool_attention_reference_wall_s", elapsed)


def _record_token_pool_kv_write_timing(*, tokens: int, elapsed: float) -> None:
    _record_native_count("token_pool_kv_write_calls")
    _record_native_count("token_pool_kv_write_tokens", tokens)
    _record_native_timing("token_pool_kv_write_wall_s", elapsed)


def _token_pool_triton_input_precision_policy() -> str:
    from wkvm.runner.gemma_token_pool_attention import (
        token_pool_triton_input_precision_policy,
    )

    return token_pool_triton_input_precision_policy()


def _token_pool_triton_dot_dtype_policy() -> str:
    from wkvm.runner.gemma_token_pool_attention import (
        token_pool_triton_dot_dtype_policy,
    )

    return token_pool_triton_dot_dtype_policy()


def _token_pool_triton_input_precision_policy_from_raw(raw: str | None) -> str:
    from wkvm.runner.gemma_token_pool_attention import (
        token_pool_triton_input_precision_policy_from_raw,
    )

    return token_pool_triton_input_precision_policy_from_raw(raw)


def _token_pool_triton_dot_dtype_policy_from_raw(raw: str | None) -> str:
    from wkvm.runner.gemma_token_pool_attention import (
        token_pool_triton_dot_dtype_policy_from_raw,
    )

    return token_pool_triton_dot_dtype_policy_from_raw(raw)


def _token_pool_triton_dispatch_plan():
    from wkvm.runner.gemma_token_pool_attention import token_pool_triton_dispatch_plan

    return token_pool_triton_dispatch_plan()


def _token_pool_triton_block_groups(groups: int, dtype: Any) -> int:
    from wkvm.runner.gemma_token_pool_attention import (
        token_pool_triton_block_groups,
    )

    return token_pool_triton_block_groups(groups, dtype)


def _token_pool_triton_split_plan(max_seq_len: Any) -> tuple[bool, int, int, int | None]:
    from wkvm.runner.gemma_token_pool import build_token_pool_triton_decode_plan

    plan = build_token_pool_triton_decode_plan(max_seq_len)
    return (
        bool(plan.should_split),
        int(plan.split_size),
        int(plan.min_splits),
        None if plan.max_splits is None else int(plan.max_splits),
    )


def _token_pool_triton_metadata_split_plan(
    metadata,
    max_seq_len: Any,
) -> tuple[bool, int, int, int | None]:
    from wkvm.runner.gemma_token_pool import token_pool_triton_decode_plan_from_metadata

    plan = token_pool_triton_decode_plan_from_metadata(metadata, max_seq_len)
    return (
        bool(plan.should_split),
        int(plan.split_size),
        int(plan.min_splits),
        None if plan.max_splits is None else int(plan.max_splits),
    )


def _token_pool_triton_effective_enabled() -> tuple[bool, bool]:
    from wkvm.runner.gemma_token_pool_attention import (
        token_pool_triton_effective_enabled,
    )

    return token_pool_triton_effective_enabled()


def _token_pool_triton_effective_enabled_from_values(
    enabled: bool | None,
    disabled: bool | None,
) -> tuple[bool, bool]:
    from wkvm.runner.gemma_token_pool_attention import (
        token_pool_triton_effective_enabled_from_values,
    )

    return token_pool_triton_effective_enabled_from_values(enabled, disabled)


def _record_token_pool_triton_attempt_timing(elapsed: float) -> None:
    _record_native_count("token_pool_attention_triton_attempts")
    _record_native_timing("token_pool_attention_triton_attempt_wall_s", elapsed)


def _record_token_pool_attention_backend_timing(
    kind: str,
    rows: int,
    elapsed: float,
) -> None:
    _record_token_pool_attention_timing(kind, rows=rows, elapsed=elapsed)


def _attention_forward_token_pool_gqa_reference_hook(*args, **kwargs):
    return _attention_forward_token_pool_gqa_reference(*args, **kwargs)


def _token_pool_attention_backend():
    global _TOKEN_POOL_ATTENTION_BACKEND
    if _TOKEN_POOL_ATTENTION_BACKEND is None:
        from wkvm.runner.gemma_token_pool_attention import (
            build_token_pool_attention_backend,
        )

        _TOKEN_POOL_ATTENTION_BACKEND = build_token_pool_attention_backend(
            reference_decode=_attention_forward_token_pool_gqa_reference_hook,
            slot_count=_slot_count,
            record_kv_write_timing=_record_token_pool_kv_write_timing,
            record_triton_attempt_timing=_record_token_pool_triton_attempt_timing,
            record_attention_timing=_record_token_pool_attention_backend_timing,
            now=time.perf_counter,
        )
    return _TOKEN_POOL_ATTENTION_BACKEND


def token_pool_triton_stats() -> dict[str, Any]:
    from wkvm.runner.gemma_token_pool_attention import token_pool_triton_stats_report

    return token_pool_triton_stats_report(split_plan=_token_pool_triton_split_plan(None))


def reset_token_pool_triton_stats(*, clear_disabled_shapes: bool = False) -> None:
    global _TOKEN_POOL_ATTENTION_BACKEND
    from wkvm.runner.gemma_token_pool_attention import (
        reset_token_pool_triton_runtime_state,
    )

    _TOKEN_POOL_ATTENTION_BACKEND = None
    reset_token_pool_triton_runtime_state(
        clear_disabled_shapes=clear_disabled_shapes,
    )


def _torch():
    import torch

    return torch


def _rms_norm(hidden_states, norm) -> Any:
    import torch.nn.functional as F

    weight = getattr(norm, "weight", None)
    if weight is not None:
        weight = _tensor_on_device(weight, hidden_states)
    return F.rms_norm(hidden_states, (hidden_states.shape[-1],), weight, norm.eps)


def _native_gemma_fused_ops():
    global _NATIVE_GEMMA_FUSED_OPS, _NATIVE_GEMMA_FUSED_OPS_UNAVAILABLE
    if _NATIVE_GEMMA_FUSED_OPS is not None:
        return _NATIVE_GEMMA_FUSED_OPS
    if _NATIVE_GEMMA_FUSED_OPS_UNAVAILABLE:
        return None
    try:
        from wkvm.runner.gemma_fused_ops import rms_norm_residual_scalar
    except Exception:
        _NATIVE_GEMMA_FUSED_OPS_UNAVAILABLE = True
        return None
    _NATIVE_GEMMA_FUSED_OPS = rms_norm_residual_scalar
    return _NATIVE_GEMMA_FUSED_OPS


def _native_gemma_fused_rms_norm_enabled() -> bool:
    global _NATIVE_GEMMA_FUSED_RMS_NORM_ENABLED
    if _NATIVE_GEMMA_FUSED_RMS_NORM_ENABLED is None:
        enabled = _env_bool("WKVM_ENABLE_NATIVE_GEMMA_FUSED_RMS_NORM")
        if enabled is None:
            enabled = True
        if _env_bool("WKVM_DISABLE_NATIVE_GEMMA_FUSED_RMS_NORM") is True:
            enabled = False
        _NATIVE_GEMMA_FUSED_RMS_NORM_ENABLED = bool(enabled)
    return bool(_NATIVE_GEMMA_FUSED_RMS_NORM_ENABLED)


def _rms_norm_residual_scalar(hidden_states, norm, residual, scalar):
    global _NATIVE_GEMMA_FUSED_OPS, _NATIVE_GEMMA_FUSED_OPS_UNAVAILABLE

    torch = _torch()
    weight = getattr(norm, "weight", None)
    width = int(hidden_states.shape[-1])
    if (
        _native_gemma_fused_rms_norm_enabled()
        and bool(getattr(hidden_states, "is_cuda", False))
        and not torch.is_grad_enabled()
        and not bool(getattr(hidden_states, "requires_grad", False))
        and weight is not None
        and scalar is not None
        and hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and hidden_states.dtype == residual.dtype == weight.dtype == scalar.dtype
        and hidden_states.device == residual.device == weight.device == scalar.device
        and tuple(hidden_states.shape) == tuple(residual.shape)
        and int(weight.numel()) == width
        and int(scalar.numel()) == 1
        and hidden_states.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and scalar.is_contiguous()
        and 0 < width <= 8192
    ):
        fused_op = _native_gemma_fused_ops()
        if fused_op is not None:
            try:
                return fused_op(hidden_states, weight, residual, scalar, norm.eps)
            except Exception:
                _NATIVE_GEMMA_FUSED_OPS = None
                _NATIVE_GEMMA_FUSED_OPS_UNAVAILABLE = True
    hidden_states = _rms_norm(hidden_states, norm)
    return (hidden_states + residual) * _tensor_on_device(scalar, hidden_states)


def _linear(x, linear) -> Any:
    import torch.nn.functional as F

    weight = _tensor_on_device(linear.weight, x)
    bias = _tensor_on_device(getattr(linear, "bias", None), x)
    return F.linear(x, weight, bias)


def _normalize_weight_backend(backend: str) -> str:
    backend = str(backend).strip().lower()
    if backend in {"hf", "live", "hf_live"}:
        backend = "hf_live"
    if backend in {"cpu", "owned-cpu", "owned_cpu"}:
        backend = "owned_cpu"
    if backend not in {"hf_live", "owned", "owned_cpu"}:
        raise ValueError(
            "native Gemma weight backend must be 'hf_live', 'owned', or 'owned_cpu'"
        )
    return backend


def _normalize_projection_backend(backend: str) -> str:
    backend = str(backend).strip().lower()
    if backend == "packed":
        backend = "qkv_gate_up_packed"
    if backend not in {
        "separate",
        "qkv_packed",
        "gate_up_packed",
        "qkv_gate_up_packed",
    }:
        raise ValueError(
            "native Gemma projection backend must be 'separate', 'qkv_packed', "
            "'gate_up_packed', or 'qkv_gate_up_packed'"
        )
    return backend


def _packs_qkv(backend: str) -> bool:
    return _normalize_projection_backend(backend) in {"qkv_packed", "qkv_gate_up_packed"}


def _packs_gate_up(backend: str) -> bool:
    return _normalize_projection_backend(backend) in {
        "gate_up_packed",
        "qkv_gate_up_packed",
    }


def _linear_signature(linear) -> tuple[Any, ...]:
    bias = getattr(linear, "bias", None)
    bias_sig = None
    if bias is not None:
        bias_sig = (
            bias.data_ptr(),
            str(bias.device),
            bias.dtype,
            tuple(bias.shape),
            getattr(bias, "_version", 0),
        )
    return (
        linear.weight.data_ptr(),
        str(linear.weight.device),
        linear.weight.dtype,
        tuple(linear.weight.shape),
        getattr(linear.weight, "_version", 0),
        bias_sig,
    )


def _tensor_on_device(tensor, reference):
    if tensor is None:
        return None
    if tensor.device == reference.device:
        return tensor
    return tensor.to(device=reference.device, non_blocking=True)


def _clone_weight(tensor, *, device=None, clone: bool = True):
    if tensor is None:
        return None
    tensor = tensor.detach()
    if device is not None:
        return tensor.to(device=device, copy=True).contiguous()
    if clone:
        return tensor.clone().contiguous()
    return tensor.contiguous()


def _cat_detached_tensors(tensors, *, dim: int = 0, device=None):
    torch = _torch()

    pieces = []
    for tensor in tensors:
        tensor = tensor.detach()
        if device is not None:
            tensor = tensor.to(device=device, copy=True)
        pieces.append(tensor)
    return torch.cat(pieces, dim=dim).contiguous()


class _TensorLinear:
    """Inference linear that owns a detached checkpoint tensor snapshot."""

    def __init__(
        self,
        linear=None,
        *,
        weight=None,
        bias=None,
        device=None,
        clone: bool = True,
    ) -> None:
        if linear is None and weight is None:
            raise ValueError("_TensorLinear requires a source linear or weight tensor")
        if weight is None:
            weight = linear.weight
        if bias is None and linear is not None:
            bias = getattr(linear, "bias", None)
        self.weight = _clone_weight(weight, device=device, clone=clone)
        self.bias = _clone_weight(bias, device=device, clone=clone)
        self.snapshot_device = device

    def __call__(self, x):
        return _linear(x, self)

    def to(self, *args, **kwargs):
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        if self.snapshot_device is not None:
            self.weight = self.weight.to(device=self.snapshot_device)
            if self.bias is not None:
                self.bias = self.bias.to(device=self.snapshot_device)
        return self


class _TensorRMSNorm:
    """Inference RMSNorm state detached from the source HF module."""

    def __init__(
        self,
        norm=None,
        *,
        weight=None,
        eps=None,
        device=None,
        clone: bool = True,
    ) -> None:
        if norm is None and eps is None:
            raise ValueError("_TensorRMSNorm requires a source norm or eps")
        self.eps = norm.eps if norm is not None else eps
        if weight is None and norm is not None:
            weight = getattr(norm, "weight", None)
        self.weight = _clone_weight(weight, device=device, clone=clone)
        self.snapshot_device = device

    def __call__(self, hidden_states):
        return _rms_norm(hidden_states, self)

    def to(self, *args, **kwargs):
        if self.weight is not None:
            self.weight = self.weight.to(*args, **kwargs)
            if self.snapshot_device is not None:
                self.weight = self.weight.to(device=self.snapshot_device)
        return self


def _snapshot_linear(linear, *, owned: bool, device=None):
    if linear is None or not owned:
        return linear
    return _TensorLinear(linear, device=device)


def _snapshot_norm(norm, *, owned: bool, device=None):
    if norm is None or not owned:
        return norm
    return _TensorRMSNorm(norm, device=device)


class _TensorEmbedding:
    """Scaled embedding backed by checkpoint tensors, without an HF module."""

    def __init__(self, weight, *, padding_idx: int | None = None, embed_scale: float = 1.0) -> None:
        self.weight = weight.detach()
        self.padding_idx = padding_idx
        self.scalar_embed_scale = float(embed_scale)
        self.embed_scale = _torch().tensor(float(embed_scale), device=self.weight.device)

    def __call__(self, input_ids):
        import torch.nn.functional as F

        return F.embedding(input_ids, self.weight, padding_idx=self.padding_idx) * self.embed_scale.to(
            self.weight.dtype
        )

    def to(self, *args, **kwargs):
        self.weight = self.weight.to(*args, **kwargs)
        self.embed_scale = self.embed_scale.to(device=self.weight.device)
        return self


def _checkpoint_tensor(state_dict: dict[str, Any], key: str, *, device=None, dtype=None, required: bool = True):
    tensor = state_dict.get(key)
    if tensor is None:
        if required:
            raise KeyError(f"missing Gemma checkpoint tensor {key!r}")
        return None
    tensor = tensor.detach()
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None and tensor.device != _torch().device(device):
        tensor = tensor.to(device=device)
    return tensor.contiguous()


def _checkpoint_files(model_path: str | Path) -> list[Path]:
    model_path = Path(model_path)
    if model_path.is_file():
        return [model_path]
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        files = sorted({model_path / name for name in index.get("weight_map", {}).values()})
        if files:
            return files
    single = model_path / "model.safetensors"
    if single.exists():
        return [single]
    files = sorted(model_path.glob("*.safetensors"))
    if files:
        return files
    raise FileNotFoundError(f"no safetensors checkpoint files found under {model_path}")


def _load_checkpoint_state_dict(
    model_path: str | Path,
    *,
    prefix: str = "model.language_model",
    device=None,
    dtype=None,
) -> dict[str, Any]:
    from safetensors.torch import safe_open

    torch = _torch()
    if dtype is not None and isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    device_arg = "cpu" if device is None else str(torch.device(device))
    prefix_dot = f"{prefix}."
    state_dict: dict[str, Any] = {}
    for checkpoint_file in _checkpoint_files(model_path):
        with safe_open(str(checkpoint_file), framework="pt", device=device_arg) as f:
            for key in f.keys():
                if not (key.startswith(prefix_dot) or key == "lm_head.weight"):
                    continue
                tensor = f.get_tensor(key)
                if dtype is not None and tensor.dtype != dtype:
                    tensor = tensor.to(dtype=dtype)
                state_dict[key] = tensor.contiguous()
    if f"{prefix}.embed_tokens.weight" not in state_dict:
        raise KeyError(
            f"checkpoint under {model_path!s} does not contain {prefix}.embed_tokens.weight"
        )
    return state_dict


def _linear_from_checkpoint(state_dict: dict[str, Any], key: str, *, device=None, dtype=None, bias_key: str | None = None):
    return _TensorLinear(
        weight=_checkpoint_tensor(state_dict, key, device=device, dtype=dtype),
        bias=(
            _checkpoint_tensor(state_dict, bias_key, device=device, dtype=dtype, required=False)
            if bias_key is not None
            else None
        ),
        clone=False,
    )


def _norm_from_checkpoint(state_dict: dict[str, Any], key: str, *, eps: float, device=None, dtype=None, required: bool = True):
    weight = _checkpoint_tensor(
        state_dict,
        key,
        device=device,
        dtype=dtype,
        required=required,
    )
    return _TensorRMSNorm(weight=weight, eps=eps, clone=False)


class _CheckpointGemma4MLP:
    def __init__(self, config, state_dict: dict[str, Any], prefix: str, *, device=None, dtype=None) -> None:
        self.config = config
        self.gate_proj = _linear_from_checkpoint(
            state_dict,
            f"{prefix}.gate_proj.weight",
            device=device,
            dtype=dtype,
        )
        self.up_proj = _linear_from_checkpoint(
            state_dict,
            f"{prefix}.up_proj.weight",
            device=device,
            dtype=dtype,
        )
        self.down_proj = _linear_from_checkpoint(
            state_dict,
            f"{prefix}.down_proj.weight",
            device=device,
            dtype=dtype,
        )

    def to(self, *args, **kwargs):
        self.gate_proj.to(*args, **kwargs)
        self.up_proj.to(*args, **kwargs)
        self.down_proj.to(*args, **kwargs)
        return self


class _NativeGemma4TextConfig:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = dict(values)
        for key, value in values.items():
            setattr(self, key, value)
        self._attn_implementation = str(
            values.get("_attn_implementation", "eager")
        )
        self._attn_implementation_internal = self._attn_implementation

    def get_text_config(self, decoder: bool = True):
        return self

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def standardize_rope_params(self) -> None:
        return None


def _load_native_gemma4_text_config(model_path: str | Path) -> _NativeGemma4TextConfig:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"checkpoint-native Gemma4 loader requires local config.json under {model_path!s}"
        )
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)
    text_config = raw_config.get("text_config", raw_config)
    if not isinstance(text_config, dict):
        raise TypeError(
            f"Gemma4 config.json under {model_path!s} does not contain a text config"
        )
    return _NativeGemma4TextConfig(text_config)


class _NativeGemma4TextRotaryEmbedding:
    def __init__(self, config, device=None) -> None:
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.layer_types = set(config.layer_types)
        self.rope_type: dict[str, str] = {}
        self._buffer_names: list[str] = []
        for layer_type in self.layer_types:
            rope_params = config.rope_parameters[layer_type]
            if rope_params is None:
                continue
            rope_type = rope_params.get("rope_type", "default")
            self.rope_type[layer_type] = rope_type
            inv_freq, attention_scaling = self._init_inv_freq(
                config,
                rope_params,
                layer_type=layer_type,
                rope_type=rope_type,
                device=device,
            )
            name = f"{layer_type}_inv_freq"
            setattr(self, name, inv_freq)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)
            self._buffer_names.append(name)

    @staticmethod
    def _init_inv_freq(
        config,
        rope_params: dict[str, Any],
        *,
        layer_type: str,
        rope_type: str,
        device=None,
    ):
        torch = _torch()
        base = rope_params["rope_theta"]
        attention_factor = 1.0
        if rope_type == "default":
            dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64).to(
                        device=device,
                        dtype=torch.float,
                    )
                    / dim
                )
            )
            return inv_freq, attention_factor
        if rope_type == "proportional":
            head_dim_key = (
                "global_head_dim" if layer_type == "full_attention" else "head_dim"
            )
            dim = getattr(config, head_dim_key, None) or (
                config.hidden_size // config.num_attention_heads
            )
            rope_proportion = rope_params.get("partial_rotary_factor", 1.0)
            rope_angles = int(rope_proportion * dim // 2)
            inv_freq_rotated = 1.0 / (
                base
                ** (
                    torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(
                        device=device,
                        dtype=torch.float,
                    )
                    / dim
                )
            )
            nope_angles = dim // 2 - rope_angles
            if nope_angles > 0:
                inv_freq = torch.cat(
                    (
                        inv_freq_rotated,
                        torch.zeros(nope_angles, dtype=torch.float32, device=device),
                    ),
                    dim=0,
                )
            else:
                inv_freq = inv_freq_rotated
            inv_freq = inv_freq / rope_params.get("factor", 1.0)
            return inv_freq, attention_factor
        raise NotImplementedError(f"unsupported Gemma4 RoPE type: {rope_type}")

    def to(self, *args, **kwargs):
        for name in self._buffer_names:
            setattr(self, name, getattr(self, name).to(*args, **kwargs))
        return self

    def __call__(self, x, position_ids, layer_type=None):
        torch = _torch()
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = (
            inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.no_grad():
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class _CheckpointGemma4Attention:
    def __init__(
        self,
        config,
        layer_idx: int,
        state_dict: dict[str, Any],
        prefix: str,
        *,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        self.layer_idx = int(layer_idx)
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        global_head_dim = getattr(config, "global_head_dim", None)
        self.head_dim = global_head_dim if not self.is_sliding and global_head_dim else config.head_dim
        self.use_alternative_attention = bool(getattr(config, "attention_k_eq_v", False) and not self.is_sliding)
        num_key_value_heads = (
            getattr(config, "num_global_key_value_heads", None)
            if self.use_alternative_attention
            else config.num_key_value_heads
        )
        num_key_value_heads = num_key_value_heads or config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = config.attention_dropout
        self.training = False
        self.is_causal = getattr(config, "use_bidirectional_attention", None) != "all"

        self.kv_shared_layer_index = _gemma4_kv_shared_layer_index(
            config,
            layer_idx,
        )
        first_kv_shared_layer_idx = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
        self.is_kv_shared_layer = self.kv_shared_layer_index is not None
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        self.store_full_length_kv = (
            not self.is_kv_shared_layer
            and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
        )

        self.q_proj = _linear_from_checkpoint(
            state_dict,
            f"{prefix}.q_proj.weight",
            device=device,
            dtype=dtype,
            bias_key=f"{prefix}.q_proj.bias" if getattr(config, "attention_bias", False) else None,
        )
        self.q_norm = _norm_from_checkpoint(
            state_dict,
            f"{prefix}.q_norm.weight",
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.k_norm = _norm_from_checkpoint(
            state_dict,
            f"{prefix}.k_norm.weight",
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
            required=False,
        )
        self.v_norm = _norm_from_checkpoint(
            state_dict,
            f"{prefix}.v_norm.weight",
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
            required=False,
        )
        if self.is_kv_shared_layer:
            self.k_proj = None
            self.v_proj = None
        else:
            self.k_proj = _linear_from_checkpoint(
                state_dict,
                f"{prefix}.k_proj.weight",
                device=device,
                dtype=dtype,
                bias_key=f"{prefix}.k_proj.bias" if getattr(config, "attention_bias", False) else None,
            )
            v_weight_key = f"{prefix}.v_proj.weight"
            self.v_proj = (
                None
                if self.use_alternative_attention and v_weight_key not in state_dict
                else _linear_from_checkpoint(
                    state_dict,
                    v_weight_key,
                    device=device,
                    dtype=dtype,
                    bias_key=f"{prefix}.v_proj.bias" if getattr(config, "attention_bias", False) else None,
                )
            )
        self.o_proj = _linear_from_checkpoint(
            state_dict,
            f"{prefix}.o_proj.weight",
            device=device,
            dtype=dtype,
            bias_key=f"{prefix}.o_proj.bias" if getattr(config, "attention_bias", False) else None,
        )

    def to(self, *args, **kwargs):
        for obj in (
            self.q_proj,
            self.q_norm,
            self.k_norm,
            self.v_norm,
            self.k_proj,
            self.v_proj,
            self.o_proj,
        ):
            to = getattr(obj, "to", None)
            if to is not None:
                to(*args, **kwargs)
        return self


class _CheckpointGemma4TextDecoderLayer:
    def __init__(
        self,
        config,
        layer_idx: int,
        state_dict: dict[str, Any],
        prefix: str,
        *,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(layer_idx)
        self.self_attn = _CheckpointGemma4Attention(
            config,
            layer_idx,
            state_dict,
            f"{prefix}.self_attn",
            device=device,
            dtype=dtype,
        )
        self.mlp = _CheckpointGemma4MLP(
            config,
            state_dict,
            f"{prefix}.mlp",
            device=device,
            dtype=dtype,
        )
        self.input_layernorm = _norm_from_checkpoint(
            state_dict,
            f"{prefix}.input_layernorm.weight",
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.post_attention_layernorm = _norm_from_checkpoint(
            state_dict,
            f"{prefix}.post_attention_layernorm.weight",
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.pre_feedforward_layernorm = _norm_from_checkpoint(
            state_dict,
            f"{prefix}.pre_feedforward_layernorm.weight",
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.post_feedforward_layernorm = _norm_from_checkpoint(
            state_dict,
            f"{prefix}.post_feedforward_layernorm.weight",
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.layer_scalar = _checkpoint_tensor(
            state_dict,
            f"{prefix}.layer_scalar",
            device=device,
            dtype=dtype,
        )
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = _linear_from_checkpoint(
                state_dict,
                f"{prefix}.per_layer_input_gate.weight",
                device=device,
                dtype=dtype,
            )
            self.per_layer_projection = _linear_from_checkpoint(
                state_dict,
                f"{prefix}.per_layer_projection.weight",
                device=device,
                dtype=dtype,
            )
            self.post_per_layer_input_norm = _norm_from_checkpoint(
                state_dict,
                f"{prefix}.post_per_layer_input_norm.weight",
                eps=config.rms_norm_eps,
                device=device,
                dtype=dtype,
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None
        self.enable_moe_block = bool(getattr(config, "enable_moe_block", False))

    def to(self, *args, **kwargs):
        for obj in (
            self.self_attn,
            self.mlp,
            self.input_layernorm,
            self.post_attention_layernorm,
            self.pre_feedforward_layernorm,
            self.post_feedforward_layernorm,
            self.per_layer_input_gate,
            self.per_layer_projection,
            self.post_per_layer_input_norm,
        ):
            to = getattr(obj, "to", None)
            if to is not None:
                to(*args, **kwargs)
        self.layer_scalar = self.layer_scalar.to(*args, **kwargs)
        return self


class _CheckpointGemma4TextModel:
    def __init__(
        self,
        config,
        state_dict: dict[str, Any],
        *,
        prefix: str = "model.language_model",
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.training = False
        self.embed_tokens = _TensorEmbedding(
            _checkpoint_tensor(
                state_dict,
                f"{prefix}.embed_tokens.weight",
                device=device,
                dtype=dtype,
            ),
            padding_idx=self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = [
            _CheckpointGemma4TextDecoderLayer(
                config,
                layer_idx,
                state_dict,
                f"{prefix}.layers.{layer_idx}",
                device=device,
                dtype=dtype,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = _norm_from_checkpoint(
            state_dict,
            f"{prefix}.norm.weight",
            eps=config.rms_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.rotary_emb = _NativeGemma4TextRotaryEmbedding(config, device=device)
        self.unique_layer_types = set(config.layer_types)
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = _TensorEmbedding(
                _checkpoint_tensor(
                    state_dict,
                    f"{prefix}.embed_tokens_per_layer.weight",
                    device=device,
                    dtype=dtype,
                ),
                padding_idx=self.padding_idx,
                embed_scale=config.hidden_size_per_layer_input**0.5,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = _linear_from_checkpoint(
                state_dict,
                f"{prefix}.per_layer_model_projection.weight",
                device=device,
                dtype=dtype,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = _norm_from_checkpoint(
                state_dict,
                f"{prefix}.per_layer_projection_norm.weight",
                eps=config.rms_norm_eps,
                device=device,
                dtype=dtype,
            )

    def get_per_layer_inputs(self, input_ids, inputs_embeds):
        if not self.hidden_size_per_layer_input:
            raise RuntimeError("Gemma4 config does not support per-layer embeddings")
        if input_ids is None:
            raise RuntimeError(
                "checkpoint-native Gemma4 requires input_ids when per-layer embeddings are enabled"
            )
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs=None):
        if not self.hidden_size_per_layer_input:
            raise RuntimeError("Gemma4 config does not support per-layer embeddings")
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        if per_layer_inputs is None:
            return per_layer_projection
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale

    def parameters(self):
        seen: set[int] = set()
        for value in self._iter_tensors():
            ptr = value.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            yield value

    def named_parameters(self, prefix: str = "model"):
        for idx, value in enumerate(self.parameters()):
            yield f"{prefix}.checkpoint_tensor_{idx}", value

    def buffers(self):
        return iter(())

    @property
    def device(self):
        return next(self.parameters()).device

    def eval(self):
        return self.train(False)

    def train(self, mode: bool = True):
        self.training = bool(mode)
        for layer in self.layers:
            layer.self_attn.training = bool(mode)
        return self

    def to(self, *args, **kwargs):
        for obj in (self.embed_tokens, self.norm, self.rotary_emb):
            obj.to(*args, **kwargs)
        if self.hidden_size_per_layer_input:
            for obj in (
                self.embed_tokens_per_layer,
                self.per_layer_model_projection,
                self.per_layer_projection_norm,
            ):
                obj.to(*args, **kwargs)
        for layer in self.layers:
            layer.to(*args, **kwargs)
        return self

    def _iter_tensors(self):
        def visit(obj):
            if obj is None:
                return
            if hasattr(obj, "weight") and getattr(obj, "weight") is not None:
                yield obj.weight
            if hasattr(obj, "bias") and getattr(obj, "bias") is not None:
                yield obj.bias
            if hasattr(obj, "layer_scalar"):
                yield obj.layer_scalar
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    yield from visit(item)
                return
            for attr in (
                "embed_tokens",
                "embed_tokens_per_layer",
                "per_layer_model_projection",
                "per_layer_projection_norm",
                "norm",
                "self_attn",
                "mlp",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "q_norm",
                "k_norm",
                "v_norm",
                "gate_proj",
                "up_proj",
                "down_proj",
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "per_layer_input_gate",
                "per_layer_projection",
                "post_per_layer_input_norm",
            ):
                if hasattr(obj, attr):
                    yield from visit(getattr(obj, attr))

        yield from visit(self)
        yield from visit(self.layers)


class _CheckpointGemma4ForCausalLMFacade:
    def __init__(
        self,
        config,
        state_dict: dict[str, Any],
        *,
        prefix: str = "model.language_model",
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        self.model = _CheckpointGemma4TextModel(
            config,
            state_dict,
            prefix=prefix,
            device=device,
            dtype=dtype,
        )
        lm_head_key = "lm_head.weight"
        lm_head_weight = _checkpoint_tensor(
            state_dict,
            lm_head_key,
            device=device,
            dtype=dtype,
            required=False,
        )
        self.tie_word_embeddings = lm_head_weight is None
        if lm_head_weight is None:
            lm_head_weight = self.model.embed_tokens.weight
        self.lm_head = _TensorLinear(weight=lm_head_weight, clone=False)
        self.training = False

    @property
    def device(self):
        return self.model.device

    def parameters(self, *args, **kwargs):
        seen: set[int] = set()
        for tensor in self.model.parameters():
            seen.add(tensor.data_ptr())
            yield tensor
        for tensor in (self.lm_head.weight, self.lm_head.bias):
            if tensor is None or tensor.data_ptr() in seen:
                continue
            seen.add(tensor.data_ptr())
            yield tensor

    def named_parameters(self, *args, **kwargs):
        seen: set[int] = set()
        for name, tensor in self.model.named_parameters():
            seen.add(tensor.data_ptr())
            yield name, tensor
        for name, tensor in (("lm_head.weight", self.lm_head.weight), ("lm_head.bias", self.lm_head.bias)):
            if tensor is None or tensor.data_ptr() in seen:
                continue
            seen.add(tensor.data_ptr())
            yield name, tensor

    def buffers(self, *args, **kwargs):
        return self.model.buffers()

    def eval(self):
        return self.train(False)

    def train(self, mode: bool = True):
        self.training = bool(mode)
        self.model.train(mode)
        return self

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        self.lm_head.to(*args, **kwargs)
        return self


class _PackedLinear:
    """Concat adjacent HF linear weights for inference-time packed GEMMs."""

    def __init__(self, *linears, cache: bool = True) -> None:
        if len(linears) < 2:
            raise ValueError("_PackedLinear requires at least two linears")
        self.linears = linears
        self.out_features = [int(linear.weight.shape[0]) for linear in linears]
        self.cache = bool(cache)
        self._signature: tuple[Any, ...] | None = None
        self._weight = None
        self._bias = None

    def _build(self):
        torch = _torch()

        weight = torch.cat(
            [linear.weight.detach() for linear in self.linears],
            dim=0,
        ).contiguous()
        if all(getattr(linear, "bias", None) is None for linear in self.linears):
            bias = None
        else:
            pieces = []
            for linear in self.linears:
                bias_piece = getattr(linear, "bias", None)
                if bias_piece is None:
                    pieces.append(linear.weight.new_zeros(linear.weight.shape[0]))
                else:
                    pieces.append(bias_piece.detach())
            bias = torch.cat(pieces, dim=0).contiguous()
        return weight, bias

    def _refresh(self) -> None:
        signature = tuple(_linear_signature(linear) for linear in self.linears)
        if signature == self._signature:
            return
        with _torch().no_grad():
            self._weight, self._bias = self._build()
        self._signature = signature

    def __call__(self, x):
        import torch.nn.functional as F

        if self.cache:
            self._refresh()
            weight, bias = self._weight, self._bias
        else:
            weight, bias = self._build()
        return F.linear(x, weight, bias).split(self.out_features, dim=-1)


class _OwnedPackedLinear:
    """Packed inference linear with copied adjacent projection tensors.

    This mirrors the vLLM/SGLang checkpoint-loaded boundary more closely than
    `_PackedLinear`: the packed weight is owned by the native bridge and is not
    rebuilt from live HF modules during forward.
    """

    def __init__(self, *linears, device=None) -> None:
        if len(linears) < 2:
            raise ValueError("_OwnedPackedLinear requires at least two linears")
        self.out_features = [int(linear.weight.shape[0]) for linear in linears]
        self.weight = _cat_detached_tensors(
            [linear.weight for linear in linears],
            dim=0,
            device=device,
        )
        if all(getattr(linear, "bias", None) is None for linear in linears):
            self.bias = None
        else:
            pieces = []
            for linear in linears:
                bias = getattr(linear, "bias", None)
                if bias is None:
                    zeros = linear.weight.new_zeros(linear.weight.shape[0])
                    if device is not None:
                        zeros = zeros.to(device=device)
                    pieces.append(zeros)
                else:
                    if device is not None:
                        bias = bias.detach().to(device=device, copy=True)
                    else:
                        bias = bias.detach()
                    pieces.append(bias)
            self.bias = _torch().cat(pieces, dim=0).contiguous()
        self.snapshot_device = device

    def __call__(self, x):
        import torch.nn.functional as F

        weight = _tensor_on_device(self.weight, x)
        bias = _tensor_on_device(self.bias, x)
        return F.linear(x, weight, bias).split(self.out_features, dim=-1)

    def to(self, *args, **kwargs):
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        if self.snapshot_device is not None:
            self.weight = self.weight.to(device=self.snapshot_device)
            if self.bias is not None:
                self.bias = self.bias.to(device=self.snapshot_device)
        return self


def _activation(name: str, x) -> Any:
    import torch.nn.functional as F

    if name == "gelu_pytorch_tanh":
        return F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu(x)
    if name in {"silu", "swish"}:
        return F.silu(x)
    if name == "relu":
        return F.relu(x)
    try:
        from transformers.activations import ACT2FN
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise NotImplementedError(f"unsupported Gemma activation: {name}") from exc
    return ACT2FN[name](x)


def _activation_mul(name: str, gate, up) -> Any:
    if getattr(gate, "requires_grad", False):
        return _activation(name, gate) * up
    if name == "gelu_pytorch_tanh":
        _torch().ops.aten.gelu_(gate, approximate="tanh")
        return gate.mul_(up)
    if name == "gelu":
        _torch().ops.aten.gelu_(gate, approximate="none")
        return gate.mul_(up)
    if name in {"silu", "swish"}:
        import torch.nn.functional as F

        return F.silu(gate, inplace=True).mul_(up)
    return _activation(name, gate) * up


def _rotate_half(x):
    torch = _torch()

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(x, cos, sin, *, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half(x) * sin)


def _repeat_kv(hidden_states, repeats: int):
    if repeats == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        repeats,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * repeats, seq_len, head_dim)


def _normalize_attention_backend(backend: str) -> str:
    backend = str(backend).strip().lower()
    if backend not in {
        "manual",
        "manual_gqa",
        "sdpa",
        "sdpa_single_gqa",
        "triton_dense_gqa",
    }:
        raise ValueError(
            "native Gemma attention backend must be 'manual', 'manual_gqa', "
            "'sdpa', 'sdpa_single_gqa', or 'triton_dense_gqa'"
        )
    return backend


def _attention_forward_manual_gqa(attn, query_states, key_states, value_states, attention_mask):
    """Manual attention that keeps grouped-query K/V heads unexpanded."""

    torch = _torch()
    import torch.nn.functional as F

    batch, query_heads, query_length, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    key_length = key_states.shape[2]
    if query_heads % kv_heads != 0:
        raise ValueError("query head count must be divisible by key/value head count")
    groups = query_heads // kv_heads
    query = query_states.reshape(batch, kv_heads, groups, query_length, head_dim)
    key_transposed = key_states.transpose(-2, -1)
    attn_weights = torch.stack(
        [
            torch.matmul(query[:, :, group], key_transposed)
            for group in range(groups)
        ],
        dim=2,
    ) * attn.scaling
    if attention_mask is not None:
        mask = attention_mask
        if mask.ndim == 4:
            mask = mask.unsqueeze(2)
        else:
            while mask.ndim < attn_weights.ndim:
                mask = mask.unsqueeze(1)
        if mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        else:
            attn_weights = attn_weights + mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if attn.training and attn.attention_dropout:
        attn_weights = F.dropout(attn_weights, p=attn.attention_dropout, training=True)
    attn_output = torch.stack(
        [
            torch.matmul(attn_weights[:, :, group], value_states)
            for group in range(groups)
        ],
        dim=2,
    )
    attn_output = attn_output.reshape(batch, query_heads, query_length, head_dim)
    attn_weights = attn_weights.reshape(batch, query_heads, query_length, key_length)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


def _attention_forward_token_pool_gqa_reference(
    attn,
    query_states,
    *,
    decode_metadata,
    token_kv_pool,
    layer_idx: int,
):
    torch = _torch()

    if query_states.shape[2] != 1:
        raise ValueError("token-pool attention only supports decode query_len == 1")
    if query_states.shape[0] != int(decode_metadata.req_pool_indices.numel()):
        raise ValueError("token-pool metadata row count must match query batch")

    kv_indptr = decode_metadata.kv_indptr.detach().cpu().reshape(-1).tolist()
    kv_indices = decode_metadata.kv_indices
    flat_keys, flat_values = token_kv_pool.gather_kv(layer_idx, kv_indices)
    outputs = []
    for row in range(query_states.shape[0]):
        start = int(kv_indptr[row])
        end = int(kv_indptr[row + 1])
        if end <= start:
            raise ValueError("token-pool attention rows must contain at least one KV token")
        key_states = flat_keys[start:end].permute(1, 0, 2).unsqueeze(0)
        value_states = flat_values[start:end].permute(1, 0, 2).unsqueeze(0)
        row_output, _ = _attention_forward_manual_gqa(
            attn,
            query_states[row : row + 1],
            key_states,
            value_states,
            None,
        )
        outputs.append(row_output)
    return torch.cat(outputs, dim=0), None


def _attention_forward_token_pool_gqa(
    attn,
    query_states,
    *,
    decode_metadata,
    paged_decode_metadata=None,
    token_kv_pool,
    layer_idx: int,
    token_pool_plan=None,
    current_key_states=None,
    current_value_states=None,
):
    timing_enabled = _native_forward_timing_enabled()
    from wkvm.runner.gemma_token_pool import build_token_pool_attention_call

    attention_call = build_token_pool_attention_call(
        token_pool_plan=token_pool_plan,
        decode_metadata=decode_metadata,
        paged_decode_metadata=paged_decode_metadata,
        token_kv_pool=token_kv_pool,
        layer_idx=layer_idx,
        attention_mask_present=False,
        query_seq_len=query_states.shape[2],
    ).with_current_kv(
        current_key_states,
        current_value_states,
    )
    result = _token_pool_attention_backend().decode_call(
        attn,
        query_states,
        attention_call=attention_call,
        timing_enabled=timing_enabled,
    )
    return result.output, result.weights


def _attention_forward_token_pool_mixed(
    attn,
    query_states,
    key_states,
    value_states,
    attention_mask,
    *,
    backend: str,
    token_pool_attention_call,
):
    """Run q=1 rows from token-pool KV and prefill rows from dense KV."""

    torch = _torch()
    row_count = int(query_states.shape[0])
    q_lens = tuple(int(value) for value in token_pool_attention_call.q_lens)
    if len(q_lens) != row_count:
        raise ValueError("token-pool mixed q_lens must match query batch")
    if any(q_len < 1 or q_len > int(query_states.shape[2]) for q_len in q_lens):
        raise ValueError("token-pool mixed q_lens exceed the padded query width")
    decode_rows = torch.as_tensor(
        token_pool_attention_call.decode_row_indices,
        dtype=torch.long,
        device=query_states.device,
    )
    prefill_rows = torch.as_tensor(
        token_pool_attention_call.prefill_row_indices,
        dtype=torch.long,
        device=query_states.device,
    )
    decode_query = query_states.index_select(0, decode_rows)[:, :, :1, :]
    decode_result = _token_pool_attention_backend().decode_call(
        attn,
        decode_query,
        attention_call=token_pool_attention_call.backend_decode_call(),
        timing_enabled=_native_forward_timing_enabled(),
    )

    if key_states is None or value_states is None:
        raise RuntimeError("token-pool mixed prefill requires dense key/value states")
    prefill_query = query_states.index_select(0, prefill_rows)
    prefill_keys = key_states.index_select(0, prefill_rows)
    prefill_values = value_states.index_select(0, prefill_rows)
    prefill_mask = attention_mask
    if prefill_mask is not None and int(prefill_mask.shape[0]) == row_count:
        prefill_mask = prefill_mask.index_select(0, prefill_rows)
    from wkvm.runner.gemma_token_pool import TokenPoolAttentionCall

    prefill_output, _ = _attention_forward(
        attn,
        prefill_query,
        prefill_keys,
        prefill_values,
        prefill_mask,
        backend=backend,
        token_pool_attention_call=TokenPoolAttentionCall(),
    )

    output = torch.zeros(
        (
            row_count,
            int(query_states.shape[2]),
            int(query_states.shape[1]),
            int(query_states.shape[3]),
        ),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    output.index_copy_(0, prefill_rows, prefill_output)
    decode_output = torch.zeros(
        (
            int(decode_rows.numel()),
            int(query_states.shape[2]),
            int(query_states.shape[1]),
            int(query_states.shape[3]),
        ),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    decode_output[:, :1].copy_(decode_result.output)
    output.index_copy_(0, decode_rows, decode_output)
    return output, None


def _is_recoverable_token_pool_triton_error(exc: RuntimeError) -> bool:
    from wkvm.runner.gemma_token_pool_attention import (
        is_recoverable_token_pool_triton_error,
    )

    return is_recoverable_token_pool_triton_error(exc)


def _attention_forward(
    attn,
    query_states,
    key_states,
    value_states,
    attention_mask,
    *,
    backend: str,
    decode_metadata=None,
    paged_decode_metadata=None,
    token_kv_pool=None,
    layer_idx: int | None = None,
    token_pool_plan=None,
    token_pool_attention_call=None,
    current_key_states=None,
    current_value_states=None,
):
    torch = _torch()
    import torch.nn.functional as F

    backend = _normalize_attention_backend(backend)
    from wkvm.runner.gemma_token_pool import build_token_pool_attention_call

    if token_pool_attention_call is None:
        token_pool_attention_call = build_token_pool_attention_call(
            token_pool_plan=token_pool_plan,
            decode_metadata=decode_metadata,
            paged_decode_metadata=paged_decode_metadata,
            token_kv_pool=token_kv_pool,
            layer_idx=layer_idx,
            attention_mask_present=attention_mask is not None,
            query_seq_len=query_states.shape[2],
        )
        token_pool_attention_call = token_pool_attention_call.with_current_kv(
            current_key_states,
            current_value_states,
        )
    if bool(getattr(token_pool_attention_call, "mixed_attention_enabled", False)):
        return _attention_forward_token_pool_mixed(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            backend=backend,
            token_pool_attention_call=token_pool_attention_call,
        )
    result = _token_pool_attention_backend().try_decode_call(
        attn,
        query_states,
        attention_call=token_pool_attention_call,
        timing_enabled=_native_forward_timing_enabled(),
    )
    if result is not None:
        return result.output, result.weights
    if key_states is None or value_states is None:
        raise RuntimeError("dense attention fallback requires key/value states")
    if backend == "manual_gqa" or (
        backend == "sdpa_single_gqa"
        and query_states.shape[2] == 1
        and attn.num_key_value_groups > 1
    ):
        return _attention_forward_manual_gqa(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
        )
    if backend == "sdpa_single_gqa":
        backend = "sdpa"
    if backend == "triton_dense_gqa":
        is_decode = query_states.shape[2] == 1
        if (
            attn.num_key_value_groups > 1
            and bool(getattr(query_states, "is_cuda", False))
            and not attn.training
            and not torch.is_grad_enabled()
        ):
            if is_decode:
                from wkvm.runner.gemma_token_pool_triton import dense_padded_gqa_decode

                attn_output = dense_padded_gqa_decode(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    num_key_value_groups=attn.num_key_value_groups,
                    scaling=attn.scaling,
                )
                _record_native_count("dense_gqa_decode_calls")
            else:
                from wkvm.runner.gemma_token_pool_triton import dense_gqa_prefill

                attn_output = dense_gqa_prefill(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    num_key_value_groups=attn.num_key_value_groups,
                    scaling=attn.scaling,
                )
                _record_native_count("dense_gqa_prefill_calls")
            return attn_output, None
        _record_native_count(
            "dense_gqa_decode_fallbacks" if is_decode else "dense_gqa_prefill_fallbacks"
        )
        backend = "sdpa"
    key_states = _repeat_kv(key_states, attn.num_key_value_groups)
    value_states = _repeat_kv(value_states, attn.num_key_value_groups)
    if backend == "sdpa":
        dropout_p = attn.attention_dropout if attn.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=attn.scaling,
        )
        return attn_output.transpose(1, 2).contiguous(), None

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attn.scaling
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(
                ~attention_mask,
                float("-inf"),
            )
        else:
            attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    if attn.training and attn.attention_dropout:
        attn_weights = F.dropout(attn_weights, p=attn.attention_dropout, training=True)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


def _gemma4_kv_shared_layer_index(config: Any, layer_idx: int) -> int | None:
    num_kv_shared_layers = int(getattr(config, "num_kv_shared_layers", 0) or 0)
    if num_kv_shared_layers <= 0:
        return None
    first_kv_shared_layer_idx = int(config.num_hidden_layers) - num_kv_shared_layers
    if first_kv_shared_layer_idx < 0 or layer_idx < first_kv_shared_layer_idx:
        return None
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return None
    prev_layers = layer_types[:first_kv_shared_layer_idx]
    current_layer_type = layer_types[layer_idx]
    return len(prev_layers) - 1 - prev_layers[::-1].index(current_layer_type)


def _normalize_wkvm_logits_indices(
    indices,
    *,
    batch_size: int,
    query_length: int,
    device,
):
    torch = _torch()
    normalized = torch.as_tensor(
        indices,
        dtype=torch.long,
        device=device,
    ).reshape(-1)
    if normalized.numel() != int(batch_size):
        raise ValueError("wkvm_logits_indices must contain one index per batch row")
    if bool(torch.any(normalized < 0)) or bool(
        torch.any(normalized >= int(query_length))
    ):
        raise ValueError("wkvm_logits_indices contains an invalid query index")
    return normalized


def _gather_batch_query_tensor(tensor, indices):
    """Gather one query position per batch row while allowing batch broadcast."""

    torch = _torch()
    batch_size = int(indices.numel())
    if tensor.ndim < 2 or int(tensor.shape[0]) not in {1, batch_size}:
        raise ValueError("fast-prefill tensor is not batch/query aligned")
    rows = torch.arange(batch_size, dtype=torch.long, device=indices.device)
    if int(tensor.shape[0]) == 1:
        rows = torch.zeros_like(rows)
    return tensor[rows, indices].unsqueeze(1)


def _gather_attention_mask_query_rows(attention_mask, indices, query_length: int):
    if attention_mask is None:
        return None
    batch_size = int(indices.numel())
    if (
        attention_mask.ndim != 4
        or int(attention_mask.shape[0]) not in {1, batch_size}
        or int(attention_mask.shape[-2]) != int(query_length)
    ):
        raise ValueError("fast-prefill attention mask is not batch/query aligned")
    torch = _torch()
    rows = torch.arange(batch_size, dtype=torch.long, device=indices.device)
    if int(attention_mask.shape[0]) == 1:
        rows = torch.zeros_like(rows)
    return attention_mask[rows, :, indices, :].unsqueeze(-2)


@dataclass
class NativeGemma4PrefixOutput:
    hidden_states: Any
    past_key_values: Any | None
    shared_kv_states: dict[str, tuple[Any, Any]] | UserDict
    per_layer_inputs: Any | None
    position_ids: Any
    causal_mask_mapping: dict[str, Any]
    kv_sharing_owner_only: bool = False


@dataclass
class _NativeAttentionMeta:
    head_dim: int
    num_key_value_groups: int
    attention_dropout: float
    training: bool
    scaling: float
    is_kv_shared_layer: bool
    layer_type: str | None
    kv_shared_layer_index: int | None
    store_full_length_kv: bool


@dataclass
class NativeGemma4CausalLMOutput:
    logits: Any
    hidden_states: Any
    past_key_values: Any | None
    shared_kv_states: dict[str, tuple[Any, Any]] | UserDict


class NativeGemma4SharedKVState:
    """Shared-KV handoff for native Gemma4 attention."""

    def load_shared_kv(
        self,
        attn: _NativeAttentionMeta,
        shared_kv_states: dict[str, tuple[Any, Any]] | UserDict,
        *,
        query_device,
        past_key_values=None,
        timing_enabled: bool,
    ) -> tuple[Any, Any] | None:
        if not attn.is_kv_shared_layer:
            return None
        phase_start = time.perf_counter() if timing_enabled else 0.0
        shared_kv = None
        get_shared_kv = getattr(past_key_values, "get_shared_kv", None)
        if callable(get_shared_kv) and attn.kv_shared_layer_index is not None:
            shared_kv = get_shared_kv(
                layer_idx=int(attn.kv_shared_layer_index),
                layer_type=attn.layer_type,
            )
        if shared_kv is None:
            if attn.layer_type not in shared_kv_states:
                raise KeyError(
                    f"missing shared Gemma4 KV state for layer type {attn.layer_type!r}"
                )
            shared_kv = shared_kv_states[attn.layer_type]
        key_states, value_states = shared_kv
        key_states = key_states.to(query_device)
        value_states = value_states.to(query_device)
        if timing_enabled:
            _record_native_timing(
                "self_attention_shared_kv_wall_s",
                time.perf_counter() - phase_start,
            )
        return key_states, value_states

    def store_shared_kv(
        self,
        attn: _NativeAttentionMeta,
        shared_kv_states: dict[str, tuple[Any, Any]] | UserDict,
        key_states,
        value_states,
        *,
        layer_idx: int | None = None,
        past_key_values=None,
    ) -> None:
        if not attn.store_full_length_kv:
            return
        shared_kv_states[attn.layer_type] = key_states, value_states
        store_shared_kv = getattr(past_key_values, "store_shared_kv", None)
        if callable(store_shared_kv) and layer_idx is not None:
            store_shared_kv(
                layer_idx=int(layer_idx),
                layer_type=attn.layer_type,
                key_states=key_states,
                value_states=value_states,
            )


class NativeGemma4AttentionBackend:
    """Native attention dispatch wrapper for dense cache and token-pool backends."""

    def __init__(self, native_attention_backend: str) -> None:
        self.native_attention_backend = _normalize_attention_backend(
            native_attention_backend
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def resolve_attention_call(
        self,
        attn: _NativeAttentionMeta,
        *,
        layer_idx: int,
        query_states,
        attention_mask,
        wkvm_token_pool_decode,
        timing_enabled: bool,
    ):
        phase_start = time.perf_counter() if timing_enabled else 0.0
        from wkvm.runner.gemma_token_pool import resolve_token_pool_attention_call

        token_pool_layer_idx = layer_idx
        if (
            wkvm_token_pool_decode is not None
            and attn.is_kv_shared_layer
            and attn.kv_shared_layer_index is not None
        ):
            token_pool_layer_idx = int(attn.kv_shared_layer_index)
        token_pool_attention_call = resolve_token_pool_attention_call(
            wkvm_token_pool_decode,
            token_pool_layer_idx,
            attn.layer_type,
            attention_mask_present=attention_mask is not None,
            query_seq_len=query_states.shape[2],
        )
        if timing_enabled:
            _record_native_timing(
                "self_attention_metadata_wall_s",
                time.perf_counter() - phase_start,
            )
        return token_pool_attention_call

    def forward(
        self,
        attn: _NativeAttentionMeta,
        *,
        layer_idx: int,
        query_states,
        key_states,
        value_states,
        attention_mask,
        past_key_values,
        wkvm_token_pool_decode,
        token_pool_attention_call=None,
        timing_enabled: bool,
    ):
        if token_pool_attention_call is None:
            token_pool_attention_call = self.resolve_attention_call(
                attn,
                layer_idx=layer_idx,
                query_states=query_states,
                attention_mask=attention_mask,
                wkvm_token_pool_decode=wkvm_token_pool_decode,
                timing_enabled=timing_enabled,
            )
        token_pool_kv_binding = token_pool_attention_call.bind_layer_kv(
            key_states,
            value_states,
            has_past_key_values=past_key_values is not None,
            is_kv_shared_layer=attn.is_kv_shared_layer,
        )
        token_pool_attention_call = token_pool_kv_binding.attention_call
        if token_pool_kv_binding.should_update_dense_cache:
            phase_start = time.perf_counter() if timing_enabled else 0.0
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                layer_idx,
            )
            if timing_enabled:
                _record_native_timing(
                    "self_attention_cache_update_wall_s",
                    time.perf_counter() - phase_start,
                )
        phase_start = time.perf_counter() if timing_enabled else 0.0
        attn_output, attn_weights = _attention_forward(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            backend=self.native_attention_backend,
            token_pool_attention_call=token_pool_attention_call,
        )
        if timing_enabled:
            _record_native_timing(
                "self_attention_attention_wall_s",
                time.perf_counter() - phase_start,
            )
        return attn_output, attn_weights, key_states, value_states


class NativeGemma4Attention:
    """Native Gemma4 attention math and token-pool backend dispatch."""

    def __init__(
        self,
        hf_attn,
        layer_idx: int,
        *,
        native_attention_backend: str,
        native_projection_backend: str,
        owned: bool,
        snapshot_device=None,
    ) -> None:
        self.layer_idx = int(layer_idx)
        self.native_attention_backend = _normalize_attention_backend(native_attention_backend)
        self.native_projection_backend = _normalize_projection_backend(
            native_projection_backend
        )
        self.attention_backend = NativeGemma4AttentionBackend(
            self.native_attention_backend
        )
        self.shared_kv_state = NativeGemma4SharedKVState()
        kv_shared_layer_index = getattr(hf_attn, "kv_shared_layer_index", None)
        if kv_shared_layer_index is None and bool(
            getattr(hf_attn, "is_kv_shared_layer", False)
        ):
            config = getattr(hf_attn, "config", None)
            if config is not None:
                kv_shared_layer_index = _gemma4_kv_shared_layer_index(
                    config,
                    self.layer_idx,
                )
        self.attn_meta = _NativeAttentionMeta(
            head_dim=int(hf_attn.head_dim),
            num_key_value_groups=int(hf_attn.num_key_value_groups),
            attention_dropout=float(hf_attn.attention_dropout),
            training=bool(hf_attn.training),
            scaling=float(hf_attn.scaling),
            is_kv_shared_layer=bool(getattr(hf_attn, "is_kv_shared_layer", False)),
            layer_type=getattr(hf_attn, "layer_type", None),
            kv_shared_layer_index=(
                None if kv_shared_layer_index is None else int(kv_shared_layer_index)
            ),
            store_full_length_kv=bool(
                getattr(hf_attn, "store_full_length_kv", False)
            ),
        )
        self.q_proj = _snapshot_linear(hf_attn.q_proj, owned=owned, device=snapshot_device)
        self.k_proj = _snapshot_linear(
            getattr(hf_attn, "k_proj", None),
            owned=owned,
            device=snapshot_device,
        )
        self.v_proj = _snapshot_linear(
            getattr(hf_attn, "v_proj", None),
            owned=owned,
            device=snapshot_device,
        )
        self.o_proj = _snapshot_linear(hf_attn.o_proj, owned=owned, device=snapshot_device)
        self.q_norm = _snapshot_norm(hf_attn.q_norm, owned=owned, device=snapshot_device)
        self.k_norm = _snapshot_norm(
            getattr(hf_attn, "k_norm", None),
            owned=owned,
            device=snapshot_device,
        )
        self.v_norm = _snapshot_norm(
            getattr(hf_attn, "v_norm", None),
            owned=owned,
            device=snapshot_device,
        )
        self._qkv_proj = None
        if (
            _packs_qkv(self.native_projection_backend)
            and not getattr(hf_attn, "is_kv_shared_layer", False)
            and getattr(hf_attn, "v_proj", None) is not None
        ):
            if owned:
                self._qkv_proj = _OwnedPackedLinear(
                    self.q_proj,
                    self.k_proj,
                    self.v_proj,
                    device=snapshot_device,
                )
                self.q_proj = None
                self.k_proj = None
                self.v_proj = None
            else:
                self._qkv_proj = _PackedLinear(
                    hf_attn.q_proj,
                    hf_attn.k_proj,
                    hf_attn.v_proj,
                )
        self.hf_attn = None if owned else hf_attn

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        hidden_states,
        *,
        position_embeddings,
        attention_mask,
        shared_kv_states,
        past_key_values,
        wkvm_token_pool_decode,
    ):
        attn = self.attn_meta
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        cos, sin = position_embeddings
        timing_enabled = _native_forward_timing_enabled()

        packed_qkv = self._qkv_proj is not None
        phase_start = time.perf_counter() if timing_enabled else 0.0
        if packed_qkv:
            query_raw, key_raw, value_raw = self._qkv_proj(hidden_states)
        else:
            query_raw = _linear(hidden_states, self.q_proj)
            key_raw = None
            value_raw = None
        if timing_enabled:
            _record_native_timing(
                "self_attention_qkv_proj_wall_s",
                time.perf_counter() - phase_start,
            )

        phase_start = time.perf_counter() if timing_enabled else 0.0
        query_states = query_raw.view(hidden_shape)
        query_states = _rms_norm(query_states, self.q_norm)
        query_states = _apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)
        if timing_enabled:
            _record_native_timing(
                "self_attention_q_norm_rope_wall_s",
                time.perf_counter() - phase_start,
            )

        token_pool_attention_call = self.attention_backend.resolve_attention_call(
            attn,
            layer_idx=self.layer_idx,
            query_states=query_states,
            attention_mask=attention_mask,
            wkvm_token_pool_decode=wkvm_token_pool_decode,
            timing_enabled=timing_enabled,
        )
        use_token_pool_shared_kv = bool(
            attn.is_kv_shared_layer
            and token_pool_attention_call.decode_attention_enabled
        )
        shared_kv = None
        if use_token_pool_shared_kv:
            key_states = None
            value_states = None
        else:
            shared_kv = self.shared_kv_state.load_shared_kv(
                attn,
                shared_kv_states,
                query_device=query_states.device,
                past_key_values=past_key_values,
                timing_enabled=timing_enabled,
            )

        if not use_token_pool_shared_kv and shared_kv is not None:
            key_states, value_states = shared_kv
        elif not use_token_pool_shared_kv:
            if key_raw is None:
                phase_start = time.perf_counter() if timing_enabled else 0.0
                key_raw = _linear(hidden_states, self.k_proj)
                if timing_enabled:
                    _record_native_timing(
                        "self_attention_qkv_proj_wall_s",
                        time.perf_counter() - phase_start,
                    )
            key_states = key_raw.view(hidden_shape)
            if value_raw is None and self.v_proj is None:
                value_states = key_states
            else:
                if value_raw is None:
                    phase_start = time.perf_counter() if timing_enabled else 0.0
                    value_raw = _linear(hidden_states, self.v_proj)
                    if timing_enabled:
                        _record_native_timing(
                            "self_attention_qkv_proj_wall_s",
                            time.perf_counter() - phase_start,
                        )
                value_states = value_raw.view(hidden_shape)

            phase_start = time.perf_counter() if timing_enabled else 0.0
            key_states = _rms_norm(key_states, self.k_norm)
            key_states = _apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = _rms_norm(value_states, self.v_norm)
            value_states = value_states.transpose(1, 2)
            if timing_enabled:
                _record_native_timing(
                    "self_attention_kv_norm_rope_wall_s",
                    time.perf_counter() - phase_start,
                )

        attn_output, attn_weights, key_states, value_states = self.attention_backend(
            attn,
            layer_idx=self.layer_idx,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            wkvm_token_pool_decode=wkvm_token_pool_decode,
            token_pool_attention_call=token_pool_attention_call,
            timing_enabled=timing_enabled,
        )
        self.shared_kv_state.store_shared_kv(
            attn,
            shared_kv_states,
            key_states,
            value_states,
            layer_idx=self.layer_idx,
            past_key_values=past_key_values,
        )
        phase_start = time.perf_counter() if timing_enabled else 0.0
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = _linear(attn_output, self.o_proj)
        if timing_enabled:
            _record_native_timing(
                "self_attention_output_proj_wall_s",
                time.perf_counter() - phase_start,
            )
        return attn_output, attn_weights

    def to(self, *args, **kwargs):
        for obj in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.q_norm,
            self.k_norm,
            self.v_norm,
            self._qkv_proj,
        ):
            to = getattr(obj, "to", None)
            if to is not None:
                to(*args, **kwargs)
        return self


class NativeGemma4TextDecoderLayer:
    """Explicit Gemma4 decoder-layer math backed by an already-loaded HF layer.

    The first production target is KV-owning sliding layers. The implementation is
    intentionally a parity bridge: it reads checkpoint weights from the HF layer,
    but does not call `Gemma4TextDecoderLayer.forward` or
    `Gemma4TextAttention.forward`.
    """

    def __init__(
        self,
        hf_layer,
        *,
        native_attention_backend: str = "manual",
        native_projection_backend: str = "separate",
        native_weight_backend: str = "hf_live",
    ) -> None:
        self.config = hf_layer.config
        self.layer_idx = int(hf_layer.layer_idx)
        self.hidden_size = int(hf_layer.hidden_size)
        self.hidden_size_per_layer_input = bool(hf_layer.hidden_size_per_layer_input)
        self.native_attention_backend = _normalize_attention_backend(native_attention_backend)
        self.native_projection_backend = _normalize_projection_backend(native_projection_backend)
        self.native_weight_backend = _normalize_weight_backend(native_weight_backend)
        self._owns_weight_tensors = self.native_weight_backend in {"owned", "owned_cpu"}
        self._weight_snapshot_device = (
            "cpu" if self.native_weight_backend == "owned_cpu" else None
        )
        self._gate_up_proj = None
        owned = self._owns_weight_tensors
        snapshot_device = self._weight_snapshot_device
        if getattr(hf_layer, "enable_moe_block", False):
            raise NotImplementedError("native Gemma4 layer does not support MoE blocks yet")
        attn = hf_layer.self_attn
        self.self_attn = NativeGemma4Attention(
            attn,
            self.layer_idx,
            native_attention_backend=self.native_attention_backend,
            native_projection_backend=self.native_projection_backend,
            owned=owned,
            snapshot_device=snapshot_device,
        )
        self.mlp_activation = hf_layer.mlp.config.hidden_activation
        self.input_layernorm = _snapshot_norm(
            hf_layer.input_layernorm,
            owned=owned,
            device=snapshot_device,
        )
        self.post_attention_layernorm = _snapshot_norm(
            hf_layer.post_attention_layernorm,
            owned=owned,
            device=snapshot_device,
        )
        self.pre_feedforward_layernorm = _snapshot_norm(
            hf_layer.pre_feedforward_layernorm,
            owned=owned,
            device=snapshot_device,
        )
        self.post_feedforward_layernorm = _snapshot_norm(
            hf_layer.post_feedforward_layernorm,
            owned=owned,
            device=snapshot_device,
        )
        self.mlp_gate_proj = _snapshot_linear(
            hf_layer.mlp.gate_proj,
            owned=owned,
            device=snapshot_device,
        )
        self.mlp_up_proj = _snapshot_linear(
            hf_layer.mlp.up_proj,
            owned=owned,
            device=snapshot_device,
        )
        self.mlp_down_proj = _snapshot_linear(
            hf_layer.mlp.down_proj,
            owned=owned,
            device=snapshot_device,
        )
        self.per_layer_input_gate = _snapshot_linear(
            getattr(hf_layer, "per_layer_input_gate", None),
            owned=owned,
            device=snapshot_device,
        )
        self.per_layer_projection = _snapshot_linear(
            getattr(hf_layer, "per_layer_projection", None),
            owned=owned,
            device=snapshot_device,
        )
        self.post_per_layer_input_norm = _snapshot_norm(
            getattr(hf_layer, "post_per_layer_input_norm", None),
            owned=owned,
            device=snapshot_device,
        )
        self.layer_scalar = (
            _clone_weight(hf_layer.layer_scalar, device=snapshot_device)
            if owned
            else hf_layer.layer_scalar
        )
        if _packs_gate_up(self.native_projection_backend):
            if owned:
                self._gate_up_proj = _OwnedPackedLinear(
                    self.mlp_gate_proj,
                    self.mlp_up_proj,
                    device=snapshot_device,
                )
                self.mlp_gate_proj = None
                self.mlp_up_proj = None
            else:
                self._gate_up_proj = _PackedLinear(
                    hf_layer.mlp.gate_proj,
                    hf_layer.mlp.up_proj,
                )
        self.hf_layer = None if owned else hf_layer

    @property
    def attn_meta(self) -> _NativeAttentionMeta:
        return self.self_attn.attn_meta

    @property
    def layer_type(self) -> str | None:
        return self.attn_meta.layer_type

    @property
    def q_proj(self):
        return self.self_attn.q_proj

    @property
    def k_proj(self):
        return self.self_attn.k_proj

    @property
    def v_proj(self):
        return self.self_attn.v_proj

    @property
    def o_proj(self):
        return self.self_attn.o_proj

    @property
    def q_norm(self):
        return self.self_attn.q_norm

    @property
    def k_norm(self):
        return self.self_attn.k_norm

    @property
    def v_norm(self):
        return self.self_attn.v_norm

    @property
    def _qkv_proj(self):
        return self.self_attn._qkv_proj

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        hidden_states,
        per_layer_input=None,
        *,
        shared_kv_states: dict[str, tuple[Any, Any]] | UserDict | None = None,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        wkvm_token_pool_decode=None,
        **_kwargs,
    ):
        if position_embeddings is None:
            raise ValueError("position_embeddings are required for native Gemma4 layer")
        shared_kv_states = shared_kv_states if shared_kv_states is not None else UserDict()

        timing_enabled = _native_forward_timing_enabled()
        layer_start = time.perf_counter() if timing_enabled else 0.0
        if timing_enabled:
            _record_native_count("layer_forward_calls")

        residual = hidden_states
        phase_start = time.perf_counter() if timing_enabled else 0.0
        hidden_states = _rms_norm(hidden_states, self.input_layernorm)
        if timing_enabled:
            _record_native_timing(
                "layer_input_norm_wall_s",
                time.perf_counter() - phase_start,
            )
            phase_start = time.perf_counter()
        hidden_states, _attn_weights = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            past_key_values=past_key_values,
            wkvm_token_pool_decode=wkvm_token_pool_decode,
        )
        if timing_enabled:
            _record_native_timing(
                "layer_self_attention_wall_s",
                time.perf_counter() - phase_start,
            )
            phase_start = time.perf_counter()
        hidden_states = _rms_norm(hidden_states, self.post_attention_layernorm)
        if timing_enabled:
            _record_native_timing(
                "layer_post_attention_norm_wall_s",
                time.perf_counter() - phase_start,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        phase_start = time.perf_counter() if timing_enabled else 0.0
        hidden_states = _rms_norm(hidden_states, self.pre_feedforward_layernorm)
        if timing_enabled:
            _record_native_timing(
                "layer_pre_feedforward_norm_wall_s",
                time.perf_counter() - phase_start,
            )
            phase_start = time.perf_counter()
        hidden_states = self._mlp(hidden_states)
        if timing_enabled:
            _record_native_timing("layer_mlp_wall_s", time.perf_counter() - phase_start)
            phase_start = time.perf_counter()
        if self.hidden_size_per_layer_input:
            hidden_states = _rms_norm(hidden_states, self.post_feedforward_layernorm)
            if timing_enabled:
                _record_native_timing(
                    "layer_post_feedforward_norm_wall_s",
                    time.perf_counter() - phase_start,
                )
            hidden_states = residual + hidden_states
            if per_layer_input is None:
                raise ValueError("per_layer_input is required for Gemma4 PLE layers")
            phase_start = time.perf_counter() if timing_enabled else 0.0
            residual = hidden_states
            hidden_states = _linear(hidden_states, self.per_layer_input_gate)
            hidden_states = _activation(self.config.hidden_activation, hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = _linear(hidden_states, self.per_layer_projection)
            hidden_states = _rms_norm_residual_scalar(
                hidden_states,
                self.post_per_layer_input_norm,
                residual,
                self.layer_scalar,
            )
            if timing_enabled:
                _record_native_timing("layer_ple_wall_s", time.perf_counter() - phase_start)
        else:
            hidden_states = _rms_norm_residual_scalar(
                hidden_states,
                self.post_feedforward_layernorm,
                residual,
                self.layer_scalar,
            )
            if timing_enabled:
                _record_native_timing(
                    "layer_post_feedforward_norm_wall_s",
                    time.perf_counter() - phase_start,
                )

        if timing_enabled:
            _record_native_timing("layer_forward_wall_s", time.perf_counter() - layer_start)
        return hidden_states

    def _self_attention(
        self,
        hidden_states,
        *,
        position_embeddings,
        attention_mask,
        shared_kv_states,
        past_key_values,
        wkvm_token_pool_decode,
    ):
        return self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            past_key_values=past_key_values,
            wkvm_token_pool_decode=wkvm_token_pool_decode,
        )

    def _mlp(self, x):
        timing_enabled = _native_forward_timing_enabled()
        phase_start = time.perf_counter() if timing_enabled else 0.0
        if self._gate_up_proj is not None:
            gate, up = self._gate_up_proj(x)
        else:
            gate = _linear(x, self.mlp_gate_proj)
            up = _linear(x, self.mlp_up_proj)
        if timing_enabled:
            _record_native_timing(
                "mlp_gate_up_proj_wall_s",
                time.perf_counter() - phase_start,
            )
            phase_start = time.perf_counter()
        output = _linear(_activation_mul(self.mlp_activation, gate, up), self.mlp_down_proj)
        if timing_enabled:
            _record_native_timing(
                "mlp_activation_down_proj_wall_s",
                time.perf_counter() - phase_start,
            )
        return output

    def to(self, *args, **kwargs):
        for obj in (
            self.self_attn,
            self.input_layernorm,
            self.post_attention_layernorm,
            self.pre_feedforward_layernorm,
            self.post_feedforward_layernorm,
            self.mlp_gate_proj,
            self.mlp_up_proj,
            self.mlp_down_proj,
            self.per_layer_input_gate,
            self.per_layer_projection,
            self.post_per_layer_input_norm,
            self._gate_up_proj,
        ):
            to = getattr(obj, "to", None)
            if to is not None:
                to(*args, **kwargs)
        if self._owns_weight_tensors and self.layer_scalar is not None:
            self.layer_scalar = self.layer_scalar.to(*args, **kwargs)
            if self._weight_snapshot_device is not None:
                self.layer_scalar = self.layer_scalar.to(device=self._weight_snapshot_device)
        return self


class NativeGemma4TextPrefix:
    """Run the first N Gemma4 text layers with explicit native layer math.

    This is an integration bridge, not the final production model runner. It
    keeps embeddings, PLE, rotary embeddings, and checkpoint weights from the
    already-loaded HF text model, while replacing the selected decoder layer
    calls with `NativeGemma4TextDecoderLayer`.
    """

    def __init__(
        self,
        hf_text_model,
        num_layers: int,
        *,
        native_attention_backend: str = "manual",
        native_projection_backend: str = "separate",
        native_weight_backend: str = "hf_live",
        release_hf_decoder_layers: bool = False,
    ) -> None:
        self.text_model = hf_text_model
        self.config = hf_text_model.config
        self.num_layers = int(num_layers)
        self.native_attention_backend = _normalize_attention_backend(native_attention_backend)
        self.native_projection_backend = _normalize_projection_backend(native_projection_backend)
        self.native_weight_backend = _normalize_weight_backend(native_weight_backend)
        self.release_hf_decoder_layers = bool(release_hf_decoder_layers)
        if self.release_hf_decoder_layers and self.native_weight_backend not in {
            "owned",
            "owned_cpu",
        }:
            raise ValueError(
                "release_hf_decoder_layers requires native_weight_backend='owned' "
                "or 'owned_cpu'"
            )
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.num_layers > int(self.config.num_hidden_layers):
            raise ValueError("num_layers exceeds Gemma4 text model depth")
        self.layers = []
        for layer_idx in range(self.num_layers):
            layer = hf_text_model.layers[layer_idx]
            self.layers.append(
                NativeGemma4TextDecoderLayer(
                    layer,
                    native_attention_backend=self.native_attention_backend,
                    native_projection_backend=self.native_projection_backend,
                    native_weight_backend=self.native_weight_backend,
                )
            )
            if self.release_hf_decoder_layers:
                import torch.nn as nn

                hf_text_model.layers[layer_idx] = nn.Identity()
                del layer
        self.released_hf_decoder_layers = (
            self.num_layers if self.release_hf_decoder_layers else 0
        )
        self.unique_layer_types = set(self.config.layer_types[: self.num_layers])
        self.kv_sharing_fast_prefill_split_layer = (
            self._kv_sharing_fast_prefill_split_layer()
        )
        self.kv_sharing_fast_prefill_eligible = (
            self.kv_sharing_fast_prefill_split_layer is not None
        )
        self.kv_sharing_fast_prefill_calls = 0
        self.kv_sharing_fast_prefill_owner_tokens = 0
        self.kv_sharing_fast_prefill_tail_tokens = 0
        self.kv_sharing_fast_prefill_fallbacks = 0
        self.kv_sharing_owner_only_calls = 0
        self.kv_sharing_owner_only_tokens = 0
        self.kv_sharing_owner_only_fallbacks = 0

    def _kv_sharing_fast_prefill_split_layer(self) -> int | None:
        total_layers = int(self.config.num_hidden_layers)
        shared_layers = int(getattr(self.config, "num_kv_shared_layers", 0) or 0)
        split_layer = total_layers - shared_layers
        layer_types = tuple(getattr(self.config, "layer_types", ()))
        if (
            self.num_layers != total_layers
            or split_layer < 1
            or split_layer >= total_layers
            or len(layer_types) != total_layers
        ):
            return None
        owner_layers = self.layers[:split_layer]
        tail_layers = self.layers[split_layer:]
        if any(layer.attn_meta.is_kv_shared_layer for layer in owner_layers):
            return None
        if len(tail_layers) != shared_layers or any(
            not layer.attn_meta.is_kv_shared_layer for layer in tail_layers
        ):
            return None
        for layer in tail_layers:
            source_idx = layer.attn_meta.kv_shared_layer_index
            if (
                source_idx is None
                or source_idx < 0
                or source_idx >= split_layer
                or layer_types[source_idx] != layer.layer_type
                or not owner_layers[source_idx].attn_meta.store_full_length_kv
            ):
                return None
        return split_layer

    def _should_use_kv_sharing_fast_prefill(
        self,
        *,
        hidden_states,
        position_embeddings,
        causal_mask_mapping,
        past_key_values,
        wkvm_token_pool_decode,
        wkvm_logits_indices,
    ) -> bool:
        batch_size, query_length = hidden_states.shape[:2]
        if (
            not self.kv_sharing_fast_prefill_eligible
            or int(query_length) < _KV_SHARING_FAST_PREFILL_MIN_TAIL_QUERY_LENGTH
            or wkvm_token_pool_decode is not None
            or wkvm_logits_indices is None
            or any(layer.attn_meta.training for layer in self.layers)
        ):
            return False
        query_lengths = getattr(past_key_values, "query_lengths", None)
        if query_lengths is not None and (
            len(query_lengths) != int(batch_size)
            or any(int(length) != int(query_length) for length in query_lengths)
        ):
            return False
        for layer_type in self.unique_layer_types:
            embeddings = position_embeddings.get(layer_type)
            if embeddings is None or len(embeddings) != 2:
                return False
            for tensor in embeddings:
                if (
                    tensor.ndim < 2
                    or int(tensor.shape[0]) not in {1, int(batch_size)}
                    or int(tensor.shape[1]) != int(query_length)
                ):
                    return False
            mask = causal_mask_mapping.get(layer_type)
            if mask is not None and (
                mask.ndim != 4
                or int(mask.shape[0]) not in {1, int(batch_size)}
                or int(mask.shape[-2]) != int(query_length)
            ):
                return False
        return True

    def _should_use_kv_sharing_owner_only(
        self,
        *,
        hidden_states,
        position_embeddings,
        causal_mask_mapping,
        past_key_values,
        wkvm_token_pool_decode,
    ) -> bool:
        batch_size, query_length = hidden_states.shape[:2]
        if (
            not self.kv_sharing_fast_prefill_eligible
            or int(query_length) <= 1
            or wkvm_token_pool_decode is not None
            or any(layer.attn_meta.training for layer in self.layers)
        ):
            return False
        query_lengths = getattr(past_key_values, "query_lengths", None)
        if query_lengths is not None and (
            len(query_lengths) != int(batch_size)
            or any(int(length) != int(query_length) for length in query_lengths)
        ):
            return False
        for layer_type in self.unique_layer_types:
            embeddings = position_embeddings.get(layer_type)
            if embeddings is None or len(embeddings) != 2:
                return False
            for tensor in embeddings:
                if (
                    tensor.ndim < 2
                    or int(tensor.shape[0]) not in {1, int(batch_size)}
                    or int(tensor.shape[1]) != int(query_length)
                ):
                    return False
            mask = causal_mask_mapping.get(layer_type)
            if mask is not None and (
                mask.ndim != 4
                or int(mask.shape[0]) not in {1, int(batch_size)}
                or int(mask.shape[-2]) != int(query_length)
            ):
                return False
        return True

    def __call__(self, *args, **kwargs) -> NativeGemma4PrefixOutput:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        *,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        per_layer_inputs=None,
        use_cache: bool | None = None,
        apply_final_norm: bool = False,
        wkvm_token_pool_decode=None,
        wkvm_kv_sharing_fast_prefill: bool = False,
        wkvm_kv_sharing_owner_only: bool = False,
        wkvm_logits_indices=None,
        **kwargs,
    ) -> NativeGemma4PrefixOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids is not None and per_layer_inputs is not None:
            raise ValueError("You cannot specify per_layer_inputs if input_ids is provided")

        timing_enabled = _native_forward_timing_enabled()
        if input_ids is not None:
            phase_start = time.perf_counter() if timing_enabled else 0.0
            inputs_embeds = self.text_model.embed_tokens(input_ids)
            if timing_enabled:
                _record_native_timing(
                    "text_embedding_wall_s",
                    time.perf_counter() - phase_start,
                )

        if self.text_model.hidden_size_per_layer_input:
            phase_start = time.perf_counter() if timing_enabled else 0.0
            if per_layer_inputs is None:
                per_layer_inputs = self.text_model.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.text_model.project_per_layer_inputs(
                inputs_embeds,
                per_layer_inputs,
            )
            if timing_enabled:
                _record_native_timing(
                    "text_per_layer_input_wall_s",
                    time.perf_counter() - phase_start,
                )

        if use_cache and past_key_values is None:
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            torch = _torch()
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            )
            position_ids = position_ids.unsqueeze(0)

        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            phase_start = time.perf_counter() if timing_enabled else 0.0
            from transformers.masking_utils import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )

            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
            if timing_enabled:
                _record_native_timing("text_mask_wall_s", time.perf_counter() - phase_start)

        hidden_states = inputs_embeds
        phase_start = time.perf_counter() if timing_enabled else 0.0
        position_embeddings = {
            layer_type: self.text_model.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in self.unique_layer_types
        }
        if timing_enabled:
            _record_native_timing("text_rotary_wall_s", time.perf_counter() - phase_start)
        shared_kv_states = kwargs.pop("shared_kv_states", UserDict())

        owner_only_requested = bool(
            wkvm_kv_sharing_fast_prefill and wkvm_kv_sharing_owner_only
        )
        owner_only = owner_only_requested and self._should_use_kv_sharing_owner_only(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            causal_mask_mapping=causal_mask_mapping,
            past_key_values=past_key_values,
            wkvm_token_pool_decode=wkvm_token_pool_decode,
        )
        if owner_only_requested and not owner_only:
            self.kv_sharing_owner_only_fallbacks += 1

        fast_prefill_indices = None
        if (
            wkvm_kv_sharing_fast_prefill
            and not owner_only_requested
            and wkvm_logits_indices is not None
        ):
            fast_prefill_indices = _normalize_wkvm_logits_indices(
                wkvm_logits_indices,
                batch_size=hidden_states.shape[0],
                query_length=hidden_states.shape[1],
                device=hidden_states.device,
            )
        fast_prefill = (
            bool(wkvm_kv_sharing_fast_prefill)
            and not owner_only_requested
            and self._should_use_kv_sharing_fast_prefill(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                causal_mask_mapping=causal_mask_mapping,
                past_key_values=past_key_values,
                wkvm_token_pool_decode=wkvm_token_pool_decode,
                wkvm_logits_indices=fast_prefill_indices,
            )
        )
        if (
            wkvm_kv_sharing_fast_prefill
            and not owner_only_requested
            and not fast_prefill
        ):
            self.kv_sharing_fast_prefill_fallbacks += 1
        split_layer = self.kv_sharing_fast_prefill_split_layer
        owner_hidden_states = None
        tail_position_embeddings = None
        tail_causal_mask_mapping = None
        tail_position_ids = None

        phase_start = time.perf_counter() if timing_enabled else 0.0
        for i, native_layer in enumerate(self.layers):
            if owner_only and i == split_layer:
                break
            if fast_prefill and i == split_layer:
                owner_hidden_states = hidden_states
                hidden_states = _gather_batch_query_tensor(
                    hidden_states,
                    fast_prefill_indices,
                )
                tail_position_embeddings = {
                    layer_type: tuple(
                        _gather_batch_query_tensor(tensor, fast_prefill_indices)
                        for tensor in embeddings
                    )
                    for layer_type, embeddings in position_embeddings.items()
                }
                tail_causal_mask_mapping = {
                    layer_type: _gather_attention_mask_query_rows(
                        mask,
                        fast_prefill_indices,
                        owner_hidden_states.shape[1],
                    )
                    for layer_type, mask in causal_mask_mapping.items()
                }
                tail_position_ids = _gather_batch_query_tensor(
                    position_ids,
                    fast_prefill_indices,
                )
            layer_type = self.config.layer_types[i]
            if per_layer_inputs is None:
                per_layer_input = None
            elif fast_prefill and i >= split_layer:
                per_layer_input = _gather_batch_query_tensor(
                    per_layer_inputs[:, :, i, :],
                    fast_prefill_indices,
                )
            else:
                per_layer_input = per_layer_inputs[:, :, i, :]
            layer_position_embeddings = position_embeddings
            layer_causal_mask_mapping = causal_mask_mapping
            layer_position_ids = position_ids
            if fast_prefill and i >= split_layer:
                layer_position_embeddings = tail_position_embeddings
                layer_causal_mask_mapping = tail_causal_mask_mapping
                layer_position_ids = tail_position_ids
            hidden_states = native_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=layer_position_embeddings[layer_type],
                attention_mask=layer_causal_mask_mapping[layer_type],
                position_ids=layer_position_ids,
                past_key_values=past_key_values,
                wkvm_token_pool_decode=wkvm_token_pool_decode,
                **kwargs,
            )
        if timing_enabled:
            _record_native_timing("text_layers_wall_s", time.perf_counter() - phase_start)

        if apply_final_norm and not owner_only:
            if self.num_layers != int(self.config.num_hidden_layers):
                raise ValueError("apply_final_norm requires running the full text stack")
            phase_start = time.perf_counter() if timing_enabled else 0.0
            hidden_states = self.text_model.norm(hidden_states)
            if timing_enabled:
                _record_native_timing(
                    "text_final_norm_wall_s",
                    time.perf_counter() - phase_start,
                )

        if fast_prefill:
            scatter_indices = fast_prefill_indices.reshape(-1, 1, 1).expand(
                -1,
                1,
                hidden_states.shape[-1],
            )
            hidden_states = owner_hidden_states.scatter(
                1,
                scatter_indices,
                hidden_states,
            )
            self.kv_sharing_fast_prefill_calls += 1
            self.kv_sharing_fast_prefill_owner_tokens += int(
                owner_hidden_states.shape[0] * owner_hidden_states.shape[1]
            )
            self.kv_sharing_fast_prefill_tail_tokens += int(
                owner_hidden_states.shape[0]
            )
        elif owner_only:
            self.kv_sharing_owner_only_calls += 1
            self.kv_sharing_owner_only_tokens += int(
                hidden_states.shape[0] * hidden_states.shape[1]
            )

        clear_shared_kv_store = getattr(
            past_key_values,
            "clear_shared_kv_store",
            None,
        )
        if callable(clear_shared_kv_store):
            clear_shared_kv_store()

        return NativeGemma4PrefixOutput(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            shared_kv_states=shared_kv_states,
            per_layer_inputs=per_layer_inputs,
            position_ids=position_ids,
            causal_mask_mapping=causal_mask_mapping,
            kv_sharing_owner_only=owner_only,
        )

    def to(self, *args, **kwargs):
        for layer in self.layers:
            layer.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        for layer in self.layers:
            layer.attn_meta.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)


class NativeGemma4ForCausalLM:
    """Causal-LM bridge using native Gemma4 text layers and loaded HF weights."""

    wkvm_forward_backend = "wkvm_native_gemma_forward_bridge"
    wkvm_no_hf_transformer_forward = True

    def __init__(
        self,
        hf_model,
        num_layers: int | None = None,
        *,
        native_attention_backend: str = "manual",
        native_projection_backend: str = "separate",
        native_weight_backend: str = "hf_live",
        release_hf_decoder_layers: bool = False,
    ) -> None:
        self.hf_model = hf_model
        self.config = hf_model.config
        self.num_layers = (
            int(self.config.num_hidden_layers) if num_layers is None else int(num_layers)
        )
        self.native_attention_backend = _normalize_attention_backend(native_attention_backend)
        self.native_projection_backend = _normalize_projection_backend(native_projection_backend)
        self.native_weight_backend = _normalize_weight_backend(native_weight_backend)
        self.release_hf_decoder_layers = bool(release_hf_decoder_layers)
        self.text_prefix = NativeGemma4TextPrefix(
            hf_model.model,
            self.num_layers,
            native_attention_backend=self.native_attention_backend,
            native_projection_backend=self.native_projection_backend,
            native_weight_backend=self.native_weight_backend,
            release_hf_decoder_layers=self.release_hf_decoder_layers,
        )
        self.released_hf_decoder_layers = self.text_prefix.released_hf_decoder_layers
        self.lm_head = hf_model.lm_head

    def __call__(self, *args, **kwargs) -> NativeGemma4CausalLMOutput:
        return self.forward(*args, **kwargs)

    @property
    def device(self):
        try:
            return self.hf_model.device
        except AttributeError:
            return next(self.parameters()).device

    @property
    def training(self) -> bool:
        return bool(getattr(self.hf_model, "training", False))

    def parameters(self, *args, **kwargs):
        return self.hf_model.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.hf_model.named_parameters(*args, **kwargs)

    def buffers(self, *args, **kwargs):
        return self.hf_model.buffers(*args, **kwargs)

    def eval(self):
        self.hf_model.eval()
        self.text_prefix.eval()
        return self

    def train(self, mode: bool = True):
        self.hf_model.train(mode)
        self.text_prefix.train(mode)
        return self

    def to(self, *args, **kwargs):
        self.hf_model.to(*args, **kwargs)
        self.text_prefix.to(*args, **kwargs)
        return self

    def forward(
        self,
        *,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        per_layer_inputs=None,
        use_cache: bool | None = None,
        logits_to_keep: int | Any = 0,
        **kwargs,
    ) -> NativeGemma4CausalLMOutput:
        # Ragged mixed batches and KV-sharing fast prefill both use one
        # per-row query index. The latter forwards it into the text stack so
        # the shared tail can execute only those query positions.
        wkvm_logits_indices = kwargs.pop("wkvm_logits_indices", None)
        wkvm_kv_sharing_fast_prefill = bool(
            kwargs.pop("wkvm_kv_sharing_fast_prefill", False)
        )
        wkvm_compute_logits = bool(kwargs.pop("wkvm_compute_logits", True))
        query_input = input_ids if input_ids is not None else inputs_embeds
        if wkvm_logits_indices is None and (
            wkvm_kv_sharing_fast_prefill
            and wkvm_compute_logits
            and isinstance(logits_to_keep, int)
            and logits_to_keep == 1
            and query_input is not None
        ):
            torch = _torch()
            wkvm_logits_indices = torch.full(
                (query_input.shape[0],),
                int(query_input.shape[1]) - 1,
                dtype=torch.long,
                device=query_input.device,
            )
        if wkvm_logits_indices is not None and query_input is not None:
            wkvm_logits_indices = _normalize_wkvm_logits_indices(
                wkvm_logits_indices,
                batch_size=query_input.shape[0],
                query_length=query_input.shape[1],
                device=query_input.device,
            )
        full_stack = self.num_layers == int(self.config.num_hidden_layers)
        text_out = self.text_prefix(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            use_cache=use_cache,
            apply_final_norm=full_stack,
            wkvm_kv_sharing_fast_prefill=wkvm_kv_sharing_fast_prefill,
            wkvm_kv_sharing_owner_only=(
                wkvm_kv_sharing_fast_prefill and not wkvm_compute_logits
            ),
            wkvm_logits_indices=wkvm_logits_indices,
            **kwargs,
        )
        hidden_states = text_out.hidden_states
        if bool(getattr(text_out, "kv_sharing_owner_only", False)):
            return NativeGemma4CausalLMOutput(
                logits=None,
                hidden_states=hidden_states,
                past_key_values=text_out.past_key_values,
                shared_kv_states=text_out.shared_kv_states,
            )
        if wkvm_logits_indices is not None:
            torch = _torch()
            indices = wkvm_logits_indices.to(device=hidden_states.device)
            row_indices = torch.arange(
                hidden_states.shape[0],
                dtype=torch.long,
                device=hidden_states.device,
            )
            hidden_states_for_logits = hidden_states[row_indices, indices].unsqueeze(1)
        elif isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None)
            hidden_states_for_logits = hidden_states[:, slice_indices, :]
        else:
            slice_indices = logits_to_keep
            hidden_states_for_logits = hidden_states[:, slice_indices, :]
        timing_enabled = _native_forward_timing_enabled()
        phase_start = time.perf_counter() if timing_enabled else 0.0
        logits = self.lm_head(hidden_states_for_logits)
        if timing_enabled:
            _record_native_timing("lm_head_wall_s", time.perf_counter() - phase_start)
        if self.config.final_logit_softcapping is not None:
            phase_start = time.perf_counter() if timing_enabled else 0.0
            logits = logits / self.config.final_logit_softcapping
            logits = logits.tanh()
            logits = logits * self.config.final_logit_softcapping
            if timing_enabled:
                _record_native_timing(
                    "lm_head_softcap_wall_s",
                    time.perf_counter() - phase_start,
                )
        return NativeGemma4CausalLMOutput(
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=text_out.past_key_values,
            shared_kv_states=text_out.shared_kv_states,
        )


def native_gemma4_from_checkpoint_state_dict(
    config,
    state_dict: dict[str, Any],
    *,
    prefix: str = "model.language_model",
    device=None,
    dtype=None,
    native_attention_backend: str = "manual",
    native_projection_backend: str = "separate",
) -> NativeGemma4ForCausalLM:
    """Build a native Gemma4 CausalLM facade from checkpoint tensors.

    This path does not instantiate `transformers.Gemma4ForCausalLM` or any HF
    decoder layers. `config` must be the Gemma4 text decoder config, not the
    outer multimodal config.
    """

    torch = _torch()
    if dtype is not None and isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    facade = _CheckpointGemma4ForCausalLMFacade(
        config,
        state_dict,
        prefix=prefix,
        device=device,
        dtype=dtype,
    )
    model = NativeGemma4ForCausalLM(
        facade,
        native_attention_backend=native_attention_backend,
        native_projection_backend=native_projection_backend,
        native_weight_backend="hf_live",
        release_hf_decoder_layers=False,
    )
    model.wkvm_checkpoint_native_loader = True
    model.wkvm_uses_hf_model_construction = False
    model.checkpoint_prefix = prefix
    return model.eval()


def load_native_gemma4_from_checkpoint(
    model_path: str | Path,
    *,
    device=None,
    dtype=None,
    prefix: str = "model.language_model",
    native_attention_backend: str = "manual",
    native_projection_backend: str = "separate",
) -> NativeGemma4ForCausalLM:
    """Load Gemma4 text weights from safetensors into the WKVM native bridge.

    This reads the local Gemma config.json directly, bypasses
    `Gemma4ForCausalLM.from_pretrained`, and does not build HF decoder-layer
    modules.
    """

    torch = _torch()
    if dtype is not None and isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    text_cfg = _load_native_gemma4_text_config(model_path)
    state_dict = _load_checkpoint_state_dict(
        model_path,
        prefix=prefix,
        device=device,
        dtype=dtype,
    )
    model = native_gemma4_from_checkpoint_state_dict(
        text_cfg,
        state_dict,
        prefix=prefix,
        device=device,
        dtype=dtype,
        native_attention_backend=native_attention_backend,
        native_projection_backend=native_projection_backend,
    )
    model.checkpoint_path = str(model_path)
    return model
