"""Small fused CUDA primitives for native Gemma decode."""

from __future__ import annotations


def _triton_modules():
    import triton
    import triton.language as tl

    return triton, tl


triton, tl = _triton_modules()


@triton.jit
def _rms_norm_residual_scalar_kernel(
    x,
    weight,
    residual,
    scalar,
    output,
    row_stride: tl.constexpr,
    width: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < width
    x_values = tl.load(
        x + row * row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight_values = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    residual_values = tl.load(
        residual + row * row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    reciprocal_rms = tl.rsqrt(tl.sum(x_values * x_values, axis=0) / width + eps)
    normalized = (x_values * reciprocal_rms * weight_values).to(
        x.dtype.element_ty
    ).to(tl.float32)
    combined = (normalized + residual_values).to(x.dtype.element_ty).to(tl.float32)
    scalar_value = tl.load(scalar).to(tl.float32)
    tl.store(
        output + row * row_stride + offsets,
        combined * scalar_value,
        mask=mask,
    )


def rms_norm_residual_scalar(x, weight, residual, scalar, eps: float):
    """Compute ``(rms_norm(x, weight) + residual) * scalar`` in one launch."""

    width = int(x.shape[-1])
    if tuple(x.shape) != tuple(residual.shape):
        raise ValueError("x and residual must have the same shape")
    if int(weight.numel()) != width:
        raise ValueError("weight length must match the hidden width")
    if int(scalar.numel()) != 1:
        raise ValueError("scalar must contain exactly one element")
    if not all(value.is_contiguous() for value in (x, weight, residual, scalar)):
        raise ValueError("fused RMSNorm inputs must be contiguous")
    if width < 1 or width > 8192:
        raise ValueError("hidden width must be in [1, 8192]")
    rows = int(x.numel() // width)
    block_size = 1 << (width - 1).bit_length()
    output = x.new_empty(x.shape)
    _rms_norm_residual_scalar_kernel[(rows,)](
        x,
        weight,
        residual,
        scalar,
        output,
        row_stride=width,
        width=width,
        eps=float(eps),
        block_size=block_size,
        num_warps=4,
    )
    return output
