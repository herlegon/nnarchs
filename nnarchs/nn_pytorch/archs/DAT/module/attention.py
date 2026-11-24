from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch.nn import functional as F

from .flash_window import flash_window_attention

_migraphx_export: ContextVar[bool] = ContextVar("migraphx_export", default=False)


def pad_heads(x: torch.Tensor) -> torch.Tensor:
    """Attention heads (the last dimension) zero-padded to a multiple of 8 channels.

    scaled_dot_product_attention's fused kernels (cuDNN, memory-efficient) only take
    such head widths and fall back to the unfused kernel otherwise; HAT's and DAT's
    heads are often 30 wide (180 channels in 6 heads). The padding is exact: the zeros
    add nothing to q @ k^T, and the attention output's extra channels are dropped.
    """
    pad = -x.shape[-1] % 8
    return F.pad(x, (0, pad)) if pad else x


@contextmanager
def migraphx_export() -> Iterator[None]:
    """ONNX exports within this block lay HAT's and DAT's attention out for MIGraphX.

    MIGraphX fuses attention into one kernel when the first product reaches the softmax
    through elementwise operations and reshapes only and the softmax feeds the second
    product directly. window_attention applies its mask in a form that keeps every
    window attention fusable (see there). DAT's channel attention, whose sequence is a
    head's channels (30) and whose head width is every pixel of the image, computes its
    second product transposed, which MIGraphX leaves as two matrix products: fused, it
    would run on one workgroup per head, and rocMLIR's first configuration for gfx942
    does not compile it.
    """
    token = _migraphx_export.set(True)
    try:
        yield
    finally:
        _migraphx_export.reset(token)


def in_migraphx_export() -> bool:
    """Whether an ONNX export runs within migraphx_export."""
    return _migraphx_export.get() and torch.onnx.is_in_onnx_export()


def window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None,
    mask: torch.Tensor | None,
    scale: float,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """softmax(q @ k^T * scale + bias + mask) @ v over windows.

    q is (B*nW, heads, N, d), k and v (B*nW, heads, M, d), d a multiple of 8
    (pad_heads); bias is (1, heads, N, M) and mask (nW, N, M), the same for every image.

    Runs use scaled_dot_product_attention (cuDNN or memory-efficient attention), which
    takes bias and mask summed into one attn_mask. Windows without a mask (plain ones,
    and the interior of shifted blocks) run in flash_window's kernel instead when it can
    (eager FP16 or BF16 inference on an NVIDIA GPU with Triton). An ONNX export keeps the operations
    apart: TensorRT fuses them into its own attention kernel and adds bias and mask
    there, where their sum, (B*nW, heads, N, M), is a tensor as large as the attention
    matrix to write and read in every shifted block.

    Within migraphx_export, the mask, tiled over the images, instead selects between the
    scores and the scores plus the mask's value: the same values for a mask whose nonzero
    entries share one value, as HAT's and DAT's (-100) do. MIGraphX, compiling for one
    image, turns the view into an unsqueeze and a squeeze that its attention fusion does
    not walk through; and added to the scores directly, mask and bias are two constant
    terms, which MIGraphX sums into one constant as large as the scores.
    """
    if torch.onnx.is_in_onnx_export():
        attn = (q * scale) @ k.transpose(-2, -1)
        if bias is not None:
            attn = attn + bias
        if mask is not None and in_migraphx_export():
            nw = mask.shape[0]
            masked = (mask != 0).unsqueeze(1).repeat(attn.shape[0] // nw, 1, 1, 1)
            attn = torch.where(masked, attn + mask.min(), attn)
        elif mask is not None:
            nw = mask.shape[0]
            attn = attn.view(-1, nw, *attn.shape[1:]) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, *attn.shape[2:])
        return attn.softmax(dim=-1) @ v
    if mask is None and dropout_p == 0.0:
        out = flash_window_attention(q, k, v, bias, scale)
        if out is not None:
            return out
    if mask is not None:
        nw = mask.shape[0]
        mask = mask.unsqueeze(1)
        bias = mask if bias is None else bias + mask
        bias = bias.unsqueeze(0).expand(q.shape[0] // nw, -1, -1, -1, -1).flatten(0, 1)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, dropout_p=dropout_p, scale=scale
    )
