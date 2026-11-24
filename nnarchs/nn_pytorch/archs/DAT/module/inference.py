"""Inference paths with fewer passes over memory, shared by DAT and HAT.

A model that takes them works channels last: a convolution reads and writes tokens where they
are. What depends only on the weights is kept, and a shifted window that does not wrap around
is a plain window at an offset, with no roll and no mask. Traced graphs keep channels first and
no caches, and so do models on ROCm, whose depthwise kernels take channels first."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .attention import window_attention
from .kernels import hip_kernels_available


def traced() -> bool:
    """Whether this call keeps channels first and no caches: a traced graph does, and ROCm's
    depthwise kernels take nothing else."""
    return (
        torch.jit.is_tracing()
        or torch.onnx.is_in_onnx_export()
        or hip_kernels_available()
    )


class Derived(nn.Module):
    """A module with tensors derived from its weights, kept for inference: moving or casting
    the model, or loading weights, drops them, and training and tracing never see them."""

    def __init__(self):
        super().__init__()
        self.derived = {}

    def _apply(self, fn, recurse=True):
        self.derived.clear()
        return super()._apply(fn, recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.derived.clear()
        super()._load_from_state_dict(*args, **kwargs)

    def keep(self, key, compute):
        if self.training or torch.is_grad_enabled():
            return compute()
        kept = self.derived.get(key)
        if kept is None:
            kept = self.derived[key] = compute()
        return kept

    def run_conv(self, key, conv, x, norm=None):
        """conv on channels-last x, then its batch norm, folded into it in inference, with
        the weight in x's layout, which cuDNN would otherwise copy for every call."""
        if self.training or torch.is_grad_enabled():
            x = conv(x)
            return x if norm is None else norm(x)
        weight, bias = self.keep(key, lambda: channels_last_conv(conv, norm))
        return F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups)


def add_into(x, y):
    """x + y in y's memory, y being a result nothing else reads; of two types, the wider's."""
    return y.add_(x) if x.dtype == y.dtype and not torch.is_grad_enabled() else x + y


def padded_rows(linear, heads, width, padded):
    """linear's weight and bias with zero rows after each head's width: its output heads are
    padded with exact zeros, without a pass over the output."""

    def pad(rows):
        shape = (3, heads, width, *rows.shape[1:])
        extra = (0, 0) * (rows.dim() - 1) + (0, padded - width)
        return F.pad(rows.reshape(shape), extra).reshape(-1, *rows.shape[1:])

    return pad(linear.weight), None if linear.bias is None else pad(linear.bias)


def channels_last_conv(conv, norm):
    weight, bias = conv.weight, conv.bias
    if norm is not None:
        dtype = weight.dtype
        # at least FP32, as batch norm computes
        wide = torch.promote_types(dtype, torch.float32)
        scale = norm.weight.to(wide) / (norm.running_var.to(wide) + norm.eps).sqrt()
        offset = norm.bias.to(wide) - norm.running_mean.to(wide) * scale
        bias = offset if bias is None else bias.to(wide) * scale + offset
        weight = (weight.to(wide) * scale.view(-1, 1, 1, 1)).to(dtype)
        bias = bias.to(dtype)
    return weight.to(memory_format=torch.channels_last), bias


def img2windows(img, H_sp, W_sp, num_heads):
    """
    Input: Images (S, B, H, W, C), channels last
    Output: Window Partition (S, B', num_heads, N, C // num_heads)
    """
    S, B, H, W, C = img.shape
    img_reshape = img.reshape(
        S, B, H // H_sp, H_sp, W // W_sp, W_sp, num_heads, C // num_heads
    )
    img_perm = img_reshape.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(
        S, -1, num_heads, H_sp * W_sp, C // num_heads
    )
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', num_heads, N, C // num_heads)
    Output: Image (B, H, W, C)
    """
    _, num_heads, _, head_dim = img_splits_hw.shape
    img = img_splits_hw.view(-1, H // H_sp, W // W_sp, num_heads, H_sp, W_sp, head_dim)
    img = img.permute(0, 1, 4, 2, 5, 3, 6).reshape(-1, H, W, num_heads * head_dim)
    return img


def _window_regions(n, split, shift, device):
    index = torch.arange(n, device=device)
    past_split = (index >= n - split).to(torch.int32)
    return past_split + (index >= n - shift).to(torch.int32)


def shifted_window_mask(H, W, split_h, split_w, shift_h, shift_w, dtype, device):
    """Swin's attention mask for shifted windows (nW, N, N): 0 between tokens of the same
    region, -100 across. Each axis has three regions, split at -split and -shift; the
    regions come from index comparisons, so an exported graph needs no scatter and the
    mask is built on the model's device."""
    img_mask = (
        _window_regions(H, split_h, shift_h, device)[:, None] * 3
        + _window_regions(W, split_w, shift_w, device)
    )
    mask_windows = (
        img_mask.view(H // split_h, split_h, W // split_w, split_w)
        .permute(0, 2, 1, 3)
        .reshape(-1, split_h * split_w)
    )
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return torch.zeros(attn_mask.shape, dtype=dtype, device=device).masked_fill(
        attn_mask != 0, -100.0
    )


def border_window_mask(H, W, split_h, split_w, shift_h, shift_w, dtype, device):
    """shifted_window_mask for the only windows it marks, those that wrap around: the right
    column above the last row, then the last row. The others are plain windows."""
    img_mask = (
        _window_regions(H, split_h, shift_h, device)[:, None] * 3
        + _window_regions(W, split_w, shift_w, device)
    )
    inner_h, inner_w = H - split_h, W - split_w
    right = img_mask[:inner_h, inner_w:].view(inner_h // split_h, split_h, 1, split_w)
    bottom = img_mask[inner_h:].view(1, split_h, W // split_w, split_w)
    mask_windows = torch.cat(
        [
            right.permute(0, 2, 1, 3).reshape(-1, split_h * split_w),
            bottom.permute(0, 2, 1, 3).reshape(-1, split_h * split_w),
        ]
    )
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return torch.zeros(attn_mask.shape, dtype=dtype, device=device).masked_fill(
        attn_mask != 0, -100.0
    )


def shifted_window_attention(
    qkv, H, W, H_sp, W_sp, shift_h, shift_w, num_heads, bias, border_mask, scale, dropout=0.0, width=None
):
    """The shifted windows of qkv (3, B, H, W, C), not rolled: the same result as rolling
    by (-shift_h, -shift_w), attending with the full mask and rolling back. A window that
    does not wrap around is a plain window at an offset, its mask all zero, so only the
    right column and last row of windows take the mask and the wrapped tokens. width keeps
    that many of each head's output channels. Output: (B, H, W, C)"""
    S, B, _, _, _ = qkv.shape
    inner_h, inner_w = H - H_sp, W - W_sp
    rows = (torch.arange(H, device=qkv.device) + shift_h) % H
    cols = (torch.arange(W, device=qkv.device) + shift_w) % W
    rows_inner, rows_last = rows[:inner_h, None], rows[inner_h:, None]
    cols_last = cols[inner_w:]

    windows = img2windows(
        qkv[:, :, shift_h : shift_h + inner_h, shift_w : shift_w + inner_w],
        H_sp,
        W_sp,
        num_heads,
    )
    inner = window_attention(windows[0], windows[1], windows[2], bias, None, scale, dropout)
    inner = windows2img(inner[..., :width], H_sp, W_sp, inner_h, inner_w)
    # B, the right column's windows, then the last row's, for the mask's order
    right = img2windows(qkv[:, :, rows_inner, cols_last], H_sp, W_sp, num_heads)
    bottom = img2windows(qkv[:, :, rows_last, cols], H_sp, W_sp, num_heads)
    n_right = right.shape[1] // B
    windows = torch.cat(
        [right.view(S, B, n_right, *right.shape[2:]), bottom.view(S, B, -1, *bottom.shape[2:])],
        dim=2,
    ).flatten(1, 2)
    border = window_attention(
        windows[0], windows[1], windows[2], bias, border_mask, scale, dropout
    )[..., :width]
    border = border.view(B, -1, *border.shape[1:])

    x = qkv.new_empty((B, H, W, inner.shape[-1]))
    x[:, shift_h : shift_h + inner_h, shift_w : shift_w + inner_w] = inner
    x[:, rows_inner, cols_last] = windows2img(
        border[:, :n_right].flatten(0, 1), H_sp, W_sp, inner_h, W_sp
    )
    x[:, rows_last, cols] = windows2img(border[:, n_right:].flatten(0, 1), H_sp, W_sp, H_sp, W)
    return x


def tail_scale_and_halo(tail: nn.Module) -> tuple[int, int] | None:
    """(scale, halo) of a high-resolution tail that banded_tail can run: its leaves are
    stride-1 convolutions padded to their kernel's half (each needs that many neighbouring
    rows at its own resolution), nearest upsamplings and pixel shuffles by whole factors
    (rows map to factor rows), and pointwise activations. The halo is in rows of the tail's
    input: ceil(padding / scale so far) per convolution. None for any other leaf."""
    scale, halo = 1, 0
    for m in tail.modules():
        if next(m.children(), None) is not None:
            continue
        if isinstance(m, nn.Conv2d):
            kh = m.kernel_size[0]
            if (
                m.stride != (1, 1)
                or m.dilation != (1, 1)
                or not isinstance(m.padding, tuple)
                or m.padding[0] != kh // 2
                or kh % 2 == 0
                or m.padding_mode != "zeros"
            ):
                return None
            halo += -(-m.padding[0] // scale)
        elif isinstance(m, nn.Upsample):
            factor = m.scale_factor
            if m.mode != "nearest" or not isinstance(factor, (int, float)) or factor != int(factor):
                return None
            scale *= int(factor)
        elif isinstance(m, nn.PixelShuffle):
            scale *= m.upscale_factor
        elif not isinstance(m, (nn.LeakyReLU, nn.ReLU, nn.PReLU, nn.SiLU, nn.GELU, nn.Identity)):
            return None
    return scale, halo


def banded_tail(
    tail: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    scale: int,
    halo: int,
    rows: int,
) -> torch.Tensor:
    """tail(x) for a tail that upscales x (B, C, H, W) by `scale` with `halo` rows of
    context per side (tail_scale_and_halo), computed on bands of `rows` rows of x and written
    into one output: the same values as the whole pass (each band sees every row its output
    rows depend on; at the image's edges the convolutions pad as they do there), while only
    one band's high-resolution activations exist at a time. One band is the whole pass."""
    h = x.shape[2]
    if h <= rows:
        return tail(x)
    out = None
    for start in range(0, h, rows):
        end = min(h, start + rows)
        lo, hi = max(0, start - halo), min(h, end + halo)
        y = tail(x[:, :, lo:hi])
        if out is None:
            layout = (
                torch.channels_last
                if y.dim() == 4 and y.is_contiguous(memory_format=torch.channels_last)
                else torch.contiguous_format
            )
            out = torch.empty(
                (*y.shape[:2], h * scale, y.shape[3]),
                dtype=y.dtype,
                device=y.device,
                memory_format=layout,
            )
        out[:, :, start * scale : end * scale] = y[
            :, :, (start - lo) * scale : (end - lo) * scale
        ]
        del y
    assert out is not None
    return out
