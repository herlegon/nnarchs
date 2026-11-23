"""Swin-style window attention as one Triton kernel for eager inference that reads q, k
and v where an image-wide qkv projection wrote them and writes each token's output in
token order: no pass partitions, rolls, gathers or merges the windows.

Shifted windows are the windows of the image rolled by (-shift, -shift): a window's row
P reads the image's row (P + shift) mod H, and so for columns. Swin's mask, -100 between
tokens of different regions of the rolled image (each axis split at -window and -shift),
comes from index comparisons in the kernel, so the windows that wrap around take it and
the others, whose tokens share one region, are unchanged by it. The relative position
bias is the (heads, N, N) tensor the architectures already keep, read in the GPU's cache.
An online FP32 softmax over tiles of keys, as flash_window's.

Triton is optional: without it, AVAILABLE is False and the callers' PyTorch path runs.
"""

from __future__ import annotations

import torch
from torch import nn
from .helpers import from_triton, plain_inference
from nnarchs.logger import nnlogger

try:
    import triton
    import triton.language as tl
except ImportError:
    AVAILABLE = False
else:
    AVAILABLE = True


# A kernel that fails to compile or launch turns this off for the rest of the process.
INDEX_WINDOW = AVAILABLE

# Head widths up to this run in the kernel.
MAX_HEAD_WIDTH = 128

# The query and key tile of every launch: a window's tokens must be a multiple of it.
TILE = 64

if AVAILABLE:

    @triton.jit
    def _region(p, n, window, shift):
        """Swin's region of the rolled image's row or column p: 0, 1 past -window,
        2 past -shift."""
        return (p >= n - window).to(tl.int32) + (p >= n - shift).to(tl.int32)

    @triton.jit
    def _index_window(
        qkv_ptr,
        bias_ptr,
        out_ptr,
        height: tl.int32,
        width: tl.int32,
        windows_w: tl.int32,
        windows_per_image: tl.int32,
        heads: tl.int32,
        shift: tl.int32,
        head_stride: tl.int32,
        d_qk: tl.int32,
        d_v: tl.int32,
        scale: tl.float32,
        window: tl.constexpr,
        masked: tl.constexpr,
        linear: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        """One program: block_m queries of one window and head against the window's
        window * window keys. qkv (B, H, W, 3, heads, head_stride) contiguous (each head
        d_qk wide, padded to head_stride); bias (heads, N, N) contiguous; out (B, H, W,
        heads, d_v) contiguous."""
        operand = out_ptr.dtype.element_ty
        if linear:
            blocks = tl.cdiv(window * window, block_m)
            qb, wh = tl.program_id(0) % blocks, tl.program_id(0) // blocks
        else:
            qb, wh = tl.program_id(0), tl.program_id(1)
        win, head = wh // heads, wh % heads
        image = win // windows_per_image
        wy = (win % windows_per_image) // windows_w
        wx = win % windows_w
        n_tokens = window * window
        row_stride = 3 * heads * head_stride

        m = qb * block_m + tl.arange(0, block_m)
        py = wy * window + m // window
        px = wx * window + m % window
        q_token = (
            (image * height + (py + shift) % height) * width + (px + shift) % width
        ).to(tl.int64)
        d = tl.arange(0, block_d)
        q = tl.load(
            qkv_ptr + q_token[:, None] * row_stride + head * head_stride + d[None, :],
            d[None, :] < d_qk,
            0,
        )
        q_region = _region(py, height, window, shift) * 3 + _region(
            px, width, window, shift
        )
        qk_scale = scale * 1.4426950408889634
        bias_rows = bias_ptr + (head * n_tokens + m[:, None]) * n_tokens
        maximum = tl.full((block_m,), -float("inf"), tl.float32)
        denominator = tl.full((block_m,), 0, tl.float32)
        acc = tl.full((block_m, block_d), 0, tl.float32)
        for start in range(0, n_tokens, block_n):
            n = start + tl.arange(0, block_n)
            ky = wy * window + n // window
            kx = wx * window + n % window
            k_token = (
                (image * height + (ky + shift) % height) * width + (kx + shift) % width
            ).to(tl.int64)
            k = tl.load(
                qkv_ptr
                + k_token[None, :] * row_stride
                + heads * head_stride
                + head * head_stride
                + d[:, None],
                d[:, None] < d_qk,
                0,
            )
            v = tl.load(
                qkv_ptr
                + k_token[:, None] * row_stride
                + 2 * heads * head_stride
                + head * head_stride
                + d[None, :],
                d[None, :] < d_v,
                0,
            )
            bias = tl.load(bias_rows + n[None, :]).to(tl.float32)
            if masked:
                k_region = _region(ky, height, window, shift) * 3 + _region(
                    kx, width, window, shift
                )
                bias = tl.where(
                    q_region[:, None] == k_region[None, :], bias, bias - 100.0
                )
            score = tl.dot(q, k) * qk_scale + bias * 1.4426950408889634
            new_max = tl.maximum(maximum, tl.max(score, 1))
            p = tl.exp2(score - new_max[:, None])
            alpha = tl.exp2(maximum - new_max)
            acc = tl.dot(p.to(operand), v, acc * alpha[:, None])
            denominator = denominator * alpha + tl.sum(p, 1)
            maximum = new_max
        tl.store(
            out_ptr + (q_token[:, None] * heads + head) * d_v + d[None, :],
            (acc / denominator[:, None]).to(operand),
            d[None, :] < d_v,
        )


def _block(width: int) -> int:
    """A head width's tile: the next power of two, at least 16 (tl.dot's minimum)."""
    return max(16, 1 << (width - 1).bit_length())


def launch(
    qkv: torch.Tensor,
    bias: torch.Tensor,
    window: int,
    shift: int,
    width: int,
    scale: float,
    masked: bool = True,
    block_m: int = TILE,
    block_n: int = TILE,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """The output (B, H, W, heads * width) for qkv (B, H, W, 3, heads, head_stride)
    contiguous, H and W multiples of the window, and bias (heads, N, N) contiguous.
    Shifted windows take Swin's mask unless masked is False (TFDATv2's windows wrap
    around unmasked)."""
    if (window * window) % block_m or (window * window) % block_n:
        # the kernel does not mask past a window's end
        raise ValueError(
            f"{window} x {window} windows are not whole {block_m}- and {block_n}-token tiles"
        )
    b, h, w, _, heads, head_stride = qkv.shape
    windows_h, windows_w = h // window, w // window
    out = qkv.new_empty((b, h, w, heads * width))
    # CUDA caps a grid's second axis at 65,535 programs, which windows * heads passes
    # above about 2.8 megapixels of 16 px windows with 6 heads: then one axis
    # (2**31 - 1), the query blocks of a window and head next to each other
    blocks = triton.cdiv(window * window, block_m)
    windows_heads = b * windows_h * windows_w * heads
    linear = windows_heads > 65535
    grid = (blocks * windows_heads,) if linear else (blocks, windows_heads)
    with torch.cuda.device(qkv.device):
        _index_window[grid](
            qkv,
            bias,
            out,
            h,
            w,
            windows_w,
            windows_h * windows_w,
            heads,
            shift,
            head_stride,
            width,
            width,
            scale,
            window=window,
            masked=masked and shift > 0,
            linear=linear,
            block_m=block_m,
            block_n=block_n,
            block_d=_block(head_stride),
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


_WEIGHTLESS = nn.Module().eval()


def usable(qkv: torch.Tensor, bias: torch.Tensor, window: int) -> bool:
    """Whether the kernel may run: plain eager inference on an NVIDIA GPU, FP16 or BF16
    qkv and bias of one type, both contiguous, the bias one (heads, N, N) for all
    windows, heads up to MAX_HEAD_WIDTH wide, and windows of a whole number of the
    kernel's 64-token tiles (16 x 16 and 8 x 8, not 12 x 12): it does not mask tokens
    past a window's end."""
    heads, n_tokens = qkv.shape[4], window * window
    return (
        INDEX_WINDOW
        and n_tokens % TILE == 0
        and qkv.is_cuda
        and torch.version.hip is None
        and qkv.dtype in (torch.float16, torch.bfloat16)
        and qkv.dim() == 6
        and qkv.is_contiguous()
        and qkv.shape[-1] <= MAX_HEAD_WIDTH
        and bias.dtype == qkv.dtype
        and bias.is_contiguous()
        and bias.numel() == heads * n_tokens * n_tokens
        and plain_inference(qkv, _WEIGHTLESS)
    )


def index_window_attention(
    qkv: torch.Tensor,
    bias: torch.Tensor,
    window: int,
    shift: int,
    width: int,
    scale: float,
    masked: bool = True,
) -> torch.Tensor | None:
    """softmax(q @ k^T * scale + bias + Swin's mask) @ v over the (shifted) windows of
    an image's qkv (B, H, W, 3, heads, head_stride), as (B, H, W, heads * width), the
    mask left out when masked is False; or
    None when the caller's PyTorch path must run: the kernel may not run (usable), or it
    failed in Triton, which switches it off for the rest of the process with one
    warning. Other errors are raised."""
    global INDEX_WINDOW
    if not usable(qkv, bias, window):
        return None
    try:
        return launch(qkv, bias, window, shift, width, scale, masked)
    except Exception as error:
        if not from_triton(error):
            raise
        INDEX_WINDOW = False
        nnlogger.warning(
            "The Triton index window attention failed (%s: %s); using the window copies",
            type(error).__name__,
            error,
        )
        return None
