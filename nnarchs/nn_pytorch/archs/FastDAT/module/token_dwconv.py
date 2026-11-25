"""A depthwise 3x3 convolution (zero padding 1, stride 1, no bias) of tokens (B, H * W, C)
in token order, as one Triton kernel for eager inference, with an optional exact GELU on
its output.

On tokens, a convolution sees their channels-last view; PyTorch's depthwise convolution
copies that view to NCHW and its result is copied back to token order before the next
Linear, two passes the kernel does not make: FDAT's blocks at 720x480 on an RTX 5090
took 0.04 ms per convolution against 0.20 ms (120 channels, GELU included) and 0.035
against 0.50 ms (240 channels).

Triton is optional: without it, AVAILABLE is False and the callers' PyTorch path runs.
"""

from __future__ import annotations
from nnarchs.logger import nnlogger

import torch
from torch import nn

from ..._shared.convolution_utils import is_depthwise_3x3
from ..._shared.execution_checks import can_use_custom_kernel
from ..._shared.triton_utils import TRITON_AVAILABLE, from_triton, tl, triton

# A kernel that fails to compile or launch turns this off for the rest of the process.
TOKEN_DWCONV = TRITON_AVAILABLE

if TRITON_AVAILABLE:

    @triton.jit
    def _token_dwconv(
        x_ptr,
        w_ptr,
        out_ptr,
        height,
        width,
        channels,
        gelu: tl.constexpr,
        block_w: tl.constexpr,
        block_c: tl.constexpr,
    ):
        """One program: block_w tokens of one row (of all the images' rows) and block_c
        channels, in FP32."""
        row = tl.program_id(0)
        y = row % height
        xs = tl.program_id(1) * block_w + tl.arange(0, block_w)
        cs = tl.program_id(2) * block_c + tl.arange(0, block_c)
        c_ok = cs < channels
        acc = tl.zeros((block_w, block_c), tl.float32)
        for dy in tl.static_range(3):
            y_ok = (y + dy - 1 >= 0) & (y + dy - 1 < height)
            for dx in tl.static_range(3):
                xx = xs + dx - 1
                ok = y_ok & (xx >= 0) & (xx < width)
                token = (row + dy - 1).to(tl.int64) * width + xx
                v = tl.load(
                    x_ptr + token[:, None] * channels + cs[None, :],
                    ok[:, None] & c_ok[None, :],
                    0.0,
                ).to(tl.float32)
                w = tl.load(w_ptr + cs * 9 + dy * 3 + dx, c_ok, 0.0).to(tl.float32)
                acc += v * w[None, :]
        if gelu:
            acc = 0.5 * acc * (1 + tl.math.erf(acc * 0.7071067811865476))
        token = row.to(tl.int64) * width + xs
        tl.store(
            out_ptr + token[:, None] * channels + cs[None, :],
            acc.to(out_ptr.dtype.element_ty),
            (xs < width)[:, None] & c_ok[None, :],
        )


def usable(x: torch.Tensor, conv: nn.Module) -> bool:
    """Whether the kernel may run conv on the tokens x: plain eager inference
    (can_use_custom_kernel) on an NVIDIA GPU, FP16 or BF16 tokens contiguous, and conv a plain
    depthwise 3x3 convolution without bias over x's channels, of x's type."""
    return (
        TOKEN_DWCONV
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.dim() == 3
        and x.is_contiguous()
        and is_depthwise_3x3(conv)
        and conv.in_channels == x.shape[-1]
        and conv.bias is None
        and conv.weight.dtype == x.dtype
        and conv.weight.is_contiguous()
        and can_use_custom_kernel(x, conv)
    )


def token_dwconv(
    x: torch.Tensor, conv: nn.Conv2d, height: int, width: int, gelu: bool = False
) -> torch.Tensor | None:
    """conv (then GELU when gelu) of the tokens x (B, height * width, C) as tokens, or
    None when the caller's PyTorch path must run: the kernel may not run (usable), or it
    failed in Triton, which switches it off for the rest of the process with one
    warning. Other errors are raised."""
    global TOKEN_DWCONV
    if not usable(x, conv):
        return None
    b, _, channels = x.shape
    out = torch.empty_like(x)
    # every channel of a token in one tile (up to 128), a short run of tokens
    block_w, block_c = 16, min(128, triton.next_power_of_2(channels))
    grid = (b * height, triton.cdiv(width, block_w), triton.cdiv(channels, block_c))
    try:
        with torch.cuda.device(x.device):
            _token_dwconv[grid](
                x,
                conv.weight,
                out,
                height,
                width,
                channels,
                gelu=gelu,
                block_w=block_w,
                block_c=block_c,
            )
    except Exception as error:
        if not from_triton(error):
            raise
        TOKEN_DWCONV = False
        nnlogger.warning(
            "The Triton depthwise convolution failed (%s: %s); using PyTorch's",
            type(error).__name__,
            error,
        )
        return None
    return out
