"""Window attention as one Triton kernel for eager inference: softmax(q @ k^T * scale
+ bias) @ v per window and head, without a mask.

The windows of HAT, DAT and DRFT that need no mask (plain windows, the interior windows
of shifted blocks, DRFT's unmasked path, whose bias is already inside q and k) attend
here in one pass: an online FP32 softmax over tiles of keys, the scores and their
exponentials kept in registers, never written. The relative position bias, when given,
is the (heads, N, M) tensor the architectures already keep, one for every window (768 KiB
for HAT-L's 6 heads of 16 x 16 windows in FP16, so it stays in the GPU's cache): read
there, it took 1.46 ms per HAT-L 1024 input's windows against 1.91 ms for the same values
looked up from the compact (2 Wh - 1) (2 Ww - 1) table by position, on an RTX 5090.

PyTorch's flash backend, which takes no bias, is missing from its Windows builds; there
these windows ran on the memory-efficient kernel with the bias as a float mask.

Triton is optional: without it, AVAILABLE is False and the callers' PyTorch path runs.
"""
from __future__ import annotations
from nnarchs.logger import nnlogger

import torch
from torch import nn

from ..._shared.execution_checks import can_use_custom_kernel
from ..._shared.triton_utils import TRITON_AVAILABLE, from_triton, tl, triton


# A kernel that fails to compile or launch turns this off for the rest of the process.
FLASH_WINDOW = TRITON_AVAILABLE

# Head widths up to this run in the kernel; wider heads keep the PyTorch path.
MAX_HEAD_WIDTH = 128

if TRITON_AVAILABLE:

    @triton.jit
    def _flash_window(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        out_ptr,
        sq_w,
        sq_h,
        sq_n,
        sk_w,
        sk_h,
        sk_n,
        sv_w,
        sv_h,
        sv_n,
        heads: tl.int32,
        n_queries: tl.int32,
        n_keys: tl.int32,
        d_qk: tl.int32,
        d_v: tl.int32,
        scale: tl.float32,
        has_bias: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_dqk: tl.constexpr,
        block_dv: tl.constexpr,
    ):
        """One program: block_m queries of one window and head against all its keys.
        q (W, heads, N, d_qk), k (W, heads, M, d_qk), v (W, heads, M, d_v) with the
        last dimension contiguous; out (W, heads, N, d_v) and bias (heads, N, M)
        contiguous."""
        operand = out_ptr.dtype.element_ty
        blocks = tl.cdiv(n_queries, block_m)
        qb, wh = tl.program_id(0) % blocks, tl.program_id(0) // blocks
        window, head = wh // heads, wh % heads
        m = qb * block_m + tl.arange(0, block_m)
        dq = tl.arange(0, block_dqk)
        dv = tl.arange(0, block_dv)
        q = tl.load(
            q_ptr + window * sq_w + head * sq_h + m[:, None] * sq_n + dq[None, :],
            (m[:, None] < n_queries) & (dq[None, :] < d_qk),
            0,
        )
        # Scores in base 2: exp2 is the hardware's exponential.
        qk_scale = scale * 1.4426950408889634
        bias_rows = bias_ptr + (head * n_queries + m[:, None]) * n_keys
        maximum = tl.full((block_m,), -float("inf"), tl.float32)
        denominator = tl.full((block_m,), 0, tl.float32)
        acc = tl.full((block_m, block_dv), 0, tl.float32)
        k_base = k_ptr + window * sk_w + head * sk_h
        v_base = v_ptr + window * sv_w + head * sv_h
        for start in range(0, n_keys, block_n):
            n = start + tl.arange(0, block_n)
            inside = n < n_keys
            k = tl.load(
                k_base + n[None, :] * sk_n + dq[:, None],
                inside[None, :] & (dq[:, None] < d_qk),
                0,
            )
            v = tl.load(
                v_base + n[:, None] * sv_n + dv[None, :],
                inside[:, None] & (dv[None, :] < d_v),
                0,
            )
            score = tl.dot(q, k) * qk_scale
            if has_bias:
                bias = tl.load(
                    bias_rows + n[None, :],
                    (m[:, None] < n_queries) & inside[None, :],
                    0,
                )
                score += bias.to(tl.float32) * 1.4426950408889634
            score = tl.where(inside[None, :], score, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(score, 1))
            p = tl.exp2(score - new_max[:, None])
            alpha = tl.exp2(maximum - new_max)
            acc = tl.dot(p.to(operand), v, acc * alpha[:, None])
            denominator = denominator * alpha + tl.sum(p, 1)
            maximum = new_max
        tl.store(
            out_ptr + (wh * n_queries + m[:, None]) * d_v + dv[None, :],
            (acc / denominator[:, None]).to(operand),
            (m[:, None] < n_queries) & (dv[None, :] < d_v),
        )


def _block(width: int) -> int:
    """A head width's tile: the next power of two, at least 16 (tl.dot's minimum)."""
    return max(16, 1 << (width - 1).bit_length())


def launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """The kernel's output (W, heads, N, d_v) for q (W, heads, N, d_qk) and k, v
    (W, heads, M, d), any strides with the last dimension contiguous; bias (heads, N, M)
    contiguous, or None."""
    windows, heads, n_queries, d_qk = q.shape
    n_keys, d_v = k.shape[2], v.shape[3]
    out = q.new_empty((windows, heads, n_queries, d_v))
    # One axis, the query blocks of a window and head next to each other (their keys
    # stay in the cache): CUDA caps a grid's second and third axes at 65,535 programs,
    # which windows * heads passes above about 2.8 megapixels of 16 px windows with 6
    # heads; the first takes 2**31 - 1.
    grid = (triton.cdiv(n_queries, block_m) * windows * heads,)
    with torch.cuda.device(q.device):
        _flash_window[grid](
            q,
            k,
            v,
            q if bias is None else bias,
            out,
            *q.stride()[:3],
            *k.stride()[:3],
            *v.stride()[:3],
            heads,
            n_queries,
            n_keys,
            d_qk,
            d_v,
            scale,
            has_bias=bias is not None,
            block_m=block_m,
            block_n=block_n,
            block_dqk=_block(d_qk),
            block_dv=_block(d_v),
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


# can_use_custom_kernel's module: the attention has no weights of its own, and q requires a
# gradient whenever a weight before it does.
_WEIGHTLESS = nn.Module().eval()


def usable(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None
) -> bool:
    """Whether the kernel may run this attention: plain eager inference
    (can_use_custom_kernel: no tracing, export, compilation, autocast, dispatch mode or
    gradient) on an NVIDIA GPU, FP16 or BF16
    operands of one type, heads up to MAX_HEAD_WIDTH wide with the last dimension
    contiguous, and a bias, if any, one (heads, N, M) for all windows, contiguous."""
    heads, n_queries = q.shape[1], q.shape[2]
    return (
        FLASH_WINDOW
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == v.dtype == q.dtype
        and max(q.shape[-1], v.shape[-1]) <= MAX_HEAD_WIDTH
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        and (
            bias is None
            or (
                bias.dtype == q.dtype
                and bias.is_contiguous()
                and bias.shape[-3:] == (heads, n_queries, k.shape[2])
                and bias.numel() == heads * n_queries * k.shape[2]
            )
        )
        and can_use_custom_kernel(q, _WEIGHTLESS)
    )


def flash_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None,
    scale: float,
) -> torch.Tensor | None:
    """softmax(q @ k^T * scale + bias) @ v from the kernel, or None when the caller's
    PyTorch path must run: the kernel may not run (usable), or it failed in Triton,
    which switches it off for the rest of the process with one warning. Other errors
    are raised.

    q is (W, heads, N, d_qk), k and v (W, heads, M, d_qk / d_v); no head padding is
    needed. bias, (heads, N, M) or (1, heads, N, M), is the same for every window."""
    global FLASH_WINDOW
    if not usable(q, k, v, bias):
        return None
    try:
        return launch(q, k, v, bias, scale)
    except Exception as error:
        if not from_triton(error):
            raise
        FLASH_WINDOW = False
        nnlogger.warning(
            "The Triton window attention failed (%s: %s); using PyTorch's attention",
            type(error).__name__,
            error,
        )
        return None
