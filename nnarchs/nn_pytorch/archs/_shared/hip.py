import importlib
import sys
from typing import Callable

import torch
from torch import nn



# The number of active TorchDispatchModes. PyTorch has no public API for it;
# looked up by name because strict type checking rejects torch._C's members.
_dispatch_mode_count: Callable[[], int] = importlib.import_module(
    "torch._C"
)._len_torch_dispatch_stack



def _tracing() -> bool:
    """Whether a tracer, torch.compile or torch.export is recording the model."""
    # torch.jit.is_tracing is documented API but not in torch.jit.__all__, so
    # strict type checking rejects the attribute; is_exporting is newer than 2.4
    jit_is_tracing = getattr(torch.jit, "is_tracing", None)
    if jit_is_tracing is not None and jit_is_tracing():
        return True
    # not imported here: an export in progress has imported it
    onnx = sys.modules.get("torch.onnx")
    if onnx is not None and onnx.is_in_onnx_export():
        return True
    is_exporting = getattr(torch.compiler, "is_exporting", None)
    return torch.compiler.is_compiling() or (
        is_exporting is not None and is_exporting()
    )



def plain_inference(x: torch.Tensor, module: nn.Module) -> bool:
    """
    Whether a custom kernel may run module's computation on x: plain eager
    inference only. Tracing (torch.jit, ONNX export), torch.compile and
    torch.export, tensor subclasses (fake or functional tensors), training,
    autograd and autocast keep the original code, which they can see through.

    So does an active TorchDispatchMode: a mode sees every operation and may
    act on tensors between them (e.g. move an idle tensor to host memory),
    while a kernel reads raw pointers it cannot see.
    """
    if _tracing() or type(x) is not torch.Tensor or module.training:
        return False
    if _dispatch_mode_count() > 0:
        return False
    if torch.is_autocast_enabled(x.device.type):
        return False
    if torch.is_grad_enabled():
        return not x.requires_grad and not any(
            p.requires_grad for p in module.parameters()
        )
    return True



def from_triton(error: BaseException) -> bool:
    """Whether the error, or one it was raised from, comes from Triton itself: a
    Triton kernel that fails to compile or launch switches itself off, while other
    errors (out of memory, wrong inputs) are raised."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__module__.partition(".")[0] == "triton":
            return True
        current = current.__cause__ or current.__context__
    return False



def is_depthwise_3x3(m: nn.Module) -> bool:
    """
    A plain nn.Conv2d (not a subclass, whose forward may differ) computing a
    depthwise 3x3 convolution with stride 1 and zero padding 1.
    """
    return (
        type(m) is nn.Conv2d
        and m.groups == m.in_channels == m.out_channels
        and m.kernel_size == (3, 3)
        and m.stride == (1, 1)
        and m.padding == (1, 1)
        and m.dilation == (1, 1)
        and m.padding_mode == "zeros"
    )
