import importlib
from typing import Callable

import torch
from torch import nn


# The number of active TorchDispatchModes. PyTorch has no public API for it;
# looked up by name because strict type checking rejects torch._C's members.
_active_dispatch_mode_count: Callable[[], int] = (
    importlib.import_module("torch._C")._len_torch_dispatch_stack
)

def is_graph_export() -> bool:
    """Whether a frozen graph is being exported: keep channels first and no caches."""
    return torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()



def is_graph_capture() -> bool:
    """Whether any graph capture is running: export, or torch.compile / torch.export."""
    return is_graph_export() or torch.compiler.is_compiling()



def can_use_custom_kernel(x: torch.Tensor, module: nn.Module) -> bool:
    """
    Whether a custom kernel may run module's computation on x: plain eager
    inference only. Tracing (torch.jit, ONNX export), torch.compile and
    torch.export, tensor subclasses (fake or functional tensors), training,
    autograd and autocast keep the original code, which they can see through.

    So does an active TorchDispatchMode: a mode sees every operation and may
    act on tensors between them (e.g. move an idle tensor to host memory),
    while a kernel reads raw pointers it cannot see.
    """
    if (
        (is_graph_capture() or type(x) is not torch.Tensor or module.training)
        or _active_dispatch_mode_count() > 0
        or torch.is_autocast_enabled(x.device.type)
    ):
        return False

    if torch.is_grad_enabled():
        return (
            not x.requires_grad
            and not any(p.requires_grad for p in module.parameters())
        )

    return True
