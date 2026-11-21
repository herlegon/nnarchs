from typing import override

import torch
from torch import Tensor, nn
from .dual_arch import (
    PremixedDeployedDynamicConv3x3,
    TokenNorm,
    _premixed_conv3x3,
)



class _FP16SafeDynamicConv(PremixedDeployedDynamicConv3x3):
    """The last group's dynamic convolution, in FP32 when the model runs in FP16.

    Its output can far exceed FP16's 65504 (125k in a trained 1x Light) although
    the final norm right after it scales it back down. Only this convolution and
    the group's residual sum run in FP32; the persistent weights stay FP16.
    """

    def __init__(self, source: PremixedDeployedDynamicConv3x3) -> None:
        nn.Module.__init__(self)
        # Reuse loaded parameters under exactly the same state-dict keys.
        self.base = source.base
        self.experts = source.experts
        self.router = source.router

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.training or x.dtype != torch.float16:
            return super().forward(x)
        # The route in the caller's context: its autocast state decides the routing
        # form here as at every other site; only the convolution leaves autocast.
        coefficients = self.coefficients(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return _premixed_conv3x3(
                self,
                x.float(),
                self.base.weight.float(),
                self.base.bias.float(),
                self.experts.float(),
                coefficients,
                lambda: self.coefficients(x),
            )



class _FP16SafeTokenNorm(TokenNorm):
    """A norm reading the FP32 sum above; it hands the trunk back its own dtype."""

    def __init__(self, source: TokenNorm) -> None:
        nn.Module.__init__(self)
        self.norm = source.norm

    @override
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).to(self.norm.weight.dtype)
