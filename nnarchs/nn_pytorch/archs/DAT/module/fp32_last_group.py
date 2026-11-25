"""Running a residual trunk's last groups' sums in FP32 when the model runs in FP16.

Trained HAT and DAT models write far beyond FP16's 65504 in their last residual group's
closing convolution (722k in a trained 4x HAT-L, 121k in a 4x DAT2), right before the
final norm scales the values back down; in FP16 that is inf, and every pixel after it
NaN. The second-to-last group's closing convolution reaches 93,000 and the last group's
blocks 98,000 on trained 2x DAT2 files, on smooth tiles and natural crops alike. Wrapping
those convolutions, the last group's norms and the final norm keeps the sums between them
in FP32, as PyTorch promotes a sum of FP32 and FP16 to FP32; the attention and feed-forward
branches stay FP16. The weights stay FP16 and keep their state-dict keys; FP32 and BF16
models are unchanged.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F



def _to_float32(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return None if tensor is None else tensor.to(torch.float32)



class FP32Conv2d(nn.Conv2d):
    """A convolution computed in FP32 (FP32 output) when its weights are FP16."""

    def __init__(self, source: nn.Conv2d) -> None:
        nn.Module.__init__(self)
        # The same parameters, under the same state-dict keys
        self.__dict__.update(source.__dict__)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training or self.weight.dtype != torch.float16:
            return super().forward(input)

        return self._conv_forward(
            input.to(torch.float32),
            self.weight.to(torch.float32),
            _to_float32(self.bias)
        )



class FP32LayerNorm(nn.LayerNorm):
    """A LayerNorm computed in FP32 when its weights are FP16, reading the FP32 sum above
    it and handing the trunk back its weights' dtype."""

    def __init__(self, source: nn.LayerNorm) -> None:
        nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if (
            self.training
            or not self.elementwise_affine
            or self.weight.dtype != torch.float16
        ):
            return super().forward(input)
        x = F.layer_norm(
            input.to(torch.float32),
            self.normalized_shape,
            _to_float32(self.weight),
            _to_float32(self.bias),
            self.eps,
        )
        return x.to(self.weight.dtype)



def keep_last_groups_in_fp32_(
    layers: nn.ModuleList,
    blocks: tuple[type[nn.Module], ...]
) -> None:
    """The closing convolutions of the last two groups in `layers` computed in FP32 under
    FP16 weights, and the norms of the last group's `blocks`, which read the FP32 sums and
    hand their branches FP16."""
    for group in layers[-2:]:
        group.set_submodule("conv", keep_last_group_in_fp32(group.get_submodule("conv"))
                            )
    for module in layers[-1].modules():
        if isinstance(module, blocks):
            for name in ("norm1", "norm2"):
                norm = module.get_submodule(name)
                if isinstance(norm, nn.LayerNorm):
                    module.set_submodule(name, FP32LayerNorm(norm))



def keep_last_group_in_fp32(conv: nn.Module) -> nn.Module:
    """The last group's closing convolution ('1conv': a Conv2d; '3conv': a Sequential of
    them) with every Conv2d computed in FP32 under FP16 weights; anything else as is."""
    if isinstance(conv, nn.Conv2d):
        return FP32Conv2d(conv)

    if isinstance(conv, nn.Sequential):
        for index, layer in enumerate(conv):
            if isinstance(layer, nn.Conv2d):
                conv[index] = FP32Conv2d(layer)

    return conv
