import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def __init__(
        self,
        in_channels: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
    ):
        super().__init__()

        if in_channels < groups or in_channels % groups != 0:
            raise ValueError("Incorrect in_channels and groups values.")

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> torch.Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )


    def _normalizer(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """[W, H] of the offset map, shaped to broadcast against the coordinates."""
        return torch.tensor(
            [offset.shape[-1], offset.shape[-2]],
            dtype=x.dtype,
            device=x.device
        ).view(1, 2, 1, 1, 1)


    def forward(self, x: torch.Tensor)-> torch.Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = self._normalizer(x, offset)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output



class ExportableDySample(DySample):
    """DySample that exports to ONNX with dynamic input sizes.

    Differences from DySample:

    - The [W, H] normalizer comes from torch._shape_as_tensor, so a trace
      records the sizes as part of the graph instead of as constants. The
      exported model then accepts any input size, not only the export one.
    - The end convolution has a configurable kernel (end_kernel), with
      padding end_kernel // 2 so the spatial size is preserved.
    - in_channels and out_ch have defaults (64 and 3).

    Parameters and buffers have the same names as in DySample, so checkpoints
    load into either class.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_ch: int = 3,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
        end_kernel: int = 1,
    ) -> None:
        super().__init__(in_channels, out_ch, scale, groups, end_convolution)
        if end_convolution and end_kernel != 1:
            self.end_conv = nn.Conv2d(
                in_channels, out_ch, end_kernel, 1, end_kernel // 2
            )

    def _normalizer(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return torch._shape_as_tensor(offset)[[4, 3]].to(x).view(1, 2, 1, 1, 1)

