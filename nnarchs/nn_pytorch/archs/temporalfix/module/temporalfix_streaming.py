import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .temporalfix_arch import TemporalFix


class TemporalFixStreaming(nn.Module):
    """Single-graph version: encodes one new frame and runs the temporal fix."""

    onnx_input_names = ("x", "feats", "new_frame", "new_mask")
    onnx_output_names = ("output", "new_feat")

    # Same axis names everywhere, so the exporter/TensorRT treat H and W as shared
    # across all inputs and outputs. Batch stays 1, new_mask has no spatial axis.
    onnx_dynamic_axes = {
        "x":         {3: "height", 4: "width"},   # (1,7,3,H,W)
        "feats":     {3: "height", 4: "width"},   # (1,7,4,H,W)
        "new_frame": {2: "height", 3: "width"},   # (1,3,H,W)
        "output":    {2: "height", 3: "width"},   # (1,3,H,W)
        "new_feat":  {2: "height", 3: "width"},   # (1,4,H,W)
    }

    def __init__(self, core: TemporalFix) -> None:
        super().__init__()
        self.core = core

    def forward(self, x, feats, new_frame, new_mask):
        new_feat = self.core.encode(new_frame)
        feats = torch.where(new_mask > 0.5, new_feat.unsqueeze(1), feats)
        out = self.core(x, feats=feats)
        return out, new_feat

    def onnx_dummy_inputs(self, h, w, dtype, device):
        return (
            torch.rand(1, 7, 3, h, w, dtype=dtype, device=device),
            torch.rand(1, 7, 4, h, w, dtype=dtype, device=device),
            torch.rand(1, 3, h, w, dtype=dtype, device=device),
            torch.zeros(1, 7, 1, 1, 1, dtype=dtype, device=device),
        )
