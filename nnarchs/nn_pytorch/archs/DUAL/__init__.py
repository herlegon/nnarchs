from itertools import combinations
import math
from typing import Literal, override
from hytils import red
from nnarchs.architecture import (
    Module,
    NnPytorchArchitecture,
    OnnxConv,
    SizeConstraint,
    TensorRTConv,
)
from nnarchs.logger import is_debugging
from nnarchs.model import PyTorchModel
from ...torch_types import StateDict
from ..helpers import get_nsequences
import torch
from torch import Tensor
from torch import nn



# Original layer indices of a full group: 2 and 5 are the context layers.
_FULL = (0, 1, 2, 3, 4, 5)
_CONTEXT = (2, 5)
# The schedule of the dual_light and dual_xs presets.
_LIGHT_INDICES = (
    (0, 1, 2, 3),
    (3, 4, 5),
    (0, 1, 2),
    (0, 1, 2, 3, 4, 5)
)
# (embed_dim, depths, pixel_unshuffle) -> preset name
_PRESETS = {
    (128, (4, 3, 3, 6), True): "Light",
    (128, (4, 3, 3, 6), False): "XS",
    (96, (6,) * 4, False): "DUAL3X",
    (160, (6,) * 6, False): "S",
    (192, (6,) * 6, False): "M",
    (224, (6,) * 10, False): "L",
    (256, (6,) * 12, False): "XL",
}


def _layer_indices(
    state_dict: StateDict, depths: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """The original index of every kept layer. Only the position of the context
    layers is in the weights: a pruned schedule they do not determine must be
    the Light/XS preset's."""
    options: list[list[tuple[int, ...]]] = []
    for group, depth in enumerate(depths):
        context = tuple(
            f"groups.{group}.layers.{i}.dual.theta" in state_dict for i in range(depth)
        )
        options.append(
            [
                indices
                for indices in combinations(_FULL, depth)
                if tuple(i in _CONTEXT for i in indices) == context
                and (group > 0 or indices[:2] == (0, 1))
            ]
        )
    if len(depths) == len(_LIGHT_INDICES) and all(
        preset in group for preset, group in zip(_LIGHT_INDICES, options)
    ):
        return _LIGHT_INDICES
    if all(len(group) == 1 for group in options):
        return tuple(group[0] for group in options)
    raise ValueError(
        "Unsupported DUAL layer schedule: the kept layers of a pruned group are"
        " not determined by the weights"
    )



def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 2
    num_out_ch = num_in_ch = 3

    # https://github.com/Rift73/spandrel/blob/chaiNNer-C/libs/spandrel/spandrel/architectures/DUAL/__init__.py

    # Folded (exported) checkpoints hold one plain 3x3 conv per EDBB conv.
    folded = "stem.weight" in state_dict
    if folded:
        in_channels = state_dict["stem.weight"].shape[1]
        embed_dim = state_dict["stem.weight"].shape[0]
    else:
        in_channels = state_dict["stem.direct.expand.weight"].shape[1]
        embed_dim = state_dict["stem.direct.reduce.weight"].shape[0]
    pixel_unshuffle = in_channels == 12

    # tail: first conv, one conv per x2 stage (+1 for PU2), last conv
    tail_convs = (get_nsequences(state_dict, "tail") + 1) // 2
    scale = 2 ** (tail_convs - 2 - int(pixel_unshuffle))

    depths = tuple(
        get_nsequences(state_dict, f"groups.{group}.layers")
        for group in range(get_nsequences(state_dict, "groups"))
    )
    indices = _layer_indices(state_dict, depths)

    table = state_dict["groups.0.layers.0.attention.relative_bias"].shape[0]
    g_window = (math.isqrt(table) + 1) // 2

    c_pos_bias = "rfb"
    rfb_rank = 16
    context = next(
        (
            key[: -len("dual.theta")]
            for key in state_dict
            if key.endswith(".dual.theta")
        ),
        None,
    )
    if context is not None:
        bias = context + "attention.factored_bias."
        if bias + "frequency" in state_dict:
            rfb_rank = state_dict[bias + "frequency"].shape[0] * 2
        elif bias + "to_query" in state_dict:
            c_pos_bias = "rib"
            rfb_rank = state_dict[bias + "to_query"].shape[2]
        else:
            c_pos_bias = "none"

    qk = (
        "cosine"
        if "groups.0.layers.0.attention.logit_scale" in state_dict
        else "dot"
    )
    dynamic_experts = state_dict["groups.0.conv.experts"].shape[0]

    variant = _PRESETS.get((embed_dim, depths, pixel_unshuffle))
    if pixel_unshuffle:
        variant = f"{variant}, pixel unshuffle"

    model.update(
        alt_arch_name=f"{model.arch.name} ({variant})",
        scale=scale,
        in_nc=num_in_ch,
        out_nc=num_out_ch,
        folded=folded,

        embed_dim=embed_dim,
        depths=depths,
        # the factories leave a full schedule implicit
        layer_indices=None if all(i == _FULL for i in indices) else indices,
        pixel_unshuffle=pixel_unshuffle,
        dynamic_experts=dynamic_experts,
        # c_windows cannot be deduced from the state dict (default 32/64)
        # Routing is not in the weights either: neighborhood attention
        # wherever it is defined (RFB bias and dot attention), legacy
        # non-overlapping windows otherwise.
        context_na=c_pos_bias == "rfb" and qk == "dot",
        # the local layers' neighborhoods need dot attention (their dense bias is
        # every DUAL's); the retrieval's need nothing
        local_na=qk == "dot",
        retrieval_na=True,
        # per-position routing and regional attention where their kernels run
        # (eager inference on NVIDIA GPUs); today's forms elsewhere
        routing_na=True,
        regional_na=True,
        g_window=g_window,
        c_pos_bias=c_pos_bias,
        rfb_rank=rfb_rank,
        qk=qk,
    )



def load_state(model: PyTorchModel) -> None:
    from .module.deploy import _FP16SafeDynamicConv, _FP16SafeTokenNorm
    from .module.dual_arch import PremixedDeployedDynamicConv3x3

    module: nn.Module = model.module

    if model.folded:
        module.eval()
        module.prepare_for_export()

    module.load_state_dict(model.state_dict, strict=True)

    # Fold the static EDBB bases for inference after the strict load above;
    # the routers and expert kernels stay input-dependent. Fold on CPU in
    # FP32 (the factors in FP64) before ModelLoader moves this module to the
    # GPU.
    module.float().eval()
    if not model.folded:
        module.prepare_for_export()

    last = module.groups[-1]
    assert isinstance(last.conv, PremixedDeployedDynamicConv3x3)
    last.conv = _FP16SafeDynamicConv(last.conv)
    module.final_norm = _FP16SafeTokenNorm(module.final_norm)
    if len(module.groups) == 2:
        # The cross-scale retrieval reads group 1's output too.
        module.xr.norm_u = _FP16SafeTokenNorm(module.xr.norm_u)
    module.requires_grad_(False).eval()



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="DUAL",
        detection_keys=(
                "mean",
                "groups.0.layers.0.attention.relative_bias",
                "groups.0.layers.0.attention.qkv.weight",
                "groups.0.layers.0.ffn.depthwise.weight",
                "groups.0.conv.experts",
                "groups.0.conv.router.fc1.weight",
                "groups.1.conv.experts",
                "final_norm.norm.weight",
                "xr.log_tau",
                "xr.query.weight",
                "xr.key.weight",
        ),
        module=Module(file="dual_arch", class_name="DUAL"),
        parse=parse,
        load_state=load_state,
        dtypes=('fp32', 'bf16', 'fp16'),
        size_constraint=SizeConstraint(min=(8, 8)),
        to_onnx = OnnxConv(
            dtypes=set(['fp32', 'bf16', 'fp16']),
            shape_strategy_types=set(['dynamic', 'static']),
        ),
        to_tensorrt=TensorRTConv(
            dtypes=set(['fp32', 'bf16', 'fp16']),
        ),
    ),
)


