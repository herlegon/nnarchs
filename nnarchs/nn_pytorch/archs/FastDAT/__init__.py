import math
from pprint import pprint
from typing import Literal
from nnarchs.architecture import (
    Module,
    NnPytorchArchitecture,
    OnnxConv,
    SizeConstraint,
    TensorRTConv,
)
from nnarchs.logger import is_debugging
from nnarchs.model import PyTorchModel
from nnarchs.nn_pytorch.archs import contains_all_keys
from ...torch_types import StateDict
from ..helpers import get_nsequences

# copy from module to avoid importing module right now
SampleMods = Literal[
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
]
SampleMods3 = Literal[SampleMods, "transpose+conv", "lda", "pa_up"]



def _detection_fct(state_dict: StateDict, keys: tuple[str | tuple[str]]) -> bool:
    # Differentiate between fdat and fdat-rd
    if (
        contains_all_keys(state_dict=state_dict, keys=keys)
        # and "groups.0.blocks.2.attn.dictionary" not in state_dict
        and (
            "conv_first.weight" in state_dict
            or "conv_first.1.weight" in state_dict
        )
    ):
        return True
    return False



def parse(model: PyTorchModel) -> None:
    in_nc: int
    state_dict: StateDict = model.state_dict
    scale: int = 2
    in_nc = out_nc = 3

    _, upsampler_index, scale, embed_dim, out_nc, mid_dim, _ = state_dict[
        "upsampler.MetaUpsample"
    ].tolist()
    upsampler_type = list(SampleMods3.__args__)[upsampler_index]

    if "conv_first.1.weight" in state_dict:
        in_nc = out_nc
        scale = 4 // (
            math.isqrt(state_dict["conv_first.1.weight"].shape[1] // in_nc)
        )
        unshuffle_mod = True
    else:
        unshuffle_mod = False
        in_nc = state_dict["conv_first.weight"].shape[1]

    num_groups = get_nsequences(state_dict, "groups")
    group_block_pattern = ["spatial", "channel"]
    depth_per_group: int = int(
        get_nsequences(state_dict, "groups.0.blocks")
        / len(group_block_pattern)
    )
    num_heads, _, window_size_sq = (
        state_dict["groups.0.blocks.0.attn.bias"].shape[:3]
    )
    window_size = math.isqrt(window_size_sq)
    ffn_expansion_ratio: float = (
        float(state_dict["groups.0.blocks.0.ffn.fc1.weight"].shape[0])
        / embed_dim
    )
    aim_reduction_ratio: int = int(
        embed_dim / state_dict["groups.0.blocks.0.inter.cg.1.weight"].shape[0]
    )
    img_range: float = 1.0

    variants: dict[int, tuple[str, float]] = {
        96: (" (tiny)", 0.5),
        108: (" (light)", 0.08),
        120: (" (medium)", 0.1),
        180: (" (large)", 0.1),
    }
    variant_name, drop_path_rate = variants.get(embed_dim, ('', 0.1))
    alt_arch_name = f"{model.arch.name}{variant_name}"
    if embed_dim == 180 and num_groups == 6:
        alt_arch_name = f"{model.arch.name} (xl)"
        drop_path_rate = 0.1

    if is_debugging():
        from .module.fdat import FDAT
        model.update(ModuleClass=FDAT)

    model.update(
        alt_arch_name=alt_arch_name,
        scale=scale,
        in_nc=in_nc,
        out_nc=out_nc,

        num_in_ch=in_nc,
        num_out_ch=out_nc,
        embed_dim=embed_dim,
        num_groups=num_groups,
        depth_per_group=depth_per_group,
        num_heads=num_heads,
        window_size=window_size,
        ffn_expansion_ratio=ffn_expansion_ratio,
        aim_reduction_ratio=aim_reduction_ratio,
        group_block_pattern=None, # ? group_block_pattern
        upsampler_type=upsampler_type,
        drop_path_rate=drop_path_rate,
        mid_dim=mid_dim,
        img_range=img_range,
        unshuffle_mod=unshuffle_mod,
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="FastDAT",
        detection_fct=_detection_fct,
        detection_keys=(
            "groups.0.blocks.0.attn.bias",
            "groups.0.blocks.0.inter.cg.1.weight",
            "groups.0.blocks.0.ffn.fc1.weight",
            "upsampler.MetaUpsample"
        ),
        module=Module(file="fdat", class_name="FDAT"),
        parse=parse,
        dtypes=('fp32', 'fp16', 'bf16'),
        size_constraint=SizeConstraint(min=(8, 8)),
        to_onnx = OnnxConv(
            dtypes=set(['fp32', 'fp16']),
            shape_strategy_types=set(['dynamic', 'static']),
        ),
        to_tensorrt=TensorRTConv(
            dtypes=set(['fp32', 'fp16']),
        ),
    ),
)



# def fdat_tiny(
#     num_in_ch: int = 3,
#     num_out_ch: int = 3,
#     scale: int = 4,
#     embed_dim: int = 96,
#     num_groups: int = 2,
#     depth_per_group: int = 2,
#     num_heads: int = 3,
#     window_size: int = 8,
#     ffn_expansion_ratio: float = 1.5,
#     aim_reduction_ratio: int = 8,
#     group_block_pattern: list[str] | None = None,
#     drop_path_rate: float = 0.05,
#     upsampler_type: SampleMods3 = "pixelshuffle",
#     img_range: float = 1.0,
# ) -> FDAT:


# def fdat_light(
#     num_in_ch: int = 3,
#     num_out_ch: int = 3,
#     scale: int = 4,
#     embed_dim: int = 108,
#     num_groups: int = 3,
#     depth_per_group: int = 2,
#     num_heads: int = 4,
#     window_size: int = 8,
#     ffn_expansion_ratio: float = 2.0,
#     aim_reduction_ratio: int = 8,
#     group_block_pattern: list[str] | None = None,
#     drop_path_rate: float = 0.08,
#     upsampler_type: SampleMods3 = "pixelshuffle",
#     img_range: float = 1.0,
# ) -> FDAT:


# def fdat_medium(
#     num_in_ch: int = 3,
#     num_out_ch: int = 3,
#     scale: int = 4,
#     embed_dim: int = 120,
#     num_groups: int = 4,
#     depth_per_group: int = 3,
#     num_heads: int = 4,
#     window_size: int = 8,
#     ffn_expansion_ratio: float = 2.0,
#     aim_reduction_ratio: int = 8,
#     group_block_pattern: list[str] | None = None,
#     drop_path_rate: float = 0.1,
#     upsampler_type: SampleMods3 = "transpose+conv",
#     img_range: float = 1.0,
# ) -> FDAT:


# def fdat_large(
#     num_in_ch: int = 3,
#     num_out_ch: int = 3,
#     scale: int = 4,
#     embed_dim: int = 180,
#     num_groups: int = 4,
#     depth_per_group: int = 4,
#     num_heads: int = 6,
#     window_size: int = 8,
#     ffn_expansion_ratio: float = 2.0,
#     aim_reduction_ratio: int = 8,
#     group_block_pattern: list[str] | None = None,
#     drop_path_rate: float = 0.1,
#     upsampler_type: SampleMods3 = "transpose+conv",
#     img_range: float = 1.0,
# ) -> FDAT:



# def fdat_xl(
#     num_in_ch: int = 3,
#     num_out_ch: int = 3,
#     scale: int = 4,
#     embed_dim: int = 180,
#     num_groups: int = 6,
#     depth_per_group: int = 6,
#     num_heads: int = 6,
#     window_size: int = 8,
#     ffn_expansion_ratio: float = 2.0,
#     aim_reduction_ratio: int = 8,
#     group_block_pattern: list[str] | None = None,
#     drop_path_rate: float = 0.1,
#     upsampler_type: SampleMods3 = "transpose+conv",
#     img_range: float = 1.0,
# ) -> FDAT:

