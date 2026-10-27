import math
from typing import Literal
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


SampleMods3 = Literal[
    "conv",
    "pixelshuffledirect",
    "pixelshuffle",
    "nearest+conv",
    "dysample",
    "transpose+conv",
    "lda",
    "pa_up"
]



def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict
    scale: int = 1

    _, upsampler_index, scale, embed_dim, num_out_ch, mid_dim, _ = state_dict[
        "upsampler.MetaUpsample"
    ].tolist()

    unshuffle_mod: bool = False
    if "conv_first.1.weight" in state_dict:
        num_in_ch = num_out_ch
        scale = 4 // math.isqrt(
            state_dict["conv_first.1.weight"].shape[1] // num_in_ch
        )
        unshuffle_mod = True

    elif "conv_first.weight" in state_dict:
        unshuffle_mod = False
        num_in_ch = state_dict["conv_first.weight"].shape[1]

    upsampler_type = list(SampleMods3.__args__)[upsampler_index]

    num_groups = get_nsequences(state_dict, "groups")
    num_blocks = get_nsequences(state_dict, "groups.0.blocks")


    def btype(i: int) -> str:
        b = f"groups.0.blocks.{i}.attn."
        if b + "dictionary" in state_dict:
            return "dictionary"
        if b + "temp" in state_dict:
            return "channel"
        return "spatial"

    types = [btype(i) for i in range(num_blocks)]
    period = next(
        p
        for p in range(1, num_blocks + 1)
        if num_blocks % p == 0 and all(types[i] == types[i % p] for i in range(num_blocks))
    )
    group_block_pattern = types[:period]
    depth_per_group = num_blocks // period
    num_heads = state_dict["groups.0.blocks.0.attn.rel_pos_bias"].shape[0]

    dict_idx = types.index("dictionary")
    num_dict_tokens = state_dict[f"groups.0.blocks.{dict_idx}.attn.dictionary"].shape[0]
    ffn_expansion_ratio = float(state_dict["groups.0.blocks.0.ffn.fc1.weight"].shape[0] / embed_dim)
    aim_reduction_ratio = (embed_dim // state_dict["groups.0.blocks.0.inter.cg.1.weight"].shape[0])

    # variant       embed_dim   heads   groups  M       blocks/group
    # fdat_rd_small     96      3       4       64      6
    # fdat_rd_medium    128     4       4       128     6
    # fdat_rd_large     192     6       6       256     6

    # upsampler_type
    embed_dim_to_suffix = {96: "small", 128: "medium", 192: "large"}
    variant = embed_dim_to_suffix.get(embed_dim, "")
    if variant:
        alt_arch_name = f"{model.arch_name} ({variant})"
    else:
        raise ValueError(red(f"[FDAT-RD] Unsupported embed dim {embed_dim}"))

    split_size: tuple[int, int] = (10, 30)

    # if is_debugging():
    #     from .module.fdat_rd import FDATRD
    #     model.update(ModuleClass=FDATRD)

    model.update(
        alt_arch_name=alt_arch_name,
        scale=scale,
        in_nc=num_in_ch,
        out_nc=num_out_ch,

        num_in_ch=num_in_ch,
        num_out_ch=num_out_ch,
        embed_dim=embed_dim,
        num_groups=num_groups,
        depth_per_group=depth_per_group,
        num_heads=num_heads,
        split_size=split_size,
        num_dict_tokens=num_dict_tokens,
        ffn_expansion_ratio=ffn_expansion_ratio,
        aim_reduction_ratio=aim_reduction_ratio,
        group_block_pattern=group_block_pattern,
        # drop_path_rate=0.1,
        mid_dim=mid_dim,
        upsampler_type=upsampler_type,
        # img_range=1.0,
        unshuffle_mod=unshuffle_mod,
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="FDAT-RD",
        detection_keys=(
            "groups.0.blocks.0.attn.rel_pos_bias",
            "groups.0.blocks.2.attn.dictionary",
            "groups.0.blocks.0.n1.weight",
            "upsampler.MetaUpsample",
        ),
        module=Module(file="fdat_rd", class_name="FDATRD"),
        parse=parse,
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
