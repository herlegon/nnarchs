from nnarchs.architecture import (
    Module,
    NnPytorchArchitecture,
    OnnxConv,
    SizeConstraint,
    TensorRTConv,
    InferType,
)
from nnarchs.logger import is_debugging
from nnarchs.model import PyTorchModel
from nnarchs.nn_pytorch.archs import contains_all_keys
from ...torch_types import StateDict



def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict
    in_nc = 3

    model.update(
        arch_name=model.arch.name,
        scale=1,
        in_nc=in_nc,
        out_nc=in_nc,

        # dim=dim,
        # config=config,
        # drop_path_rate=0.,
        # input_resolution=256
    )



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="temporalfix",
        detection_keys=(
            "encode.cnn0.weight",
            "aligner.blocks.0.conv0.0.0.weight",
            "conf_head.net.0.0.weight",
        ),
        module=Module(file="temporalfix", class_name="TemporalFix"),
        parse=parse,
        dtypes=('fp32', 'fp16', 'bf16'),
        size_constraint=SizeConstraint(min=(8, 8), modulo=2),
        to_onnx = OnnxConv(
            dtypes=set(['fp32', 'fp16', 'bf16']),
            shape_strategy_types=set(['dynamic', 'static']),
        ),
        to_tensorrt=TensorRTConv(
            dtypes=set(['fp32', 'fp16', 'bf16']),
        ),
        infer_type=InferType(
            type='temporal',
            inputs=7,
            outputs=1,
        )
    ),
)
