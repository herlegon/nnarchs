from functools import partial
from hytils import (
    yellow,
)
import onnx
from typing import Literal
from nnarchs.logger import nnlogger
from nnarchs.architecture import NnOnnxArchitecture, TensorRTConv
from nnarchs.model import OnnxModel
from ..inference.session import OnnxSession
from .parser import (
    parse_onnx_model,
    parse_temporal_onnx_model,
)


def is_model_bgd_removal(model: onnx.ModelProto, item: Literal['node', 'output'], keys: tuple[str]) -> bool:
    if len(model.graph.input) != 1:
        # raise ValueError("ONNX model: input length shall be equal to 1")
        return False

    if item == 'node':
        parsed = tuple([gn for gn in model.graph.node[-2].input])
    elif item == 'output':
        parsed = tuple([go.name for go in model.graph.output])
    return True if parsed == keys else False


def is_model_rife(model: onnx.ModelProto) -> bool:
    # Detect ONNX rife models
    # highly experimental because not enough unique keys/nodes
    graph = model.graph
    if len(graph.input) != 1:
        # raise ValueError("ONNX model: input length shall be equal to 1")
        return False

    shape = graph.input[0].type.tensor_type.shape
    input_shape = [dim.dim_value for dim in shape.dim]

    shape = graph.output[0].type.tensor_type.shape
    output_shape = [dim.dim_value for dim in shape.dim]

    try:
        if (graph.node[1].input[0] == graph.input[0].name
            and graph.node[1].op_type == 'Split'
            and input_shape[1] == 11

            and graph.node[-1].output[0] == graph.output[0].name
            and output_shape[1] == 3
            and graph.node[-1].op_type == 'Add'
        ):
            nnlogger.debug(yellow("rife"))
            return True
    except:
        pass
    return False


def is_model_generic(model: onnx.ModelProto) -> bool:
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        nnlogger.debug("[D] Not a SISR model")
        return False

    return True


def is_model_temporal(model: onnx.ModelProto) -> bool:
    graph = model.graph
    if len(graph.input) < 2:
        nnlogger.debug("[D] Temporal model")
        return False

    return True


def create_session(model: OnnxModel) -> OnnxSession:
    nnlogger.debug(yellow("load onnx model"))
    return OnnxSession(model)



MODEL_ARCHITECTURES: tuple[NnOnnxArchitecture] = (
    NnOnnxArchitecture(
        name='Rife',
        category='Frame interpolation',
        detection_fct=is_model_rife,
    ),
    NnOnnxArchitecture(
        name='unknown',
        detection_fct=is_model_generic,
        # Allow conversion to tensorrt
        to_tensorrt=TensorRTConv(
            dtypes=('fp32', 'fp16', 'bf16'),
            weak_typing=False,
            shape_strategy_types=('dynamic', 'fixed', 'static'),
        ),
    ),
    NnOnnxArchitecture(
        name='unknown',
        detection_fct=is_model_temporal,
        parse=parse_temporal_onnx_model,
        # Allow conversion to tensorrt
        to_tensorrt=TensorRTConv(
            dtypes=('fp32', 'fp16', 'bf16'),
            weak_typing=False,
            shape_strategy_types=('dynamic', 'fixed', 'static'),
        ),
    )
)

# Append common variables for all architectures
for arch in MODEL_ARCHITECTURES:
    arch.update(dict(
        type=arch.name,
        parse=partial(parse_onnx_model, scale=arch.scale) if arch.parse is None else arch.parse,
        create_session=create_session,
        dtypes=['fp32', 'fp16', 'bf16']
    ))
