from __future__ import annotations
from hytils import red
from nnarchs.import_libs import is_tensorrt_available
from nnarchs.model import OnnxModel
from nnarchs.nn_types import Hdtype, ShapeStrategy
if is_tensorrt_available():
    from nnarchs.nn_tensor_rt.trt_types import TensorrtModel
    from nnarchs.nn_tensor_rt.archs.onnx_to_trt import onnx_to_trt_engine
    from nnarchs.nn_tensor_rt.archs.onnx_to_trt_temporal import temporal_onnx_to_trt_engine
else:
    # print("[W] TensorRT is not supported: model cannot be converted")
    def onnx_to_trt_engine(*args):
        raise RuntimeError("TensorRT is not supported")


def to_tensorrt(
    model: OnnxModel,
    device: str,
    dtypes: set[Hdtype],
    shape_strategy: ShapeStrategy,
) -> TensorrtModel:

    if model.infer_type.type == 'temporal':
        return temporal_onnx_to_trt_engine(
            model=model,
            device=device,
            dtypes=dtypes,
            shape_strategy=shape_strategy
        )

    elif model.infer_type.type == 'simple':
        return onnx_to_trt_engine(
            model=model,
            device=device,
            dtypes=dtypes,
            shape_strategy=shape_strategy
        )

    raise NotImplementedError(red(f"Can't convert a `{model.infer_type.type}` model to a tensorRT engine"))
