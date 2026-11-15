from __future__ import annotations
from typing import TYPE_CHECKING
from hytils import red
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
from ...torch_types import StateDict
import onnx
import torch

from io import BytesIO
from contextlib import nullcontext

from nnarchs.architecture import SizeConstraint
from nnarchs.import_libs import is_cuda_available
from nnarchs.nn_types import Hdtype, ShapeStrategy
from nnarchs.utils.torch_tensor import HdtypeToTorch
from ...inference.session import PyTorchSession
if TYPE_CHECKING:
    from nnarchs.model import PyTorchModel




def parse(model: PyTorchModel) -> None:
    state_dict: StateDict = model.state_dict
    in_nc = 3

    model.update(
        arch_name=model.arch.name,
        scale=1,
        in_nc=in_nc,
        out_nc=in_nc,

        conf_thresh=0.6,
        min_support=1,
        gate_slope=12.0,
        count_slope=4.0
    )



def _to_onnx(
    model: PyTorchModel,
    dtype: Hdtype,
    opset: int = 20,
    shape_strategy: ShapeStrategy | None = None,
    device: str = 'cpu',
    export_args: dict = {},
) -> onnx.ModelProto | None:
    """Returns an Onnx model as a byte buffer
        if size is not None, use it to convert to a static shape
    """
    # TODO use model.arch to determine if static is possible or not
    print(f"[V] PyTorch to ONNX")
    import warnings
    def _show(message, category, filename, lineno, file=None, line=None):
        print(f"[TRACE] {filename}:{lineno}: {str(message)[:80]}")

    warnings.showwarning = _show
    warnings.simplefilter("always", torch.jit.TracerWarning)

    from .module.temporalfix_streaming import TemporalFixStreaming

    # No ModuleClass override: the session builds the regular TemporalFix
    session: PyTorchSession = model.arch.create_session(model)

    fp16: bool = bool(dtype == 'fp16')
    if not is_cuda_available():
        print("[W] cuda not available, fallback to cpu")
        device = 'cpu'
        fp16 = False

    # https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-convert-model
    print(f"[V] PyTorch to ONNX: initialize a session, device={device}, dtype={dtype}, fp16={fp16}, opset={opset}")
    # loads weights, device, dtype, eval()
    session.initialize(device=device, dtype=dtype)

    # Wrap AFTER the weights are loaded (the wrapper has no parameters of its own,
    # so device/dtype are preserved).
    core = session.module
    streaming = TemporalFixStreaming(core).eval()
    session.module = streaming
    model.module = streaming


    print(f"[V]  shape strategy: {shape_strategy}")
    input_names = list(streaming.onnx_input_names)
    output_names = list(streaming.onnx_output_names)

    if shape_strategy is not None and shape_strategy.type == 'static':
        w, h = shape_strategy.opt_size
        dynamic_axes = None
    else:
        # Dynamic H/W. Trace with a size that is NOT a degenerate value
        # (multiple of the largest scale, 8, and not tiny) so no shape gets specialized.
        size = model.size_constraint
        w, h = size.min if size is not None and size.min is not None else (64, 64)
        w, h = max(w, 64), max(h, 64)
        dynamic_axes = streaming.onnx_dynamic_axes

    tensor_dtype: torch.dtype = HdtypeToTorch[dtype]
    dummy_inputs = streaming.onnx_dummy_inputs(h, w, tensor_dtype, device)

    print(f"[V]   dummy inputs:")
    for d in dummy_inputs:
        print(f"[V]     d: {d.shape}, {d.dtype}, {d.device}")

    model_proto: onnx.ModelProto
    if tensor_dtype == torch.bfloat16:
        context = torch.autocast(
            device_type="cuda" if "cuda" in device else "cpu",
            dtype=tensor_dtype
        )
    else:
        context = nullcontext()

    with context:
        with BytesIO() as bytes_io:
            torch.onnx.export(
                session.module,
                dummy_inputs,
                bytes_io,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                opset_version=opset,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
                dynamo=False,
                **export_args,
            )
            bytes_io.seek(0)
            model_proto = onnx.load(bytes_io)

        try:
            onnx.checker.check_model(model_proto)
        except Exception as e:
            print(f"[E] Converted Onnx model is not valid: {type(e)}")
            return None

    print(f"[V] ONNX model proto generated")

    return model_proto



MODEL_ARCHITECTURES: tuple[NnPytorchArchitecture] = (
    NnPytorchArchitecture(
        name="temporalfix",
        detection_keys=(
            "encode.cnn0.weight",
            "aligner.blocks.0.conv0.0.0.weight",
            "conf_head.net.0.0.weight",
        ),
        module=Module(file="temporalfix_arch", class_name="TemporalFix"),
        parse=parse,
        dtypes=('fp32', 'fp16', 'bf16'),
        size_constraint=SizeConstraint(min=(8, 8), modulo=2),
        to_onnx = OnnxConv(
            dtypes=set(['fp32', 'fp16', 'bf16']),
            shape_strategy_types=set(['dynamic', 'static']),
            fct=_to_onnx,
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
