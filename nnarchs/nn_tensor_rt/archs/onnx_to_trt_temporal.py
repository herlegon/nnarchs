from copy import deepcopy
from hytils import red, yellow
import numpy as np
from pprint import pprint

from nnarchs.logger import nnlogger
from nnarchs.nn_types import Hdtype, ShapeStrategy
from nnarchs.import_libs import trt
TrtDType = trt.DataType

from nnarchs.model import OnnxModel
from ..inference.session import TRT_LOGGER
from ..trt_types import TrtEngine




def temporal_onnx_to_trt_engine(
    model: OnnxModel,
    device: str,
    dtypes: set[Hdtype],
    shape_strategy: ShapeStrategy,
) -> TrtEngine:
    """
    Convert a temporal ONNX model to a serialized TensorRT engine.

    The model may have several inputs and outputs, for example:
        inputs:  x (1,7,3,H,W), feats (1,7,4,H,W), new_frame (1,3,H,W), new_mask (1,7,1,1,1)
        outputs: output (1,3,H,W), new_feat (1,4,H,W)

    Static ONNX: no optimization profile is needed.
    Dynamic ONNX: a single profile is built, where the dynamic spatial dims (the last
    two dims of every input) follow the shape strategy and other dynamic dims (batch) are 1.
    """
    # Typing is strong by default, it can be forced to weak
    # if the initial arch forces it or if forced by the user.
    use_strongly_typed: bool = True
    if model.torch_arch is not None:
        use_strongly_typed = not model.torch_arch.to_tensorrt.weak_typing
        if model.force_weak_typing:
            use_strongly_typed = False
    print(red(f"use_strongly_typed: {use_strongly_typed}"))

    has_fp16: bool = bool('fp16' in dtypes)
    has_bf16: bool = bool('bf16' in dtypes)

    print(f"[V] Start converting to TRT, request fp16={has_fp16}, bf16={has_bf16}")
    network_flags = 0
    if use_strongly_typed:
        network_flags += 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)

    with (
        trt.Builder(TRT_LOGGER) as builder,
        builder.create_network(flags=network_flags) as network,
        trt.OnnxParser(network, TRT_LOGGER) as onnx_parser
    ):
        runtime = trt.Runtime(TRT_LOGGER)

        # Parse the serializedonnx model
        if model.model_proto is not None:
            # Serialize the onnx model. Don't use filepath as it may
            # not an optimized onnx model
            serialized_onnx_model = model.model_proto.SerializeToString()
            success = onnx_parser.parse(serialized_onnx_model)
        if not success:
            for idx in range(onnx_parser.num_errors):
                print(f"Error: {onnx_parser.get_error(idx)}")
            raise ValueError("TensorRT: failed to parse the ONNX model")

        # Inputs / outputs: the temporal model has several of each
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        if not inputs:
            raise ValueError("Missing input tensor in model")

        for t in inputs:
            print(f"[V]   input  '{t.name}': shape={tuple(t.shape)}, dtype={t.dtype}")
        for t in outputs:
            print(f"[V]   output '{t.name}': shape={tuple(t.shape)}, dtype={t.dtype}")

        is_onnx_fp16 = any(t.dtype == trt.DataType.HALF for t in inputs)
        is_onnx_bf16 = any(t.dtype == trt.DataType.BF16 for t in inputs)
        print(f"[V]   is_onnx_fp16={is_onnx_fp16}, is_onnx_bf16={is_onnx_bf16}, "
              f"to_fp16={has_fp16}, to_bf16={has_bf16}")

        # Create a build configuration specifying how TensorRT should optimize the model
        builder_config = builder.create_builder_config()

        # for debug
        builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        if use_strongly_typed:
            # The precision is taken from the ONNX tensors: nothing to set.
            # A requested dtype that the ONNX does not carry would be silently ignored.
            if has_fp16 and not is_onnx_fp16:
                print(yellow("[W] fp16 requested but the ONNX is not fp16: "
                             "export the ONNX in fp16, the engine will not be fp16"))
            if has_bf16 and not is_onnx_bf16:
                print(yellow("[W] bf16 requested but the ONNX is not bf16"))
        else:
            if is_onnx_fp16 or has_fp16:
                builder_config.set_flag(trt.BuilderFlag.FP16)
                print("[V]   set fp16 flag")
            if is_onnx_bf16 or has_bf16:
                builder_config.set_flag(trt.BuilderFlag.BF16)
                print("[V]   set bf16 flag")


        # Static / dynamic: an input is dynamic if any of its dims is -1
        dynamic_inputs = [t for t in inputs if -1 in tuple(t.shape)]
        strategy: ShapeStrategy = deepcopy(shape_strategy)
        fixed_trt: bool = shape_strategy.is_fixed()

        if not dynamic_inputs:
            print(yellow("onnx_to_trt_engine: static onnx model"))
        else:
            if fixed_trt:
                print(yellow("onnx_to_trt_engine: fixed trt"))
                min_wh = opt_wh = max_wh = strategy.opt_size
            else:
                print(yellow("onnx_to_trt_engine: dynamic onnx model"))
                if strategy.min_size == (0, 0):
                    strategy.min_size = strategy.opt_size
                if strategy.max_size == (0, 0):
                    strategy.max_size = strategy.opt_size
                min_wh, opt_wh, max_wh = strategy.min_size, strategy.opt_size, strategy.max_size

            def _resolve(tensor, wh: tuple[int, int]) -> tuple[int, ...]:
                # Strategy sizes are (w, h). The last two dims of an input are (H, W),
                # any other dynamic dim (batch) is set to 1.
                w, h = wh
                dims = list(tensor.shape)
                n = len(dims)
                for idx, d in enumerate(dims):
                    if d == -1:
                        dims[idx] = h if idx == n - 2 else (w if idx == n - 1 else 1)
                return tuple(dims)

            # A single profile covers all the dynamic inputs (same H/W bounds).
            # Fully static inputs (e.g. new_mask) are left untouched.
            profile = builder.create_optimization_profile()
            for t in dynamic_inputs:
                shapes = (
                    _resolve(t, min_wh),
                    _resolve(t, opt_wh),
                    _resolve(t, max_wh)
                )
                print(f"[V]   profile '{t.name}': min={shapes[0]}, opt={shapes[1]}, max={shapes[2]}")
                profile.set_shape(input=t.name, min=shapes[0], opt=shapes[1], max=shapes[2])
            builder_config.add_optimization_profile(profile)

        nnlogger.info("[I] Building a TensortRT engine; this may take a while...")
        engine_bytes = builder.build_serialized_network(network, builder_config)
        if engine_bytes is None:
            raise RuntimeError("Failed to create Tensor RT engine")
        trt_engine = runtime.deserialize_cuda_engine(engine_bytes)

    return trt_engine


