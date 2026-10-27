from __future__ import annotations
from collections.abc import Callable, Set
from dataclasses import dataclass, field
from hytils import absolute_path, path_split
from importlib import util as importlib_util
import inspect
import os
import sys
import onnx
from pathlib import Path

try:
    import torch
    torch_device = torch.device
except:
    torch_device = str
from torch import nn as nn
from typing import Any, Literal, Optional, overload
from .model import (
    OnnxModel,
    NnModel,
    PyTorchModel,
    SizeConstraint,
)
from .logger import nnlogger
from .nn_types import (
    Hdtype,
    NnArchitectureType,
    ShapeStrategy,
    ShapeStrategyType,
)
from .session import NnModelSession


ParseFunction = Callable[[NnModel], None]


@dataclass(slots=True)
class InferType:
    type: Literal[
        'simple',
        'inpaint',
        'temporal',
        'optical_flow'
    ] = 'simple'
    inputs: int = 1
    outputs: int = 1



@dataclass
class NnGenericArchitecture:
    name: str = 'unknown'
    type: NnArchitectureType = NnArchitectureType()
    category: str = 'unknown'

    # Function used to detect arch,
    # can be customized for more scalability
    detection_fct: tuple[Callable] | Callable = None

    """Parse a model object and update the model"""
    parse: ParseFunction = None

    create_session: Callable[[NnModel], NnModelSession] = None

    # Supported datatypes for inference and conversion
    #   TODO: as it may differ, use a dataclass
    dtypes: list[Hdtype] = field(default_factory=list)

    size_constraint: SizeConstraint = None

    # Simple, temporal, nb of inputs/outputs, etc.
    infer_type: InferType = field(default_factory=InferType)

    _locked: bool = field(default=False, init=False, repr=False)

    def lock(self):
        self._locked = True

    def __setattr__(self, key, value):
        if getattr(self, '_locked', False) and key != '_locked':
            raise AttributeError(f"Cannot modify '{key}'; object is locked.")
        super().__setattr__(key, value)

    @overload
    def update(self, d: dict[Any, Any]) -> None: ...
    @overload
    def update(self, k: str, v:Any) -> None: ...
    def update(self,
        d: Optional[dict[Any, Any]] = None,
        k: Optional[str] = None,
        v: Optional[Any] = None):
        """Only existing attributes will be updated"""
        if d is not None:
            for k, v in d.items():
                if k is not None and hasattr(self, k):
                    setattr(self, k, v)

        elif k is not None and hasattr(self, k):
            setattr(self, k, v)

    def __str__(self) -> str:
        class_str = f"{self.__class__}: {'{'}\n"
        for k, v in self.__dict__.items():
            # if k in ['model_proto', 'state_dict', 'engine', 'arch', 'framework']:
            #     class_str += (
            #         f"\t{k}: {f'{type(v).__name__} ...' if v is not None else 'None'}\n"
            #     )
            #     continue
            class_str += f"\t{k}: {type(v).__name__} = {v}\n"
        class_str += "}\n"
        return class_str


@dataclass(slots=True)
class OnnxConv:
    dtypes: Set[Hdtype] = field(
        # default_factory=lambda: {'fp32', 'fp16', 'bf16'}
        default_factory=set
    )
    shape_strategy_types: Set[ShapeStrategyType] = field(
        # default_factory=lambda: {'dynamic', 'static'}
        default_factory=set
    )
    fct: Callable[
        [
            PyTorchModel,
            Hdtype,
            int,    # opset
            bool,   # static
            torch.device | str,
            ShapeStrategy,
            int
        ],
        onnx.ModelProto
    ] = None


@dataclass(slots=True)
class TensorRTConv:
    # Some archs don't support strong typing,
    #   caution: conversion might fail or slower inference
    dtypes: Set[Hdtype] = field(
        # default_factory=lambda: {'fp32', 'fp16', 'bf16'}
        default_factory=set
    )
    weak_typing: bool = False
    shape_strategy_types: Set[ShapeStrategyType] = field(
        default_factory=lambda: {'dynamic', 'fixed', 'static'}
        # default_factory=set
    )


@dataclass
class Module:
    file: str = ""
    class_name: str = ""
    module_class: nn.Module | None = None


@dataclass
class NnPytorchArchitecture(NnGenericArchitecture):
    detection_keys: tuple[str | tuple[str]] | dict = field(default_factory=tuple)
    module: Module = field(default_factory=Module)

    to_onnx: OnnxConv = None
    to_tensorrt: TensorRTConv = None

    _caller_dir: Path = Path()


    def __post_init__(self):
        frame = inspect.currentframe()
        try:
            # Go up two levels: one for the __post_init__ call,
            #   and one for the instantiation
            caller_file = frame.f_back.f_back.f_globals.get('__file__', 'unknown')
        finally:
            del frame
        self._caller_dir = Path(caller_file).resolve().parent


    def import_module(self) -> None:
        """Dynamic import the nn.module
        """
        if self.module.module_class is not None:
            return

        nnarchs_dir = Path(__file__).resolve().parents[1]
        arch_module_key: str = ".".join(self._caller_dir.relative_to(nnarchs_dir).parts)

        arch_module: Module = self.module
        if any(not x for x in (arch_module.file, arch_module.class_name)):
            print("[W] Missing module package or class_name")

        nn_module_filepath = self._caller_dir.joinpath(
            "module", *arch_module.file.split(".")
        )
        if nn_module_filepath.suffix != ".py":
            nn_module_filepath = nn_module_filepath.with_suffix(".py")

        sys_module_key = f"{arch_module_key}.module.{arch_module.file}"
        # nnlogger.info(f"{sys_module_key}, {nn_module_filepath}")
        module_spec = importlib_util.spec_from_file_location(
            name=sys_module_key,
            location=nn_module_filepath
        )
        module = importlib_util.module_from_spec(module_spec)
        sys.modules[sys_module_key] = module
        module_spec.loader.exec_module(module)
        self.module.module_class = getattr(module, arch_module.class_name)



@dataclass
class NnOnnxArchitecture(NnGenericArchitecture):
    scale: int | None = None
    to_tensorrt: TensorRTConv = None


@dataclass
class NnTensorrtArchitecture(NnGenericArchitecture):
    version: str = ''



NnArchitecture = (
    NnGenericArchitecture
    | NnOnnxArchitecture
    | NnPytorchArchitecture
    | NnTensorrtArchitecture
)



# GetModelArchFct = Callable[
#     [str | Path, dict[str, dict], str | Any],
#     tuple[NnArchitecture | None, NnModelObject | None]
# ]

GetModelArchFct = Callable[[str | Path, dict[str, Any]], tuple[str, Any]]

def detect_model_arch(
    model: Any,
    architectures: dict[str, NnArchitecture]
) -> NnArchitecture | None:
    """Detect the model architecture and returns it"""

    for arch in architectures.values():
        nnlogger.debug(f"Try with arch: {arch.name}")

        if (detection := arch.detection_fct) is None:
            raise NotImplementedError(f"Detection function is not implemented for {arch.name}")

        if isinstance(detection, tuple | list):
            # List of keys for detection
            for func in detection:
                if func(model):
                    nnlogger.debug(f"{arch.name}: used the list of keys")
                    return arch
        else:
            # Use a customized function for detection
            if detection(model):
                nnlogger.debug(f"{arch.name}: used the custom method for detection")
                return arch

    return None
