from __future__ import annotations
import copy
import gc
from hytils import (
    absolute_path,
    is_access_granted,
    parent_directory,
    red,
)
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from safetensors.torch import save_file

from .load import load_state_dict
from nnarchs.metadata import generate_metadata
if TYPE_CHECKING:
    from nnarchs.model import PyTorchModel


def save_as(
    model: PyTorchModel,
    filepath: Path | None = None,
    directory: Path | None = None,
    basename: str | None = None,
    ext: Literal['.pth', '.safetensors'] = '.pth',
) -> Path:

    if filepath is not None:
        directory = filepath.parent

    else:
        filepath: Path = directory / f"{basename}{ext}"

    directory = directory.resolve()
    if not is_access_granted(directory, 'w'):
        raise PermissionError(f"{directory} is not writable")

    metadata: dict[str, str] = generate_metadata(model, model.metadata)

    state_dict, _ = load_state_dict(model.filepath, device='cpu')
    if state_dict is None:
        raise ValueError(red(f"Not a supported model") + f"{model.filepath}")

    if 'metadata' in state_dict:
        del state_dict['metadata']

    if ext == '.pth':
        state_dict[f'metadata'] = json.dumps(metadata)
        torch.save(state_dict, filepath)

    elif ext == '.safetensors':
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and not v.is_contiguous():
                state_dict[k] = v.contiguous()

        tmp_fp = filepath.with_name(filepath.name + ".tmp")
        save_file(state_dict, tmp_fp, metadata=metadata)
        os.replace(tmp_fp, filepath)

    else:
        raise ValueError(f"Not supported: {ext}")

    return filepath
