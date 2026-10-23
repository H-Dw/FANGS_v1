import inspect
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from esm.models.esm3 import ESM3
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.utils.constants.esm3 import data_root
from esm.utils.constants.models import (
    ESM3_FUNCTION_DECODER_V0,
    ESM3_OPEN_SMALL,
    ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0,
    ESM3_FINETUNING,
)

ModelBuilder = Callable[..., nn.Module]

# Optional file or directory used when a caller does not pass `checkpoint`.
# A file selects that weight when the filename matches, and sibling `.pth`
# files in the same directory for the encoder, decoder, and function decoder.
_CHECKPOINT_OVERRIDE: str | Path | None = None


def set_checkpoint(checkpoint: str | Path | None) -> None:
    """Remember a checkpoint file or directory for later local loads."""
    global _CHECKPOINT_OVERRIDE
    _CHECKPOINT_OVERRIDE = checkpoint


def resolve_weight_path(filename: str, checkpoint: str | Path | None = None) -> Path:
    """Resolve `filename` from an explicit checkpoint file or directory.

    `checkpoint` may be a `.pth` file or a directory. A matching file is used
    directly. Any other file is treated as a pointer to its parent directory,
    which must contain `filename`. A directory is searched for `filename`, then
    `data/weights/filename`. When omitted, the legacy `data_root()/data/weights`
    location is used.
    """
    source = _CHECKPOINT_OVERRIDE if checkpoint is None else checkpoint
    if source is None:
        return Path(data_root()) / "data" / "weights" / filename

    path = Path(source).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if path.is_file():
        if path.name == filename:
            return path
        sibling = path.parent / filename
        if sibling.is_file():
            return sibling
        raise FileNotFoundError(
            f"Checkpoint {path} does not provide {filename}. Expected {sibling}."
        )
    if path.is_dir():
        for candidate in (path / filename, path / "data" / "weights" / filename):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Could not find {filename} in checkpoint directory {path}."
        )
    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def _load_state_dict(filename: str, device: torch.device | str, checkpoint: str | Path | None = None):
    path = resolve_weight_path(filename, checkpoint)
    print(f"Loading ESM3 weights: {path}")
    return torch.load(path, map_location=device)


def ESM3_sm_open_v0_T(device: torch.device | str = "cpu", checkpoint: str | Path | None = None):
    model = (
        ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_name=ESM3_STRUCTURE_ENCODER_V0,
            structure_decoder_name=ESM3_STRUCTURE_DECODER_V0,
            function_decoder_name=ESM3_FUNCTION_DECODER_V0,
        )
        .to(device)
        .train()
    )
    state_dict = _load_state_dict("esm3_sm_open_v1.pth", device, checkpoint)
    model.load_state_dict(state_dict)
    model.checkpoint = checkpoint
    return model

def ESM3_sm_open_v0(device: torch.device | str = "cpu", checkpoint: str | Path | None = None):
    model = (
        ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_name=ESM3_STRUCTURE_ENCODER_V0,
            structure_decoder_name=ESM3_STRUCTURE_DECODER_V0,
            function_decoder_name=ESM3_FUNCTION_DECODER_V0,
        )
        .to(device)
        .eval()
    )
    state_dict = _load_state_dict("esm3_sm_open_v1.pth", device, checkpoint)
    model.load_state_dict(state_dict)
    model.checkpoint = checkpoint
    return model


def ESM3_structure_encoder_v0(device: torch.device | str = "cpu", checkpoint: str | Path | None = None):
    model = (
        StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        )
        .to(device)
        .eval()
    )
    state_dict = _load_state_dict("esm3_structure_encoder_v0.pth", device, checkpoint)
    model.load_state_dict(state_dict)
    return model


def ESM3_structure_decoder_v0(device: torch.device | str = "cpu", checkpoint: str | Path | None = None):
    model = (
        StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).to(device).eval()
    )
    state_dict = _load_state_dict("esm3_structure_decoder_v0.pth", device, checkpoint)
    model.load_state_dict(state_dict)
    return model


def ESM3_function_decoder_v0(device: torch.device | str = "cpu", checkpoint: str | Path | None = None):
    model = FunctionTokenDecoder().to(device).eval()
    state_dict = _load_state_dict("esm3_function_decoder_v0.pth", device, checkpoint)
    model.load_state_dict(state_dict)
    return model


LOCAL_MODEL_REGISTRY: dict[str, ModelBuilder] = {
    ESM3_OPEN_SMALL: ESM3_sm_open_v0,
    ESM3_STRUCTURE_ENCODER_V0: ESM3_structure_encoder_v0,
    ESM3_STRUCTURE_DECODER_V0: ESM3_structure_decoder_v0,
    ESM3_FUNCTION_DECODER_V0: ESM3_function_decoder_v0,
    ESM3_FINETUNING: ESM3_sm_open_v0_T,
}


def load_local_model(
    model_name: str,
    device: torch.device | str = "cpu",
    checkpoint: str | Path | None = None,
) -> nn.Module:
    if model_name not in LOCAL_MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in local model registry.")
    builder = LOCAL_MODEL_REGISTRY[model_name]
    if checkpoint is None:
        checkpoint = _CHECKPOINT_OVERRIDE
    parameters = inspect.signature(builder).parameters
    accepts_checkpoint = "checkpoint" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_checkpoint:
        return builder(device, checkpoint=checkpoint)
    return builder(device)


# Register custom versions of ESM3 for use with the local inference API
def register_local_model(model_name: str, model_builder: ModelBuilder) -> None:
    LOCAL_MODEL_REGISTRY[model_name] = model_builder
