"""Model construction and checkpoint helpers for the RF Challenge mode."""

from __future__ import annotations

import logging
import importlib
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# ``src`` must remain a top-level import during legacy torch unpickling. Its
# parent is registered lazily, so the offline bundle lives directly inside the
# challenge package at ``rfchallenge/src``.
OFFICIAL_STARTER_ROOT = Path(__file__).resolve().parent
DEFAULT_RFCHALLENGE_STAGE = 255

# Keep the challenge-facing model surface explicit.  The main experiment
# registry contains many stages whose heads and auxiliary objectives are
# written for two-source blind separation.  Every entry below points to a
# dedicated single-SOI configuration with a two-channel I/Q output head.
RFCHALLENGE_STAGE_CONFIGS: dict[int, Path] = {
    4: PACKAGE_ROOT / "config" / "model_config_stage4_rfchallenge.yaml",
    12: PACKAGE_ROOT / "config" / "model_config_stage12_rfchallenge.yaml",
    79: PACKAGE_ROOT / "config" / "model_config_stage79_rfchallenge.yaml",
    197: PACKAGE_ROOT / "config" / "model_config_stage197_rfchallenge.yaml",
    235: PACKAGE_ROOT / "config" / "model_config_stage235_rfchallenge.yaml",
    255: PACKAGE_ROOT / "config" / "model_config_stage255_rfchallenge.yaml",
    261: PACKAGE_ROOT / "config" / "model_config_stage261_rfchallenge.yaml",
    290: PACKAGE_ROOT / "config" / "model_config_stage290_rfchallenge.yaml",
    295: PACKAGE_ROOT / "config" / "model_config_stage295_rfchallenge.yaml",
    299: PACKAGE_ROOT / "config" / "model_config_stage299_rfchallenge.yaml",
    309: PACKAGE_ROOT / "config" / "model_config_stage309_rfchallenge.yaml",
    310: PACKAGE_ROOT / "config" / "model_config_stage310_rfchallenge.yaml",
    333: PACKAGE_ROOT / "config" / "model_config_stage333_rfchallenge.yaml",
    336: PACKAGE_ROOT / "config" / "model_config_stage336_rfchallenge.yaml",
    342: PACKAGE_ROOT / "config" / "model_config_stage342_rfchallenge.yaml",
    350: PACKAGE_ROOT / "config" / "model_config_stage350_rfchallenge.yaml",
    351: PACKAGE_ROOT / "config" / "model_config_stage351_rfchallenge.yaml",
    352: PACKAGE_ROOT / "config" / "model_config_stage352_rfchallenge.yaml",
    353: PACKAGE_ROOT / "config" / "model_config_stage353_rfchallenge.yaml",
    354: PACKAGE_ROOT / "config" / "model_config_stage354_rfchallenge.yaml",
    355: PACKAGE_ROOT / "config" / "model_config_stage355_rfchallenge.yaml",
    356: PACKAGE_ROOT / "config" / "model_config_stage356_rfchallenge.yaml",
    357: PACKAGE_ROOT / "config" / "model_config_stage357_rfchallenge.yaml",
    358: PACKAGE_ROOT / "config" / "model_config_stage358_rfchallenge.yaml",
}


def supported_rfchallenge_stages() -> tuple[int, ...]:
    """Return stage IDs with dedicated single-SOI challenge configs."""

    return tuple(sorted(RFCHALLENGE_STAGE_CONFIGS))


def resolve_stage_config(stage: int) -> Path:
    """Resolve a challenge stage to its dedicated config with a clear error."""

    stage_id = int(stage)
    try:
        config_path = RFCHALLENGE_STAGE_CONFIGS[stage_id]
    except KeyError as error:
        supported = ", ".join(str(value) for value in supported_rfchallenge_stages())
        raise ValueError(
            f"Stage {stage_id} is not registered for the RF Challenge; "
            f"supported stages: {supported}"
        ) from error
    if not config_path.is_file():
        raise FileNotFoundError(
            f"RF Challenge config for Stage {stage_id} is missing: {config_path}"
        )
    return config_path


def select_device(requested: str | None = None) -> torch.device:
    """Select a device without requiring CUDA for protocol smoke tests."""

    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_single_soi_model(
    config_path: str | Path,
    frame_length: int,
    device: torch.device | str | None = None,
    logger: logging.Logger | None = None,
    model_overrides: Mapping[str, Any] | None = None,
) -> tuple[torch.nn.Module, Any]:
    """Build any IQUMamba model with the RF Challenge's two-channel head.

    Existing IQUMamba experiment configs normally use ``num_classes=4`` for
    two-source BSS and may enable deep supervision. The RF Challenge's
    known-SOI task has one complex target and one waveform loss, so the
    runtime overrides are deliberately local and do not modify YAML on disk:
    two input I/Q channels, two output I/Q channels, and no auxiliary
    deep-supervision heads.
    """

    try:
        from util.config import MambaConfig
        from util.utils import Create_Mamba_model
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Run RF Challenge commands from the IQUMamba1D directory, for example "
            "'python -m rfchallenge.cli ...'."
        ) from error

    runtime_device = select_device(str(device) if device is not None else None)
    config = MambaConfig(str(config_path))
    if model_overrides:
        config.model_config.update(dict(model_overrides))
    # Keep the single-SOI contract authoritative even if a generic IQUMamba
    # experiment YAML was originally written for two-source separation.
    # This keeps one unambiguous full-resolution SOI head: intermediate
    # deep-supervision resolutions do not belong to the RF Challenge
    # objective.
    config.model_config.update(
        {
            "input_channels": 2,
            "num_classes": 2,
            "deep_supervision": False,
        }
    )
    model = Create_Mamba_model(
        config,
        logger=logger,
        input_size_=int(frame_length),
        device_override=runtime_device,
    )
    model = model.to(runtime_device)
    return model, config


def extract_single_soi_output(model_output: Any) -> torch.Tensor:
    """Normalize IQUMamba's tensor/list/auxiliary output contracts to ``(B,2,L)``."""

    output = model_output
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, (list, tuple)):
        output = output[-1]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    if output.ndim != 3 or output.shape[1] != 2:
        raise ValueError(
            "RF Challenge models must return a single complex SOI with shape "
            f"(B, 2, L); got {tuple(output.shape)}"
        )
    return output


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    best_validation_loss: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a portable checkpoint without symbolic links or implicit state."""

    payload: dict[str, Any] = {
        "format": "iqumamba-rfchallenge-v1",
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if best_validation_loss is not None:
        payload["best_validation_loss"] = float(best_validation_loss)
    if metadata:
        payload["metadata"] = metadata
    return payload


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically save a checkpoint using Windows-compatible ``os.replace``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve a checkpoint file or an official released-weight directory.

    The starter kit points ``model_dir`` at a case directory and always loads
    ``model_dir/weights.pt``.  Accepting that directory directly keeps Kaggle
    dataset paths usable without copying read-only inputs into ``/working``.
    """

    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"Checkpoint is neither a file nor directory: {candidate}")

    preferred_names = ("weights.pt", "best.pt", "checkpoint.pt")
    for name in preferred_names:
        preferred = candidate / name
        if preferred.is_file():
            return preferred
    discovered = sorted(
        path
        for pattern in ("*.pt", "*.pth", "*.ckpt")
        for path in candidate.glob(pattern)
        if path.is_file()
    )
    if len(discovered) == 1:
        return discovered[0]
    if not discovered:
        raise FileNotFoundError(
            f"No checkpoint file found in directory {candidate}; expected weights.pt"
        )
    names = ", ".join(path.name for path in discovered)
    raise ValueError(
        f"Ambiguous checkpoint directory {candidate}; choose one file explicitly: {names}"
    )


def _normalize_checkpoint_state(
    model: torch.nn.Module,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize DataParallel and old local WaveNet parameter names."""

    normalized = {
        key.removeprefix("module."): value for key, value in state.items()
    }
    target_keys = set(model.state_dict())
    source_has_official_layers = any(
        key.startswith("residual_layers.") for key in normalized
    )
    source_has_legacy_blocks = any(
        key.startswith("residual_blocks.") for key in normalized
    )
    target_has_official_layers = any(
        key.startswith("residual_layers.") for key in target_keys
    )
    target_has_legacy_blocks = any(
        key.startswith("residual_blocks.") for key in target_keys
    )
    if source_has_legacy_blocks and target_has_official_layers:
        normalized = {
            (
                "residual_layers." + key.removeprefix("residual_blocks.")
                if key.startswith("residual_blocks.")
                else key
            ): value
            for key, value in normalized.items()
        }
    elif source_has_official_layers and target_has_legacy_blocks:
        normalized = {
            (
                "residual_blocks." + key.removeprefix("residual_layers.")
                if key.startswith("residual_layers.")
                else key
            ): value
            for key, value in normalized.items()
        }
    return normalized


def _register_official_checkpoint_compatibility() -> Path:
    """Expose the vendored upstream ``src`` package for legacy unpickling."""

    source_package = OFFICIAL_STARTER_ROOT / "src"
    if not source_package.is_dir():
        raise ModuleNotFoundError(
            "Official WaveNet checkpoint references the upstream 'src' package, "
            "but the bundled compatibility source is missing: "
            f"{source_package}. Upload the complete IQUMamba project directory."
        )
    root_text = str(OFFICIAL_STARTER_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        importlib.invalidate_caches()
    return OFFICIAL_STARTER_ROOT


def _torch_load_checkpoint(
    checkpoint_path: Path,
    runtime_device: torch.device,
) -> Any:
    """Load a trusted checkpoint, retrying with the bundled official package."""

    def _load() -> Any:
        try:
            return torch.load(
                checkpoint_path,
                map_location=runtime_device,
                weights_only=False,
            )
        except TypeError:
            return torch.load(checkpoint_path, map_location=runtime_device)

    try:
        return _load()
    except ModuleNotFoundError as error:
        missing_module = str(getattr(error, "name", "") or "")
        if missing_module != "src" and not missing_module.startswith("src."):
            raise
        _register_official_checkpoint_compatibility()
        return _load()


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    device: torch.device | str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load new RF Challenge, official WaveNet, or raw state-dict checkpoints."""

    checkpoint_path = resolve_checkpoint_path(path)
    runtime_device = select_device(str(device) if device is not None else None)
    # The released artifact is trusted competition data. ``weights_only=False``
    # is required by older files containing OmegaConf/src configuration objects.
    loaded = _torch_load_checkpoint(checkpoint_path, runtime_device)

    if isinstance(loaded, dict):
        state = (
            loaded.get("model_state_dict")
            or loaded.get("model")
            or loaded.get("state_dict")
            or loaded
        )
    else:
        state = loaded
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint payload in {checkpoint_path}")
    normalized_state = _normalize_checkpoint_state(model, state)
    model.load_state_dict(normalized_state, strict=strict)
    if optimizer is not None and isinstance(loaded, dict) and "optimizer_state_dict" in loaded:
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
    return loaded if isinstance(loaded, dict) else {}
