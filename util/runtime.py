import os
from typing import Any, Mapping


def _safe_call(default: Any, fn, *args):
    try:
        return fn(*args)
    except Exception:
        return default


def collect_accelerator_diagnostics(torch_module, environ: Mapping[str, str] | None = None) -> dict:
    """Collect lightweight accelerator state without requiring CUDA to work."""
    env = os.environ if environ is None else environ
    cuda = getattr(torch_module, "cuda", None)
    version = getattr(torch_module, "version", None)

    cuda_available = bool(_safe_call(False, cuda.is_available)) if cuda is not None else False
    cuda_device_count = int(_safe_call(0, cuda.device_count)) if cuda is not None else 0
    cuda_device_name = None
    if cuda_available and cuda_device_count > 0 and hasattr(cuda, "get_device_name"):
        cuda_device_name = _safe_call(None, cuda.get_device_name, 0)

    return {
        "cuda_available": cuda_available,
        "cuda_version": getattr(version, "cuda", None),
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
    }


def format_accelerator_diagnostics(diagnostics: Mapping[str, Any]) -> str:
    visible = diagnostics.get("cuda_visible_devices")
    if visible is None or visible == "":
        visible = "<unset>"
    name = diagnostics.get("cuda_device_name") or "<none>"
    return (
        f"torch.cuda.is_available={diagnostics.get('cuda_available')}, "
        f"torch.version.cuda={diagnostics.get('cuda_version')}, "
        f"cuda.device_count={diagnostics.get('cuda_device_count')}, "
        f"cuda.device_name={name}, "
        f"CUDA_VISIBLE_DEVICES={visible}"
    )


def resolve_training_device(torch_module, require_cuda: bool = False):
    diagnostics = collect_accelerator_diagnostics(torch_module)
    if require_cuda and not diagnostics["cuda_available"]:
        raise RuntimeError(
            "CUDA was required (--require_cuda), but torch.cuda.is_available() is False. "
            f"Diagnostics: {format_accelerator_diagnostics(diagnostics)}"
        )
    return torch_module.device("cuda" if diagnostics["cuda_available"] else "cpu")


def should_pin_memory(device, no_pin_memory: bool = False) -> bool:
    if no_pin_memory:
        return False
    return getattr(device, "type", str(device)) == "cuda"


def log_accelerator_diagnostics(logger, diagnostics: Mapping[str, Any]) -> None:
    logger.info(f"Accelerator diagnostics: {format_accelerator_diagnostics(diagnostics)}")
    if not diagnostics.get("cuda_available", False):
        logger.warning(
            "CUDA is not available in this Python process; training will run on CPU. "
            "For heavy models this can be tens of seconds per batch. "
            "Use --require_cuda to fail fast instead of accidentally training on CPU."
        )
