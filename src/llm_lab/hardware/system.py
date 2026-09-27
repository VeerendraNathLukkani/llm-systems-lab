"""Runtime facts about the machine that determine whether a vLLM run will work."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

# torch is imported lazily throughout this module: importing it costs a couple of
# seconds, and `llmlab info` should still say something useful if it is missing.

BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class GpuInfo:
    """A single visible CUDA device."""

    index: int
    name: str
    total_memory_gb: float
    compute_capability: str


@dataclass(frozen=True)
class SystemInfo:
    """Versions and GPU inventory, captured for diagnostics and reproducibility."""

    python_version: str
    platform: str
    torch_version: str | None
    vllm_version: str | None
    cuda_available: bool
    cuda_version: str | None
    gpus: list[GpuInfo]
    torch_error: str | None = None


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def collect_system_info() -> SystemInfo:
    """Collect the runtime information needed to diagnose a vLLM execution."""
    torch_version = _installed_version("torch")
    cuda_available = False
    cuda_version = None
    gpus: list[GpuInfo] = []

    torch_error = None
    if torch_version:
        try:
            import torch
        except ImportError as exc:
            # An installed torch that will not import is usually a truncated install or
            # a missing CUDA shared library. Report it instead of looking GPU-less.
            torch_error = str(exc)
        else:
            cuda_version = torch.version.cuda
            cuda_available = torch.cuda.is_available()
            for index in range(torch.cuda.device_count() if cuda_available else 0):
                props = torch.cuda.get_device_properties(index)
                gpus.append(
                    GpuInfo(
                        index=index,
                        name=props.name,
                        total_memory_gb=props.total_memory / BYTES_PER_GB,
                        compute_capability=f"{props.major}.{props.minor}",
                    )
                )

    return SystemInfo(
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        torch_version=torch_version,
        vllm_version=_installed_version("vllm"),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpus=gpus,
        torch_error=torch_error,
    )


def cuda_device_count() -> int:
    """Number of CUDA devices PyTorch can see, or 0 if torch is absent.

    Only a missing torch counts as zero devices. A torch that is installed but fails to
    import is a broken environment, not a machine without a GPU, so that error is left
    to propagate.
    """
    try:
        import torch
    except ModuleNotFoundError:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def gpu_memory_used_gb(index: int = 0) -> float | None:
    """VRAM in use on the device according to NVML, or None if NVML cannot answer.

    Device-wide on purpose: vLLM runs its engine in a separate process, so torch's
    per-process allocator stats miss most of it. NVML rather than
    torch.cuda.mem_get_info() because under WSL2 the latter reports a virtualized
    budget -- measured at 1.0 GiB against nvidia-smi's 5.4 GiB for the same engine.
    """
    try:
        import pynvml
    except ModuleNotFoundError:
        return None

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / BYTES_PER_GB
    finally:
        pynvml.nvmlShutdown()


def empty_cuda_cache() -> None:
    """Return cached CUDA blocks to the driver, if there is a CUDA runtime at all."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_system_info(info: SystemInfo) -> str:
    """Render system information as the aligned text block `llmlab info` prints."""
    lines = [
        f"python           {info.python_version}",
        f"platform         {info.platform}",
        f"torch            {info.torch_version or 'not installed'}",
        f"vllm             {info.vllm_version or 'not installed'}",
        f"cuda available   {info.cuda_available}",
        f"cuda (torch)     {info.cuda_version or 'n/a'}",
    ]
    if info.torch_error:
        lines.append(f"torch import     FAILED: {info.torch_error}")
    if not info.gpus:
        lines.append("gpus             none visible")
    for gpu in info.gpus:
        lines.append(
            f"gpu {gpu.index}            {gpu.name}, {gpu.total_memory_gb:.1f} GiB, "
            f"sm_{gpu.compute_capability.replace('.', '')}"
        )
    return "\n".join(lines)
