"""LLM Systems Lab: vLLM inference, system inspection and basic benchmarking."""

from .benchmark.runner import BenchmarkResult, run_benchmark
from .core.config import ConfigError, EngineConfig, GenerationConfig, load_config
from .core.request import GenerationRequest
from .core.response import GenerationResponse
from .hardware.system import SystemInfo, collect_system_info
from .inference.vllm.engine import EngineError, VllmEngine

__version__ = "0.1.0"

__all__ = [
    "BenchmarkResult",
    "ConfigError",
    "EngineConfig",
    "EngineError",
    "GenerationConfig",
    "GenerationRequest",
    "GenerationResponse",
    "SystemInfo",
    "VllmEngine",
    "collect_system_info",
    "load_config",
    "run_benchmark",
    "__version__",
]
