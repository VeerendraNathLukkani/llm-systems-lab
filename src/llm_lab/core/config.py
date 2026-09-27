"""Validated configuration for a vLLM run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# What vLLM accepts for --dtype. "auto" follows the dtype in the model's config.json.
DTYPES = frozenset({"auto", "half", "float16", "bfloat16", "float", "float32"})


class ConfigError(ValueError):
    """Raised when a configuration value would be rejected by vLLM, or is nonsense."""


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling defaults applied to every request unless a request overrides them."""

    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ConfigError(f"temperature must be >= 0 (0 means greedy), got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ConfigError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1, got {self.max_tokens}")


@dataclass(frozen=True)
class EngineConfig:
    """Everything needed to construct a vLLM ``LLM`` instance, validated on creation."""

    model: str
    dtype: str = "auto"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    trust_remote_code: bool = False
    seed: int | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def __post_init__(self) -> None:
        if not self.model or not self.model.strip():
            raise ConfigError("model must be a Hugging Face model id or a local path")
        if self.dtype not in DTYPES:
            raise ConfigError(f"dtype must be one of {sorted(DTYPES)}, got {self.dtype!r}")
        if self.max_model_len < 1:
            raise ConfigError(f"max_model_len must be >= 1, got {self.max_model_len}")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ConfigError(
                "gpu_memory_utilization is a fraction of total VRAM and must be in (0, 1], "
                f"got {self.gpu_memory_utilization}"
            )
        if self.tensor_parallel_size < 1:
            raise ConfigError(f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}")


def load_config(path: str | Path) -> EngineConfig:
    """Read the YAML form of EngineConfig, with `generation` as a nested mapping."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")

    settings = dict(raw)
    generation = settings.pop("generation", {}) or {}
    try:
        return EngineConfig(generation=GenerationConfig(**generation), **settings)
    except TypeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
