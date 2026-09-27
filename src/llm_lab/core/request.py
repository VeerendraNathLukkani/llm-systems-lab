"""Input to a single generation call."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import GenerationConfig


@dataclass(frozen=True)
class GenerationRequest:
    """One prompt, plus any sampling values that should differ from the engine defaults."""

    prompt: str
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt is empty")

    def sampling(self, defaults: GenerationConfig) -> GenerationConfig:
        """Merge this request's overrides onto the engine defaults.

        ``replace`` re-runs GenerationConfig validation, so an out-of-range override is
        rejected here rather than inside vLLM.
        """
        overrides = {
            field: value
            for field, value in (
                ("temperature", self.temperature),
                ("top_p", self.top_p),
                ("max_tokens", self.max_tokens),
            )
            if value is not None
        }
        return replace(defaults, **overrides) if overrides else defaults
