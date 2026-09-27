"""Normalized result of a generation call."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationResponse:
    """What one prompt produced, with the measurements worth recording."""

    text: str
    prompt_tokens: int
    output_tokens: int
    elapsed_seconds: float
    finish_reason: str | None

    @property
    def output_tokens_per_second(self) -> float:
        """Decode throughput, counting only generated tokens."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.output_tokens / self.elapsed_seconds
