"""Repeat-generation benchmark for a single vLLM model."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from ..core.config import EngineConfig
from ..core.request import GenerationRequest
from ..core.response import GenerationResponse
from ..hardware.system import gpu_memory_used_gb
from ..inference.vllm.engine import VllmEngine


@dataclass(frozen=True)
class BenchmarkResult:
    """Load time plus the per-run measurements from one benchmark invocation."""

    model: str
    prompt: str
    load_seconds: float
    runs: list[GenerationResponse]
    gpu_memory_used_gb: float | None

    @property
    def mean_latency_seconds(self) -> float:
        """Mean wall-clock time of one generation call."""
        return mean(run.elapsed_seconds for run in self.runs)

    @property
    def mean_output_tokens_per_second(self) -> float:
        """Mean decode throughput across runs."""
        return mean(run.output_tokens_per_second for run in self.runs)

    @property
    def mean_output_tokens(self) -> float:
        """Mean completion length; varies between runs whenever temperature > 0."""
        return mean(run.output_tokens for run in self.runs)


def run_benchmark(
    config: EngineConfig, prompt: str, runs: int = 3, warmup: int = 1
) -> BenchmarkResult:
    """Load the model once, then time `runs` generations of the same prompt.

    Warmup generations are discarded: the first call after load pays for CUDA graph
    capture and kernel autotuning, which would otherwise dominate the mean.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    request = GenerationRequest(prompt=prompt)
    engine = VllmEngine(config)
    engine.load()
    load_seconds = engine.load_seconds or 0.0
    try:
        for _ in range(warmup):
            engine.generate(request)
        results = [engine.generate(request) for _ in range(runs)]
        # Read the driver's figure while the engine is still up.
        memory_used = gpu_memory_used_gb()
    finally:
        engine.close()

    return BenchmarkResult(
        model=config.model,
        prompt=prompt,
        load_seconds=load_seconds,
        runs=results,
        gpu_memory_used_gb=memory_used,
    )


def format_benchmark(result: BenchmarkResult) -> str:
    """Render a benchmark result as the text block `llmlab benchmark` prints."""
    first = result.runs[0]
    lines = [
        f"model              {result.model}",
        f"runs               {len(result.runs)}",
        f"load               {result.load_seconds:.2f} s",
        f"prompt tokens      {first.prompt_tokens}",
        f"output tokens      {result.mean_output_tokens:.1f} (mean)",
        f"latency            {result.mean_latency_seconds:.2f} s (mean)",
        f"throughput         {result.mean_output_tokens_per_second:.1f} output tok/s (mean)",
    ]
    if result.gpu_memory_used_gb is not None:
        lines.append(f"gpu memory in use  {result.gpu_memory_used_gb:.1f} GiB (whole device)")
    return "\n".join(lines)
