"""Benchmark tests against a fake engine, so no model is ever loaded."""

import pytest

from llm_lab.benchmark import runner
from llm_lab.benchmark.runner import BenchmarkResult, format_benchmark, run_benchmark
from llm_lab.core.config import EngineConfig
from llm_lab.core.response import GenerationResponse


def response(output_tokens, elapsed_seconds):
    return GenerationResponse(
        text="x" * output_tokens,
        prompt_tokens=4,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed_seconds,
        finish_reason="stop",
    )


class FakeEngine:
    instances = []

    def __init__(self, config):
        self.config = config
        self.load_seconds = None
        self.prompts = []
        self.closed = False
        FakeEngine.instances.append(self)

    def load(self):
        self.load_seconds = 12.5

    def generate(self, request):
        self.prompts.append(request.prompt)
        return response(output_tokens=10 * len(self.prompts), elapsed_seconds=2.0)

    def close(self):
        self.closed = True
        # Mirrors releasing engine state: the benchmark must have recorded the load
        # time before this point.
        self.load_seconds = None


@pytest.fixture
def fake_engine(monkeypatch):
    FakeEngine.instances = []
    monkeypatch.setattr(runner, "VllmEngine", FakeEngine)
    monkeypatch.setattr(runner, "gpu_memory_used_gb", lambda: 38.0)
    return FakeEngine


@pytest.fixture
def config():
    return EngineConfig(model="fake/model")


def test_warmup_generations_are_excluded_from_the_results(fake_engine, config):
    result = run_benchmark(config, "hello", runs=3, warmup=2)

    assert len(fake_engine.instances[0].prompts) == 5
    assert len(result.runs) == 3
    # The first two responses (10 and 20 output tokens) are discarded.
    assert [run.output_tokens for run in result.runs] == [30, 40, 50]


def test_load_time_is_recorded_before_the_engine_is_closed(fake_engine, config):
    result = run_benchmark(config, "hello", runs=1, warmup=0)

    assert result.load_seconds == 12.5
    assert fake_engine.instances[0].closed


def test_engine_is_closed_even_when_generation_fails(fake_engine, config, monkeypatch):
    def explode(request):
        raise RuntimeError("engine core died")

    monkeypatch.setattr(FakeEngine, "generate", staticmethod(explode))

    with pytest.raises(RuntimeError):
        run_benchmark(config, "hello", runs=1, warmup=0)
    assert fake_engine.instances[0].closed


def test_gpu_memory_is_read_while_the_engine_is_alive(fake_engine, config):
    assert run_benchmark(config, "hello", runs=1, warmup=0).gpu_memory_used_gb == 38.0


@pytest.mark.parametrize(("runs", "warmup"), [(0, 1), (-1, 0), (1, -1)])
def test_invalid_run_counts_are_rejected(fake_engine, config, runs, warmup):
    with pytest.raises(ValueError):
        run_benchmark(config, "hello", runs=runs, warmup=warmup)
    assert FakeEngine.instances == []


def test_means_are_computed_across_runs():
    result = BenchmarkResult(
        model="fake/model",
        prompt="hello",
        load_seconds=10.0,
        runs=[response(100, 2.0), response(200, 2.0)],
        gpu_memory_used_gb=None,
    )

    assert result.mean_output_tokens == 150
    assert result.mean_latency_seconds == 2.0
    assert result.mean_output_tokens_per_second == pytest.approx(75.0)


def test_format_omits_gpu_memory_when_it_is_unavailable():
    result = BenchmarkResult(
        model="fake/model",
        prompt="hello",
        load_seconds=10.0,
        runs=[response(100, 2.0)],
        gpu_memory_used_gb=None,
    )

    text = format_benchmark(result)

    assert "gpu memory" not in text
    assert "throughput         50.0 output tok/s (mean)" in text


def test_format_includes_gpu_memory_when_it_is_available():
    result = BenchmarkResult(
        model="fake/model",
        prompt="hello",
        load_seconds=10.0,
        runs=[response(100, 2.0)],
        gpu_memory_used_gb=38.4,
    )

    assert "gpu memory in use  38.4 GiB (whole device)" in format_benchmark(result)
