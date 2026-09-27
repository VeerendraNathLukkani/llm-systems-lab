"""CLI wiring tests against a fake engine.

The engine has its own tests; what is checked here is that the CLI hands it the right
objects and prints to the right streams. A previous refactor left `llmlab run` passing
a raw string where the engine expects a GenerationRequest, and nothing caught it.
"""

import pytest

from llm_lab import cli
from llm_lab.core.request import GenerationRequest
from llm_lab.core.response import GenerationResponse


class FakeEngine:
    instances = []

    def __init__(self, config):
        self.config = config
        self.requests = []
        FakeEngine.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def info(self):
        return {"model": self.config.model, "max_model_len": self.config.max_model_len}

    def generate(self, request):
        assert isinstance(request, GenerationRequest)
        self.requests.append(request)
        return GenerationResponse(
            text="paged attention is ...",
            prompt_tokens=8,
            output_tokens=20,
            elapsed_seconds=1.0,
            finish_reason="stop",
        )


@pytest.fixture
def fake_engine(monkeypatch):
    FakeEngine.instances = []
    monkeypatch.setattr(cli, "VllmEngine", FakeEngine)
    return FakeEngine


def test_run_sends_the_prompt_as_a_request(fake_engine, capsys):
    assert cli.main(["run", "--model", "fake/model", "--prompt", "explain paged attention"]) == 0

    request = fake_engine.instances[0].requests[0]
    assert request.prompt == "explain paged attention"
    assert request.max_tokens is None


def test_run_applies_sampling_flags_to_the_request(fake_engine):
    cli.main(
        ["run", "--model", "fake/model", "--prompt", "hi", "--temperature", "0", "--max-tokens", "8"]
    )

    config = fake_engine.instances[0].config
    assert config.generation.temperature == 0.0
    assert config.generation.max_tokens == 8
    assert config.generation.top_p == 0.9


def test_run_applies_engine_flags(fake_engine):
    cli.main(
        [
            "run",
            "--model", "fake/model",
            "--prompt", "hi",
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.8",
            "--dtype", "bfloat16",
        ]
    )

    config = fake_engine.instances[0].config
    assert config.max_model_len == 2048
    assert config.gpu_memory_utilization == 0.8
    assert config.dtype == "bfloat16"


def test_run_puts_only_the_completion_on_stdout(fake_engine, capsys):
    cli.main(["run", "--model", "fake/model", "--prompt", "hi"])

    captured = capsys.readouterr()
    assert captured.out.strip() == "paged attention is ..."
    assert "loading fake/model" in captured.err
    assert "20 output tokens" in captured.err


def test_info_needs_no_engine(capsys):
    assert cli.main(["info"]) == 0
    assert "python" in capsys.readouterr().out


def test_missing_model_is_an_error(capsys):
    assert cli.main(["run", "--prompt", "hi"]) == 1
    assert "pass --model or --config" in capsys.readouterr().err


def test_invalid_sampling_flag_is_an_error(capsys):
    assert cli.main(["run", "--model", "fake/model", "--prompt", "hi", "--top-p", "5"]) == 1
    assert "top_p must be in (0, 1]" in capsys.readouterr().err


def test_engine_failure_reports_the_chained_cause(monkeypatch, capsys):
    original = RuntimeError("CUDA out of memory")

    def explode(config):
        raise cli.EngineError("not enough GPU memory") from original

    monkeypatch.setattr(cli, "VllmEngine", explode)

    assert cli.main(["run", "--model", "fake/model", "--prompt", "hi"]) == 1
    err = capsys.readouterr().err
    assert "error: not enough GPU memory" in err
    assert "caused by: RuntimeError: CUDA out of memory" in err
