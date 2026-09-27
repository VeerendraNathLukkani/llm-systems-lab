"""Engine tests against a stand-in for the vllm module.

Loading a real model needs a GPU and several GiB of weights, so these tests inject a
fake `vllm` into sys.modules and check the wiring around it: preflight checks, argument
passing, response accounting and error translation.
"""

import sys
import types

import pytest

from llm_lab.core.config import ConfigError, EngineConfig, GenerationConfig
from llm_lab.core.request import GenerationRequest
from llm_lab.inference.vllm import engine as engine_module
from llm_lab.inference.vllm.engine import (
    EngineError,
    EngineLoadError,
    GenerationError,
    VllmEngine,
)


class FakeCompletion:
    def __init__(self, text="generated", token_ids=(1, 2, 3), finish_reason="stop"):
        self.text = text
        self.token_ids = list(token_ids)
        self.finish_reason = finish_reason


class FakeRequestOutput:
    def __init__(self, prompt_token_ids=(1, 2), completion=None):
        self.prompt_token_ids = list(prompt_token_ids)
        self.outputs = [completion or FakeCompletion()]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeLLM.instances.append(self)
        self.llm_engine = types.SimpleNamespace(
            model_config=types.SimpleNamespace(dtype="torch.bfloat16", max_model_len=4096)
        )

    def generate(self, prompts, sampling_params):
        self.calls.append((prompts, sampling_params))
        return [FakeRequestOutput()]


@pytest.fixture
def fake_vllm(monkeypatch):
    FakeLLM.instances = []
    module = types.ModuleType("vllm")
    module.LLM = FakeLLM
    module.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(engine_module, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(engine_module, "empty_cuda_cache", lambda: None)
    return module


@pytest.fixture
def config():
    return EngineConfig(
        model="fake/model",
        dtype="bfloat16",
        max_model_len=4096,
        generation=GenerationConfig(temperature=0.5, top_p=0.8, max_tokens=32),
    )


def test_load_refuses_without_a_cuda_device(monkeypatch, config):
    monkeypatch.setattr(engine_module, "cuda_device_count", lambda: 0)

    with pytest.raises(EngineError, match="no CUDA device"):
        VllmEngine(config).load()


def test_load_refuses_tensor_parallel_larger_than_gpu_count(monkeypatch, config):
    monkeypatch.setattr(engine_module, "cuda_device_count", lambda: 2)
    engine = VllmEngine(EngineConfig(model="fake/model", tensor_parallel_size=4))

    with pytest.raises(EngineError, match="only 2 CUDA device"):
        engine.load()


def test_load_passes_config_through_to_vllm(fake_vllm, config):
    engine = VllmEngine(config)
    engine.load()

    assert FakeLLM.instances[0].kwargs == {
        "model": "fake/model",
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.90,
        "tensor_parallel_size": 1,
        "trust_remote_code": False,
    }
    assert engine.load_seconds >= 0


def test_unset_seed_is_not_passed_to_vllm(fake_vllm, config):
    VllmEngine(config).load()

    # vLLM rejects seed=None, so the argument has to be absent rather than empty.
    assert "seed" not in FakeLLM.instances[0].kwargs


def test_seed_is_passed_when_set(fake_vllm):
    VllmEngine(EngineConfig(model="fake/model", seed=0)).load()

    assert FakeLLM.instances[0].kwargs["seed"] == 0


def test_load_twice_is_rejected(fake_vllm, config):
    engine = VllmEngine(config)
    engine.load()

    with pytest.raises(EngineError, match="already loaded"):
        engine.load()


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"), "not enough GPU memory"),
        (ValueError("Repository Not Found for url ... 404"), "not a Hugging Face repo id"),
        (OSError("You are trying to access a gated repo (401)"), "gated or private"),
        (ValueError("Total number of attention heads (28) must be divisible by 3"), "divide evenly"),
        (RuntimeError("something obscure"), "failed to initialise"),
    ],
)
def test_load_failures_get_actionable_messages(monkeypatch, fake_vllm, config, raised, expected):
    def explode(**kwargs):
        raise raised

    monkeypatch.setattr(fake_vllm, "LLM", explode)
    engine = VllmEngine(config)

    with pytest.raises(EngineLoadError, match=expected) as excinfo:
        engine.load()
    assert excinfo.value.__cause__ is raised


def test_generate_uses_configured_sampling_parameters(fake_vllm, config):
    engine = VllmEngine(config)
    engine.load()

    engine.generate(GenerationRequest("hello"))

    prompts, params = FakeLLM.instances[0].calls[0]
    assert prompts == ["hello"]
    assert params.kwargs == {"temperature": 0.5, "top_p": 0.8, "max_tokens": 32}


def test_generate_overrides_apply_to_a_single_call(fake_vllm, config):
    engine = VllmEngine(config)
    engine.load()

    engine.generate(GenerationRequest("hello", temperature=0.0, max_tokens=8))
    engine.generate(GenerationRequest("hello"))

    first, second = (call[1].kwargs for call in FakeLLM.instances[0].calls)
    assert first == {"temperature": 0.0, "top_p": 0.8, "max_tokens": 8}
    assert second == {"temperature": 0.5, "top_p": 0.8, "max_tokens": 32}


def test_generate_validates_overrides(fake_vllm, config):
    engine = VllmEngine(config)
    engine.load()

    with pytest.raises(ConfigError, match="top_p"):
        engine.generate(GenerationRequest("hello", top_p=3))


def test_generate_counts_tokens_and_timing(fake_vllm, config):
    engine = VllmEngine(config)
    engine.load()

    result = engine.generate(GenerationRequest("hello"))

    assert result.text == "generated"
    assert result.prompt_tokens == 2
    assert result.output_tokens == 3
    assert result.finish_reason == "stop"
    assert result.elapsed_seconds > 0
    assert result.output_tokens_per_second == pytest.approx(3 / result.elapsed_seconds)


def test_generation_failure_keeps_the_original_exception(monkeypatch, fake_vllm, config):
    engine = VllmEngine(config)
    engine.load()
    boom = RuntimeError("engine core died")
    monkeypatch.setattr(FakeLLM.instances[0], "generate", lambda *a, **k: (_ for _ in ()).throw(boom))

    with pytest.raises(GenerationError, match="fake/model") as excinfo:
        engine.generate(GenerationRequest("hello"))
    assert excinfo.value.__cause__ is boom


def test_generate_before_load_is_rejected(config):
    with pytest.raises(EngineError, match="not loaded"):
        VllmEngine(config).generate(GenerationRequest("hello"))


def test_info_reports_resolved_model_settings(fake_vllm, config):
    engine = VllmEngine(config)
    engine.load()

    info = engine.info()

    assert info["model"] == "fake/model"
    assert info["dtype"] == "torch.bfloat16"
    assert info["max_model_len"] == 4096
    assert info["load_seconds"] == engine.load_seconds


def test_context_manager_loads_and_releases(fake_vllm, config):
    with VllmEngine(config) as engine:
        assert engine.llm is not None
    assert engine.llm is None


def test_close_is_idempotent(fake_vllm, config):
    engine = VllmEngine(config)
    engine.load()
    engine.close()
    engine.close()
    assert engine.llm is None
