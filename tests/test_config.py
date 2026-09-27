import pytest

from llm_lab.core.config import ConfigError, EngineConfig, GenerationConfig, load_config
from llm_lab.core.request import GenerationRequest
from llm_lab.core.response import GenerationResponse


def test_defaults_are_usable():
    config = EngineConfig(model="Qwen/Qwen3-8B")
    assert config.dtype == "auto"
    assert config.tensor_parallel_size == 1
    assert config.generation.max_tokens == 256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": ""},
        {"model": "   "},
        {"model": "m", "dtype": "fp16"},
        {"model": "m", "max_model_len": 0},
        {"model": "m", "gpu_memory_utilization": 0},
        {"model": "m", "gpu_memory_utilization": 1.5},
        {"model": "m", "tensor_parallel_size": 0},
    ],
)
def test_engine_config_rejects_bad_values(kwargs):
    with pytest.raises(ConfigError):
        EngineConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"top_p": 0},
        {"top_p": 1.2},
        {"max_tokens": 0},
    ],
)
def test_generation_config_rejects_bad_values(kwargs):
    with pytest.raises(ConfigError):
        GenerationConfig(**kwargs)


def test_temperature_zero_is_allowed_for_greedy_decoding():
    assert GenerationConfig(temperature=0).temperature == 0


def test_load_config_reads_nested_generation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "model: Qwen/Qwen3-8B\n"
        "dtype: bfloat16\n"
        "max_model_len: 8192\n"
        "generation:\n"
        "  temperature: 0.2\n"
        "  max_tokens: 64\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model == "Qwen/Qwen3-8B"
    assert config.dtype == "bfloat16"
    assert config.max_model_len == 8192
    assert config.generation.temperature == 0.2
    assert config.generation.max_tokens == 64
    assert config.generation.top_p == 0.9


def test_load_config_reports_unknown_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model: m\nquantization: awq\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="quantization"):
        load_config(path)


def test_load_config_validates_values(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model: m\ngpu_memory_utilization: 2.0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="gpu_memory_utilization"):
        load_config(path)


def test_request_overrides_only_the_values_it_sets():
    defaults = GenerationConfig(temperature=0.7, top_p=0.9, max_tokens=256)

    sampling = GenerationRequest(prompt="hi", temperature=0.0, max_tokens=16).sampling(defaults)

    assert sampling.temperature == 0.0
    assert sampling.max_tokens == 16
    assert sampling.top_p == 0.9


def test_request_without_overrides_returns_the_defaults():
    defaults = GenerationConfig()
    assert GenerationRequest(prompt="hi").sampling(defaults) is defaults


def test_request_overrides_are_validated():
    with pytest.raises(ConfigError, match="top_p"):
        GenerationRequest(prompt="hi", top_p=3).sampling(GenerationConfig())


def test_request_rejects_an_empty_prompt():
    with pytest.raises(ValueError, match="prompt is empty"):
        GenerationRequest(prompt="   ")


def test_response_throughput_counts_output_tokens_only():
    response = GenerationResponse(
        text="hello",
        prompt_tokens=10,
        output_tokens=50,
        elapsed_seconds=2.0,
        finish_reason="stop",
    )
    assert response.output_tokens_per_second == 25.0


def test_response_throughput_is_zero_when_no_time_elapsed():
    response = GenerationResponse(
        text="",
        prompt_tokens=1,
        output_tokens=0,
        elapsed_seconds=0.0,
        finish_reason="length",
    )
    assert response.output_tokens_per_second == 0.0
