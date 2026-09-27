"""Thin wrapper around vLLM's offline ``LLM`` API."""

from __future__ import annotations

import gc
import time
from typing import Any

from ...core.config import EngineConfig
from ...core.request import GenerationRequest
from ...core.response import GenerationResponse
from ...hardware.system import cuda_device_count, empty_cuda_cache


class EngineError(RuntimeError):
    """Base class for failures raised by the vLLM engine wrapper."""


class EngineLoadError(EngineError):
    """Model initialization failed; the message says what to change."""


class GenerationError(EngineError):
    """vLLM raised while generating for an already-loaded model."""


class VllmEngine:
    """One vLLM `LLM` instance and the config it was built from.

    Deliberately thin: every vLLM argument passed below comes straight from
    EngineConfig, so what the engine is doing stays readable.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.llm: Any = None
        self.load_seconds: float | None = None

    def load(self) -> None:
        """Check the GPU preconditions, then build the vLLM engine and time it."""
        if self.llm is not None:
            raise EngineError("engine is already loaded; create a new VllmEngine for a different model")

        try:
            devices = cuda_device_count()
        except ImportError as exc:
            raise EngineLoadError(
                "PyTorch is installed but will not import, so vLLM cannot start. A missing "
                "CUDA shared library usually means a truncated or mismatched install; "
                "rebuild the environment. Run `llmlab info` for the full import error."
            ) from exc

        if devices == 0:
            raise EngineError(
                "no CUDA device is visible to PyTorch, and vLLM needs one. "
                "Check that torch is installed with CUDA support, that `nvidia-smi` "
                "works, and that CUDA_VISIBLE_DEVICES is not empty."
            )
        if self.config.tensor_parallel_size > devices:
            raise EngineError(
                f"tensor_parallel_size={self.config.tensor_parallel_size} but only "
                f"{devices} CUDA device(s) are visible"
            )

        # Imported here so that `llmlab info` and the config code do not pay for vLLM's
        # multi-second import, and so the package is usable on machines without it.
        from vllm import LLM

        arguments = {
            "model": self.config.model,
            "dtype": self.config.dtype,
            "max_model_len": self.config.max_model_len,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "trust_remote_code": self.config.trust_remote_code,
        }
        # vLLM validates seed as an int and rejects an explicit None, so an unset seed
        # means omitting the argument and taking vLLM's own default.
        if self.config.seed is not None:
            arguments["seed"] = self.config.seed

        started = time.perf_counter()
        try:
            self.llm = LLM(**arguments)
        except Exception as exc:
            raise EngineLoadError(_load_failure_message(self.config, exc)) from exc
        self.load_seconds = time.perf_counter() - started

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Run one prompt to completion and return it with its token counts and timing."""
        if self.llm is None:
            raise EngineError("engine is not loaded; call load() first")

        from vllm import SamplingParams

        sampling = request.sampling(self.config.generation)
        params = SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
        )

        started = time.perf_counter()
        try:
            outputs = self.llm.generate([request.prompt], params)
        except Exception as exc:
            raise GenerationError(
                f"vLLM failed to generate with {self.config.model!r}: {exc}"
            ) from exc
        elapsed = time.perf_counter() - started

        output = outputs[0]
        completion = output.outputs[0]
        return GenerationResponse(
            text=completion.text,
            prompt_tokens=len(output.prompt_token_ids),
            output_tokens=len(completion.token_ids),
            elapsed_seconds=elapsed,
            finish_reason=completion.finish_reason,
        )

    def info(self) -> dict[str, Any]:
        """Report the settings vLLM actually resolved, not just the ones requested."""
        if self.llm is None:
            raise EngineError("engine is not loaded; call load() first")
        model_config = self.llm.llm_engine.model_config
        return {
            "model": self.config.model,
            "dtype": str(model_config.dtype),
            "max_model_len": model_config.max_model_len,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "load_seconds": self.load_seconds,
        }

    def close(self) -> None:
        """Drop the engine. vLLM releases VRAM when the LLM object is collected."""
        if self.llm is None:
            return
        self.llm = None
        gc.collect()
        empty_cuda_cache()

    def __enter__(self) -> VllmEngine:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _load_failure_message(config: EngineConfig, exc: Exception) -> str:
    """Turn vLLM's initialization errors into a message naming the setting to change."""
    text = str(exc).lower()
    model = config.model

    if "out of memory" in text or "no available memory" in text or "kv cache" in text:
        return (
            f"not enough GPU memory to serve {model!r} with "
            f"max_model_len={config.max_model_len} and "
            f"gpu_memory_utilization={config.gpu_memory_utilization}. "
            "Lower max_model_len, raise gpu_memory_utilization if other processes are not "
            "using the GPU, or pick a smaller model."
        )
    if "401" in text or "gated" in text or "authorized" in text:
        return (
            f"{model!r} is gated or private on Hugging Face. Accept the model terms and "
            "export HF_TOKEN before loading."
        )
    if "404" in text or "not a local folder" in text or "repositorynotfound" in text:
        return (
            f"{model!r} is not a Hugging Face repo id or a local model directory. "
            "Check the spelling, including the org prefix."
        )
    if "divisible" in text or "attention heads" in text:
        return (
            f"{model!r} cannot be split across tensor_parallel_size="
            f"{config.tensor_parallel_size}: its attention heads do not divide evenly. "
            "Use a tensor parallel size that divides the model's head count."
        )
    if "trust_remote_code" in text:
        return (
            f"{model!r} ships custom modelling code. Pass trust_remote_code=True "
            "(--trust-remote-code) if you trust the repository."
        )
    if "not supported" in text or "unsupported" in text:
        return f"vLLM does not support this configuration for {model!r}: {exc}"
    return f"vLLM failed to initialise {model!r}: {exc}"
