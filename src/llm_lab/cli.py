"""Command line entry point: `llmlab info`, `llmlab run`, `llmlab benchmark`."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .benchmark.runner import format_benchmark, run_benchmark
from .core.config import DTYPES, ConfigError, EngineConfig, load_config
from .core.request import GenerationRequest
from .hardware.system import collect_system_info, format_system_info
from .inference.vllm.engine import EngineError, VllmEngine


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, dispatch, and turn known failures into an exit code of 1."""
    parser = argparse.ArgumentParser(prog="llmlab", description="vLLM inference lab")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("info", help="show Python, torch, vLLM and GPU information")

    run = commands.add_parser("run", help="load a model and generate once")
    _add_engine_arguments(run)
    run.add_argument("--prompt", required=True)

    benchmark = commands.add_parser("benchmark", help="measure load time and generation throughput")
    _add_engine_arguments(benchmark)
    benchmark.add_argument("--prompt", required=True)
    benchmark.add_argument("--runs", type=int, default=3)
    benchmark.add_argument("--warmup", type=int, default=1)

    args = parser.parse_args(argv)

    if args.command == "info":
        print(format_system_info(collect_system_info()))
        return 0

    try:
        config = _build_config(args)
        if args.command == "run":
            return _run(config, args.prompt)
        return _benchmark(config, args.prompt, args.runs, args.warmup)
    except (ConfigError, EngineError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        cause = exc.__cause__
        if cause is not None:
            print(f"caused by: {type(cause).__name__}: {cause}", file=sys.stderr)
        return 1


def _add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the engine and sampling flags shared by `run` and `benchmark`."""
    parser.add_argument("--config", help="YAML config file; other flags override it")
    parser.add_argument("--model", help="Hugging Face model id or local path")
    parser.add_argument("--dtype", choices=sorted(DTYPES))
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)


def _build_config(args: argparse.Namespace) -> EngineConfig:
    """Build the engine config from a YAML file, explicit flags, or both."""
    if args.config:
        config = load_config(args.config)
    elif args.model:
        config = EngineConfig(model=args.model)
    else:
        raise ConfigError("pass --model or --config")

    generation = replace(
        config.generation, **_provided(args, "temperature", "top_p", "max_tokens")
    )
    return replace(
        config,
        generation=generation,
        **_provided(
            args,
            "model",
            "dtype",
            "max_model_len",
            "gpu_memory_utilization",
            "tensor_parallel_size",
            "trust_remote_code",
            "seed",
        ),
    )


def _provided(args: argparse.Namespace, *names: str) -> dict[str, object]:
    """CLI flags the user actually passed. Everything else defaults to None."""
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def _announce_load(config: EngineConfig) -> None:
    """Break the long silence of vLLM import, weight download and engine startup."""
    print(f"loading {config.model} with vLLM ...", file=sys.stderr, flush=True)


def _run(config: EngineConfig, prompt: str) -> int:
    """Generate once. Completion goes to stdout, diagnostics to stderr."""
    _announce_load(config)
    with VllmEngine(config) as engine:
        for key, value in engine.info().items():
            print(f"{key}: {value}", file=sys.stderr)
        response = engine.generate(GenerationRequest(prompt))

    print(response.text)
    print(
        f"\n[{response.prompt_tokens} prompt tokens, {response.output_tokens} output tokens, "
        f"{response.elapsed_seconds:.2f} s, {response.output_tokens_per_second:.1f} tok/s, "
        f"finish_reason={response.finish_reason}]",
        file=sys.stderr,
    )
    return 0


def _benchmark(config: EngineConfig, prompt: str, runs: int, warmup: int) -> int:
    """Load the model, time repeated generations, and print the summary."""
    _announce_load(config)
    print(format_benchmark(run_benchmark(config, prompt, runs=runs, warmup=warmup)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
