# LLM Systems Lab

A reproducible lab for serving and benchmarking LLMs. **v0.1 covers vLLM inference only.**

## What v0.1 does

- Validates a vLLM configuration (model, dtype, max model length, GPU memory
  utilization, tensor parallel size, sampling defaults) before anything is loaded.
- Reports what decides whether a run will work: Python, PyTorch and vLLM versions, CUDA
  availability and version, GPU name, VRAM and compute capability.
- Loads a Hugging Face model with vLLM, runs one prompt, and returns the text with
  prompt/output token counts, latency and output tokens/sec.
- Benchmarks repeated generations: load time, mean latency, mean throughput, GPU memory.
- Translates the common load failures — no CUDA device, unknown model id, gated repo,
  OOM, impossible tensor-parallel split — into messages naming the setting to change,
  with the original exception chained.

## Requirements

- Linux for inference — vLLM publishes no Windows or macOS wheels; WSL2 counts
- Python 3.11+
- One NVIDIA GPU sized for the model, with a working CUDA driver
- vLLM 0.10+ and a CUDA build of PyTorch (both installed as dependencies)

## Installation

With [uv](https://docs.astral.sh/uv/) — creates `.venv`, resolves everything into
`uv.lock` (commit it to pin the environment) and installs the project editable:

```bash
uv sync --extra dev
uv run llmlab info    # run commands without activating the venv
```

With pip instead:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`vllm` is declared for Linux only, so on Windows or macOS this installs everything
except the engine — enough for `llmlab info` and the tests, but not for inference. On
Windows, torch is pulled from PyTorch's cu130 index rather than PyPI (which serves a
CPU-only wheel there), so `llmlab info` still reports the GPU.

vLLM pins its own PyTorch build — install into a clean environment rather than on top of
an existing torch. For gated models, `export HF_TOKEN=...` first.

On WSL2, put the venv on the Linux filesystem (`export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/llm-lab`)
— installing into a `/mnt/c` path silently truncates the large CUDA wheels — and export two
vLLM settings, because pinned memory is opt-in there and FlashInfer's sampler wants an nvcc
that the pip wheels do not ship:

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
```

## `llmlab info`

```bash
llmlab info
```

```text
python           3.11.9
torch            2.7.0
vllm             0.10.1
cuda available   True
cuda (torch)     12.8
gpu 0            NVIDIA L40S, 44.4 GiB, sm_89
```

## Inference

```bash
llmlab run --model Qwen/Qwen3-8B --prompt "Explain paged attention briefly."
```

Completion goes to stdout, model info and the token/latency line to stderr, so
`llmlab run ... > out.txt` keeps just the text. Overridable flags: `--dtype`,
`--max-model-len`, `--gpu-memory-utilization`, `--tensor-parallel-size`, `--seed`,
`--temperature`, `--top-p`, `--max-tokens`, `--trust-remote-code`.

Settings can live in a YAML file instead — the same keys, with the sampling values
nested under `generation:` — and flags override it:

```bash
llmlab run --config qwen3-8b.yaml --prompt "Explain paged attention briefly."
```

From Python:

```python
from llm_lab import EngineConfig, GenerationRequest, VllmEngine

with VllmEngine(EngineConfig(model="Qwen/Qwen3-8B")) as engine:
    response = engine.generate(GenerationRequest("Explain paged attention briefly."))
    print(response.text, response.output_tokens_per_second)
```

## Benchmark

```bash
llmlab benchmark --model Qwen/Qwen3-8B --prompt "Explain paged attention briefly." \
  --runs 5 --warmup 1
```

Prints model load time, prompt tokens, mean output tokens, mean latency, mean output
tokens/sec and GPU memory in use.

## Decision-model experiments

`src/llm_lab/inference/decision/` holds a separate line of experiments on System-One
decision models (`laya`) -- support ticket triage and an agent action guard. It shares
this repo but not the vLLM engine above, and is not covered by the test suite. See
[its README](src/llm_lab/inference/decision/README.md) for the scripts and findings.

## Tests

```bash
uv run pytest
```

No weights are downloaded and no GPU is needed: the tests inject fakes for `vllm` and
`torch`. GPU tests, if added, are marked `gpu` and excluded with `-m "not gpu"`.

## Limitations

- One model per process; reloading a different model in-process is not supported.
  `llmlab info` is the only command that works without a GPU.
- Sequential, one prompt per call — this measures single-request latency, not vLLM's
  continuous-batching throughput under concurrency.
- Multi-GPU only via `tensor_parallel_size`; no pipeline parallelism, no multi-node.
- With `temperature > 0` completion length varies, so benchmark numbers vary. Use
  `--temperature 0` and `--seed` for comparable runs.
- GPU memory is read from NVML for the whole device, so other processes count. It is a
  point-in-time reading taken after the timed runs, not a peak.
- Verified end to end on one machine only: Qwen3-0.6B on an RTX 3050 6 GB under WSL2,
  vLLM 0.27.1 / torch 2.13 / CUDA 13.0. Other models and GPUs are untested.
