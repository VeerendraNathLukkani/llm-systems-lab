"""System inspection tests, driven by a fake torch module.

The dev machine may have no CUDA at all, so torch is replaced in sys.modules rather
than probed. system.py imports torch inside its functions, which makes that work.
"""

import builtins
import sys
import types

import pytest

from llm_lab.hardware import system
from llm_lab.hardware.system import (
    collect_system_info,
    cuda_device_count,
    empty_cuda_cache,
    format_system_info,
    gpu_memory_used_gb,
)

GB = 1024**3


def block_torch_import(monkeypatch, error):
    """Make `import torch` raise `error`, leaving every other import alone."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)


def fake_torch(*, cuda_available=True, devices=1, free=10 * GB, total=48 * GB):
    module = types.ModuleType("torch")
    module.version = types.SimpleNamespace(cuda="12.8")
    properties = types.SimpleNamespace(
        name="NVIDIA L40S", total_memory=48 * GB, major=8, minor=9
    )
    module.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        device_count=lambda: devices,
        get_device_properties=lambda index: properties,
        mem_get_info=lambda index: (free, total),
        empty_cache=lambda: module.cuda.__dict__.__setitem__("emptied", True),
    )
    return module


@pytest.fixture
def torch_with_gpu(monkeypatch):
    module = fake_torch()
    monkeypatch.setitem(sys.modules, "torch", module)
    monkeypatch.setattr(
        system, "_installed_version", lambda package: {"torch": "2.7.0", "vllm": "0.10.1"}[package]
    )
    return module


def test_collect_system_info_reports_the_visible_gpu(torch_with_gpu):
    info = collect_system_info()

    assert info.torch_version == "2.7.0"
    assert info.vllm_version == "0.10.1"
    assert info.cuda_available is True
    assert info.cuda_version == "12.8"
    assert len(info.gpus) == 1
    assert info.gpus[0].name == "NVIDIA L40S"
    assert info.gpus[0].total_memory_gb == pytest.approx(48.0)
    assert info.gpus[0].compute_capability == "8.9"


def test_collect_system_info_without_cuda_lists_no_gpus(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_available=False))
    monkeypatch.setattr(system, "_installed_version", lambda package: "1.0")

    info = collect_system_info()

    assert info.cuda_available is False
    assert info.gpus == []


def test_collect_system_info_without_torch_still_reports_the_host(monkeypatch):
    monkeypatch.setattr(system, "_installed_version", lambda package: None)

    info = collect_system_info()

    assert info.torch_version is None
    assert info.vllm_version is None
    assert info.cuda_available is False
    assert info.cuda_version is None
    assert info.python_version


def test_cuda_device_count_is_zero_without_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_available=False, devices=4))
    assert cuda_device_count() == 0


def test_cuda_device_count_is_zero_without_torch(monkeypatch):
    block_torch_import(monkeypatch, ModuleNotFoundError("No module named 'torch'"))
    assert cuda_device_count() == 0


def test_cuda_device_count_propagates_a_broken_torch(monkeypatch):
    block_torch_import(monkeypatch, ImportError("libnvshmem_host.so.3: cannot open shared object file"))

    # A torch that will not import is a broken environment, not a machine with no GPU.
    with pytest.raises(ImportError, match="libnvshmem_host"):
        cuda_device_count()


def test_cuda_device_count_counts_visible_devices(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(devices=2))
    assert cuda_device_count() == 2


def fake_pynvml(used=38 * GB, error=None):
    module = types.ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    module.NVMLError = NVMLError
    module.shutdowns = 0

    def init():
        if error:
            raise NVMLError(error)

    module.nvmlInit = init
    module.nvmlShutdown = lambda: module.__dict__.__setitem__("shutdowns", module.shutdowns + 1)
    module.nvmlDeviceGetHandleByIndex = lambda index: index
    module.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(used=used)
    return module


def test_gpu_memory_used_comes_from_nvml(monkeypatch):
    module = fake_pynvml(used=38 * GB)
    monkeypatch.setitem(sys.modules, "pynvml", module)

    assert gpu_memory_used_gb() == pytest.approx(38.0)
    assert module.shutdowns == 1


def test_gpu_memory_used_is_none_when_nvml_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml(error="driver not loaded"))
    assert gpu_memory_used_gb() is None


def test_empty_cuda_cache_without_torch_does_nothing(monkeypatch):
    block_torch_import(monkeypatch, ModuleNotFoundError("No module named 'torch'"))
    empty_cuda_cache()


def test_collect_system_info_reports_a_torch_that_will_not_import(monkeypatch):
    monkeypatch.setattr(system, "_installed_version", lambda package: "2.9.0")
    block_torch_import(monkeypatch, ImportError("libnvshmem_host.so.3: cannot open shared object file"))

    info = collect_system_info()

    assert info.torch_version == "2.9.0"
    assert info.cuda_available is False
    assert "libnvshmem_host" in info.torch_error
    assert "torch import     FAILED" in format_system_info(info)


def test_format_system_info_names_missing_pieces(monkeypatch):
    monkeypatch.setattr(system, "_installed_version", lambda package: None)

    text = format_system_info(collect_system_info())

    assert "torch            not installed" in text
    assert "vllm             not installed" in text
    assert "none visible" in text


def test_format_system_info_lists_each_gpu(torch_with_gpu):
    text = format_system_info(collect_system_info())

    assert "gpu 0" in text
    assert "NVIDIA L40S" in text
    assert "sm_89" in text
