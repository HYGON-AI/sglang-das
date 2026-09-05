# Copyright (c) 2026 gencheng liu
# SPDX-License-Identifier: Apache-2.0

"""Runtime-built AITER W4A8 kernels for HCU DeepEP layouts."""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import torch

_SOURCE_DIR = Path(__file__).resolve().parents[2] / "jit" / "csrc" / "moe"
_REQUIRED_AITER_HEADERS = ("moe_w4a8_opt_hip.h", "moe_wna16_utils.h")
_LOAD_LOCK = Lock()

# Match the options used by AITER's module_moe_c_kernel HCU build. The
# extended VGPR option is required by the selected decode/N-loop schedules.
_AITER_HCU_FLAGS = (
    "-DDTK_ENV",
    "-DGPU_ENABLE_FP8",
    "-D__HIP_PLATFORM_HCC__=1",
    "-U__HIP_NO_HALF_CONVERSIONS__",
    "-U__HIP_NO_HALF_OPERATORS__",
    "-Wno-macro-redefined",
    "-Wno-switch-bool",
    "-Wno-undefined-func-template",
    "-Wno-unused-result",
    "-Wno-vla-cxx-extension",
    "-fgpu-flush-denormals-to-zero",
    "-fno-offload-uniform-block",
    "-mllvm",
    "-support-768-vgprs=true",
    "-mllvm",
    "-disable-machine-sink",
    "-mllvm",
    "--amdgpu-kernarg-preload-count=16",
    "-mllvm",
    "--lsr-drop-solution=1",
    "-mllvm",
    "-amdgpu-coerce-illegal-types=1",
    "-mllvm",
    "-amdgpu-early-inline-all=true",
    "-mllvm",
    "-amdgpu-function-calls=false",
    "-mllvm",
    "-enable-post-misched=0",
)


def _has_required_headers(path: Path) -> bool:
    return all((path / header).is_file() for header in _REQUIRED_AITER_HEADERS)


def _find_aiter_moe_sources() -> Path:
    override = os.getenv("AITER_MOE_SRC")
    if override:
        candidate = Path(override).expanduser().resolve()
        if _has_required_headers(candidate):
            return candidate
        raise FileNotFoundError(
            f"AITER_MOE_SRC={candidate} does not contain "
            f"{', '.join(_REQUIRED_AITER_HEADERS)}"
        )

    spec = importlib.util.find_spec("aiter")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            "AITER is required for the HCU W4A8 DeepEP backend. Install the "
            "HYGON AITER package or set AITER_MOE_SRC to its generated MoE sources."
        )
    root = Path(spec.origin).resolve().parent
    candidates = (
        root / "jit" / "build" / "module_moe_c_kernel" / "build" / "srcs",
        root / "jit" / "build" / "module_moe_c_kernel" / "build" / "include",
    )
    for candidate in candidates:
        if _has_required_headers(candidate):
            return candidate
    raise FileNotFoundError(
        "The installed AITER package does not contain generated W4A8 MoE "
        "headers. Build AITER's module_moe_c_kernel once or set AITER_MOE_SRC."
    )


def _target_arch() -> str:
    configured = os.getenv("PYTORCH_ROCM_ARCH")
    if configured:
        arch = configured.split(";")[0].split(":")[0]
    elif torch.cuda.is_available():
        arch = str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0]
    else:
        raise RuntimeError(
            "No HCU device is visible. Set PYTORCH_ROCM_ARCH when cross-compiling."
        )
    if not arch.startswith("gfx"):
        raise RuntimeError(f"Unexpected HCU architecture name: {arch!r}")
    return arch


@lru_cache(maxsize=1)
def _load_extension() -> Any:
    with _LOAD_LOCK:
        from torch.utils.cpp_extension import load

        compiler = Path(
            os.getenv("HCU_EXTENSION_COMPILER", "/opt/dtk/bin/aicc")
        ).expanduser()
        if not compiler.is_file():
            raise FileNotFoundError(
                "The HCU W4A8 kernel requires DTK's AI compiler. Set "
                f"HCU_EXTENSION_COMPILER; not found at {compiler}."
            )
        arch = _target_arch()
        previous_compiler = os.environ.get("PYTORCH_NVCC")
        previous_arch = os.environ.get("PYTORCH_ROCM_ARCH")
        os.environ["PYTORCH_NVCC"] = str(compiler)
        os.environ["PYTORCH_ROCM_ARCH"] = arch
        try:
            return load(
                name=f"sglang_w4a8_deepep_aiter_{arch}",
                sources=[
                    str(_SOURCE_DIR / "w4a8_deepep_binding.cpp"),
                    str(_SOURCE_DIR / "w4a8_deepep_aiter.hip"),
                ],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=[
                    "-O3",
                    "-std=c++17",
                    f"--offload-arch={arch}",
                    *_AITER_HCU_FLAGS,
                ],
                extra_include_paths=[str(_find_aiter_moe_sources())],
                with_cuda=True,
                verbose=os.getenv("SGLANG_W4A8_JIT_VERBOSE", "0") == "1",
            )
        finally:
            if previous_compiler is None:
                os.environ.pop("PYTORCH_NVCC", None)
            else:
                os.environ["PYTORCH_NVCC"] = previous_compiler
            if previous_arch is None:
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
            else:
                os.environ["PYTORCH_ROCM_ARCH"] = previous_arch


def preload_w4a8_deepep_aiter() -> None:
    """Compile and load the extension before CUDA graph capture."""
    _load_extension()


def w4a8_mmac_contiguous_out(*args: Any, **kwargs: Any) -> None:
    _load_extension().w4a8_mmac_contiguous_out(*args, **kwargs)


def w4a8_mmac_masked_out(*args: Any, **kwargs: Any) -> None:
    _load_extension().w4a8_mmac_masked_out(*args, **kwargs)
