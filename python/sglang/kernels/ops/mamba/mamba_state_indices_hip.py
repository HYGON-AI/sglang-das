"""HIP C++ replay metadata kernel. Build/load/warm up BEFORE serving replay.

prepare_replay_state_indices() owns compilation and first module launch.
The hot function never invokes a compiler, copies data to CPU or synchronizes.
The Triton implementation remains available through the serving dispatcher.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import torch


@functools.cache
def _build_and_load(arch: str):
    import fcntl

    source = Path(__file__).with_name("csrc") / "mamba_state_indices_hip.cu"
    compiler = os.environ.get("HIPCC") or shutil.which("hipcc")
    if not compiler:
        raise RuntimeError("HIP compiler not found; set HIPCC")
    # Runtime/toolkit version and resolved compiler separate incompatible caches.
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC", "--offload-arch=" + arch]
    key = hashlib.sha256(
        source.read_bytes()
        + repr((arch, torch.version.hip, str(Path(compiler).resolve()), flags)).encode()
    ).hexdigest()[:20]
    root = Path(
        os.environ.get(
            "SGLANG_REPLAY_INDICES_HIP_CACHE",
            str(Path.home() / ".cache/sglang/replay_indices_hip"),
        )
    )
    build = root / key
    build.mkdir(parents=True, exist_ok=True)
    binary = build / "mamba_state_indices_hip.so"
    with (build / "build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.is_file():
            temporary = build / ("module." + str(os.getpid()) + ".so")
            command = [compiler, *flags, str(source), "-o", str(temporary)]
            result = subprocess.run(command, capture_output=True, text=True)
            (build / "build.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(
                    "Replay indices HIP build failed: " + result.stderr[-4000:]
                )
            temporary.replace(binary)
    module = ctypes.CDLL(str(binary))
    module.sgl_fused_replay_state_indices.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    module.sgl_fused_replay_state_indices.restype = ctypes.c_int
    module.sgl_replay_state_indices_error.argtypes = [ctypes.c_int]
    module.sgl_replay_state_indices_error.restype = ctypes.c_char_p
    return module


_modules: dict[torch.device, ctypes.CDLL] = {}


def prepare_replay_state_indices(device) -> None:
    """Compile, load and warm the generic kernel outside graph capture/replay."""
    device = torch.device(device)
    with torch.cuda.device(device):
        device = torch.device("cuda", torch.cuda.current_device())
        if device in _modules:
            return
        if torch.version.hip is None:
            raise RuntimeError("Replay indices HIP requires a HIP runtime")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare replay indices HIP before graph capture")
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        module = _build_and_load(arch)
        req = torch.zeros(1, device=device, dtype=torch.int64)
        mapping = torch.zeros(1, device=device, dtype=torch.int32)
        out = torch.empty(1, device=device, dtype=torch.int32)
        error = module.sgl_fused_replay_state_indices(
            req.data_ptr(),
            mapping.data_ptr(),
            out.data_ptr(),
            1,
            1,
            torch.cuda.current_stream(device).cuda_stream,
        )
        if error:
            raise RuntimeError(module.sgl_replay_state_indices_error(error).decode())
        # Only this explicit startup warmup synchronizes, never the hot path.
        torch.cuda.current_stream(device).synchronize()
        _modules[device] = module


def fused_replay_state_indices_hip(
    *,
    req_pool_indices: torch.Tensor,
    mamba_index_mapping: torch.Tensor,
    out_state_indices: torch.Tensor,
    valid_bs: int,
    total_bs: int,
    v2p: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather valid slots; zero padded req IDs and fill padded slots with -1.

    Same preconditions as Triton: static identity v2p pool, valid request IDs.
    Metadata validation below never reads GPU values. Buffers cannot alias.
    """
    if v2p is not None:
        # The current main pool can remap live virtual slots. This HIP kernel
        # only implements the identity case; retain main's fused translation.
        from .mamba_state_indices_triton import fused_replay_state_indices

        return fused_replay_state_indices(
            req_pool_indices=req_pool_indices,
            mamba_index_mapping=mamba_index_mapping,
            out_state_indices=out_state_indices,
            valid_bs=valid_bs,
            total_bs=total_bs,
            v2p=v2p,
        )
    tensors = (req_pool_indices, mamba_index_mapping, out_state_indices)
    device = req_pool_indices.device
    if not (0 <= valid_bs <= total_bs <= 2147483647):
        raise ValueError("Require 0 <= valid_bs <= total_bs <= INT_MAX")
    if any(
        not t.is_cuda or t.device != device or t.ndim != 1 or not t.is_contiguous()
        for t in tensors
    ):
        raise ValueError(
            "Replay indices require contiguous 1D tensors on one HIP device"
        )
    if tuple(t.dtype for t in tensors) != (torch.int64, torch.int32, torch.int32):
        raise ValueError("Replay indices require req=int64, mapping=int32, out=int32")
    if req_pool_indices.numel() < total_bs or out_state_indices.numel() < total_bs:
        raise ValueError("Replay indices buffers are shorter than total_bs")
    module = _modules.get(device)
    if module is None:
        raise RuntimeError("Call prepare_replay_state_indices(device) before replay")
    with torch.cuda.device(device):
        error = module.sgl_fused_replay_state_indices(
            req_pool_indices.data_ptr(),
            mamba_index_mapping.data_ptr(),
            out_state_indices.data_ptr(),
            valid_bs,
            total_bs,
            torch.cuda.current_stream(device).cuda_stream,
        )
    if error:
        raise RuntimeError(
            "Replay indices HIP launch failed: "
            + module.sgl_replay_state_indices_error(error).decode()
        )
    return out_state_indices[:total_bs]
