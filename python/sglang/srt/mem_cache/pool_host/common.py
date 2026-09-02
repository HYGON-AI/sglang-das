from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

import torch

from sglang.srt.mem_cache.storage.mmap import alloc_mmap
from sglang.srt.utils import is_hcu

logger = logging.getLogger(__name__)

_is_hcu = is_hcu()
_HIP_RUNTIME = None


class HostTensorAllocator:
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        return alloc_mmap(dims, dtype)


class ShmHostTensorAllocator(HostTensorAllocator):
    def __init__(self):
        super().__init__()
        self.fds = []
        self.mms = []

    @property
    def fd(self):
        return self.fds[0] if self.fds else None

    @property
    def mm(self):
        return self.mms[0] if self.mms else None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        from sglang.srt.mem_cache.storage.mmap import alloc_shm

        tensor, fd, mm = alloc_shm(dims, dtype)
        self.fds.append(fd)
        self.mms.append(mm)
        return tensor

    def __del__(self):
        for fd in getattr(self, "fds", []):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fds = []


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    elif allocator_type == "mori":
        try:
            from sglang.srt.mem_cache.storage.umbp.umbp_host_allocator import (
                UMBPHostTensorAllocator,
            )

            return UMBPHostTensorAllocator()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "UMBPHostTensorAllocator unavailable (%s). "
                "Falling back to torch.empty-based allocator.",
                exc,
            )
            return HostTensorAllocator()
    elif allocator_type == "shm":
        return ShmHostTensorAllocator()
    else:
        return HostTensorAllocator()


def get_allocator_type(server_args) -> str:
    backend = getattr(server_args, "hicache_storage_backend", None)
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = getattr(
            server_args, "hicache_storage_backend_extra_config", None
        )
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


def _cuda_host_register(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    n_bytes = buffer.numel() * buffer.element_size()
    # HCU transfer kernels dereference host memory from the device. Register
    # those pages as mapped so hipHostGetDevicePointer can translate them.
    flags = 2 if _is_hcu else 0
    rc = cudart.cudaHostRegister(buffer.data_ptr(), n_bytes, flags)
    if int(rc) != 0:
        raise RuntimeError(
            f"cudaHostRegister failed (rc={int(rc)}, "
            f"{cudart.cudaGetErrorString(rc)}) for ptr={buffer.data_ptr():#x} "
            f"size={n_bytes}; host buffer is not pinned and device transfers "
            f"may silently return stale data."
        )


def _hip_host_get_device_pointer(host_ptr: int) -> int:
    """Translate a mapped HCU host pointer into the device address space."""

    import ctypes

    global _HIP_RUNTIME
    if _HIP_RUNTIME is None:
        last_error = None
        for library in ("libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"):
            try:
                _HIP_RUNTIME = ctypes.CDLL(library)
                break
            except OSError as error:  # noqa: PERF203
                last_error = error
        if _HIP_RUNTIME is None:
            raise RuntimeError(
                "Failed to load the HIP runtime for hipHostGetDevicePointer: "
                f"{last_error}"
            )

    device_ptr = ctypes.c_void_p()
    rc = _HIP_RUNTIME.hipHostGetDevicePointer(
        ctypes.byref(device_ptr), ctypes.c_void_p(host_ptr), ctypes.c_uint(0)
    )
    if int(rc) != 0 or device_ptr.value is None:
        raise RuntimeError(
            "hipHostGetDevicePointer failed "
            f"(rc={int(rc)}) for mapped host ptr={host_ptr:#x}."
        )
    return int(device_ptr.value)


def kernel_accessible_host_ptr(tensor: torch.Tensor) -> int:
    """Return the address a GPU transfer kernel can dereference."""

    if not _is_hcu or tensor.is_cuda:
        return tensor.data_ptr()
    return _hip_host_get_device_pointer(tensor.data_ptr())


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    rc = cudart.cudaHostUnregister(buffer.data_ptr())
    if int(rc) != 0:
        # Best-effort on shutdown: warn, don't raise -- a leak is reclaimed at exit.
        logger.warning(
            "cudaHostUnregister failed (rc=%d, %s) for ptr=%#x",
            int(rc),
            cudart.cudaGetErrorString(rc),
            buffer.data_ptr(),
        )


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        _cuda_host_register(buffer)
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)
