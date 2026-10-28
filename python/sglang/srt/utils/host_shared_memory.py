import ctypes
import functools
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory

import torch

from sglang.srt.distributed.naive_distributed import get_naive_distributed
from sglang.srt.utils import check_cuda_result, is_hcu

logger = logging.getLogger(__name__)

_HIP_HOST_MALLOC_NUMA_USER = 0x20000000
_MPOL_INTERLEAVE = 3


class HostSharedMemoryManager:
    """Shared host tensors accessible to GPUs in a local process group.

    ``group`` accepts a GroupCoordinator; omitting it preserves the existing
    NaiveDistributed initialization path. ``use_hsa_ipc`` opts into HCU's
    single exported allocation instead of per-process host registration.
    Keep the manager alive while its tensors are in use.
    """

    def __init__(self, base_name: str, *, group=None, use_hsa_ipc: bool = False):
        if use_hsa_ipc and not is_hcu():
            raise ValueError("HSA IPC host allocations require HCU")
        self._base_name = base_name
        self._group = group
        self._use_hsa_ipc = use_hsa_ipc
        self._operation_index = 0
        self._records: list[_Record] = []

    def malloc(self, *, shape, dtype):
        meta_tensor = torch.empty(size=shape, dtype=dtype, device="meta")
        raw = self._malloc_raw(num_bytes=meta_tensor.nbytes)
        return raw.view(dtype).view(*shape)

    def _malloc_raw(self, *, num_bytes: int) -> torch.Tensor:
        group = self._group if self._group is not None else get_naive_distributed()
        rank = group.rank_in_group if self._group is not None else group.get_rank()
        if self._use_hsa_ipc:
            allocation = _HcuSharedAllocation(num_bytes, group, rank)
            self._records.append(
                _Record(allocation.bytes, allocation.device_ptr, allocation)
            )
            return allocation.bytes

        import cuda.bindings.runtime as cuda_rt

        self._operation_index += 1
        shm_name = f"{self._base_name}_op{self._operation_index}"

        # TODO handle dispose
        if rank == 0:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=num_bytes)

        group.barrier()

        if rank != 0:
            shm = shared_memory.SharedMemory(name=shm_name)

        tensor = torch.frombuffer(shm.buf, dtype=torch.uint8, count=num_bytes)

        check_cuda_result(
            cuda_rt.cudaHostRegister(
                tensor.data_ptr(), num_bytes, cuda_rt.cudaHostRegisterPortable
            )
        )

        group.barrier()

        self._records.append(_Record(tensor, tensor.data_ptr(), shm))
        return tensor

    def device_pointer(self, tensor: torch.Tensor) -> int:
        """GPU address of a contiguous tensor allocated by this manager.

        Imported HSA storage is not registered with HIP's host-pointer tracker;
        retain its IPC device address instead of registering the storage again.
        """
        if not tensor.is_contiguous():
            raise ValueError("Shared host tensor must be contiguous")
        ptr = tensor.data_ptr()
        for record in self._records:
            offset = ptr - record.tensor.data_ptr()
            if 0 <= offset and offset + tensor.nbytes <= record.tensor.nbytes:
                return record.device_ptr + offset
        raise ValueError("Tensor does not belong to this shared memory manager")


@dataclass
class _Record:
    tensor: torch.Tensor
    device_ptr: int
    owner: object


# Can have multi instances if needed
_instance: HostSharedMemoryManager | None = None


def get_host_shared_memory_manager():
    assert _instance is not None
    return _instance


def set_host_shared_memory_manager(instance: HostSharedMemoryManager):
    global _instance
    assert _instance is None
    _instance = instance


class _HsaIpcHandle(ctypes.Structure):
    # HCU's DTK exposes the hsa_amd_* ABI too; these are runtime symbol names.
    # hsa_amd_ipc_memory_t is a process-independent 256-bit allocation handle.
    _fields_ = [("words", ctypes.c_uint32 * 8)]


@functools.cache
def _hcu_runtime():
    from sglang.srt.distributed.device_communicators.cuda_wrapper import (
        get_loaded_hip_runtime,
    )

    torch.cuda.init()
    # Reuse HCU's loaded HIP runtime and HSA ABI SONAME, avoiding AITER's
    # separate private HSA library. DTK retains the hip*/hsa_amd_* symbols.
    try:
        hip = get_loaded_hip_runtime()
        hsa = ctypes.CDLL("libhsa-runtime64.so.1", mode=os.RTLD_NOLOAD)
    except OSError as exc:
        raise RuntimeError(
            "The active DTK HIP/HSA runtimes are required for HCU shared host allocations"
        ) from exc

    def bind(fn, *args, result=ctypes.c_int):
        fn.argtypes, fn.restype = args, result

    ptr = ctypes.c_void_p
    out_ptr = ctypes.POINTER(ptr)
    handle_ptr = ctypes.POINTER(_HsaIpcHandle)
    bind(hip.hipHostMalloc, out_ptr, ctypes.c_size_t, ctypes.c_uint)
    bind(hip.hipHostFree, ptr)
    bind(hip.hipHostGetDevicePointer, out_ptr, ptr, ctypes.c_uint)
    bind(hip.hipGetErrorName, ctypes.c_int, result=ctypes.c_char_p)
    bind(hip.hipGetErrorString, ctypes.c_int, result=ctypes.c_char_p)
    bind(hsa.hsa_amd_ipc_memory_create, ptr, ctypes.c_size_t, handle_ptr)
    bind(
        hsa.hsa_amd_ipc_memory_attach,
        handle_ptr,
        ctypes.c_size_t,
        ctypes.c_uint,
        ptr,
        out_ptr,
    )
    bind(hsa.hsa_amd_ipc_memory_detach, ptr)
    bind(hsa.hsa_status_string, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))
    return hip, hsa


def _check_hip(lib, code, operation):
    if code:
        name, detail = lib.hipGetErrorName(code), lib.hipGetErrorString(code)
        raise RuntimeError(
            f"{operation}: HIP code={code}, "
            f"{name.decode() if name else 'unknown'}: "
            f"{detail.decode() if detail else 'unknown'}"
        )


def _check_hsa(lib, code, operation):
    if code:
        detail = ctypes.c_char_p()
        lib.hsa_status_string(code, ctypes.byref(detail))
        raise RuntimeError(
            f"{operation}: HSA code={code}, "
            f"{detail.value.decode() if detail.value else 'unknown'}"
        )


class _NumaPolicy:
    """Save/restore this thread's policy; never change system-wide NUMA settings."""

    def __init__(self):
        from sglang.srt.utils.numa_utils import get_libnuma

        self.lib = get_libnuma(use_errno=True)
        if self.lib is None:
            raise RuntimeError("libnuma is required for HCU shared host allocations")
        self.lib.numa_max_possible_node.restype = ctypes.c_int
        self.count = self.lib.numa_max_possible_node() + 1
        if self.count <= 0:
            raise RuntimeError("Cannot determine NUMA node mask size")
        self.bits = ctypes.sizeof(ctypes.c_ulong) * 8
        self.mask_type = ctypes.c_ulong * ((self.count + self.bits - 1) // self.bits)
        self.mode, self.mask = ctypes.c_int(), self.mask_type()
        self.lib.get_mempolicy.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        self.lib.set_mempolicy.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
        ]
        self._check(
            self.lib.get_mempolicy(
                ctypes.byref(self.mode), self.mask, self.count, None, 0
            ),
            "get_mempolicy",
        )
        self.nodes = [
            node
            for node in range(self.count)
            if self.mask[node // self.bits] & (1 << (node % self.bits))
        ]
        if not self.nodes:
            # MPOL_DEFAULT/LOCAL has no explicit mask. Respect the cpuset.
            with open("/proc/self/status") as f:
                allowed = next(
                    line.split(":", 1)[1].strip()
                    for line in f
                    if line.startswith("Mems_allowed_list:")
                )
            for item in allowed.split(","):
                bounds = item.split("-")
                self.nodes.extend(range(int(bounds[0]), int(bounds[-1]) + 1))

    @staticmethod
    def _check(rc, operation):
        if rc:
            err = ctypes.get_errno()
            raise OSError(err, f"{operation}: {os.strerror(err)}")

    @contextmanager
    def interleave(self, nodes):
        mask = self.mask_type()
        for node in nodes:
            mask[node // self.bits] |= 1 << (node % self.bits)
        self._check(
            self.lib.set_mempolicy(_MPOL_INTERLEAVE, mask, self.count),
            "set NUMA interleave",
        )
        try:
            yield
        finally:
            self._check(
                self.lib.set_mempolicy(self.mode.value, self.mask, self.count),
                "restore NUMA policy",
            )


class _HcuSharedAllocation:
    """One pinned host allocation, imported into every local group process.

    HSA_USERPTR_FOR_PAGED_MEM=0 must be set before runtime initialization:
    ordinary USERPTR allocations cannot be exported through HSA IPC. Importing
    one allocation avoids charging the full table against KFD's system-memory
    budget once per rank. Kernels still read the complete table directly.

    Retain this object for the entire lifetime of all tensor views and GPU
    reads. close() is collective and is only safe after all reads have finished.
    """

    def __init__(self, nbytes, group, rank):
        self.group, self.nbytes = group, nbytes
        self.rank = rank
        page_size = os.sysconf("SC_PAGE_SIZE")
        self.span = ((nbytes + page_size - 1) // page_size) * page_size
        self.base = ctypes.c_void_p()
        self.hip = self.hsa = None
        self.bytes = None
        self.handle = _HsaIpcHandle()
        try:
            nodes = self._stage("runtime/NUMA preparation", self._prepare)
            self.nodes = sorted({node for peer_nodes in nodes for node in peer_nodes})
            handles = self._stage("allocation/export", self._allocate)
            self.handle = _HsaIpcHandle.from_buffer_copy(handles[0])
            self._stage("IPC attach/tensor view", self._attach)
        except Exception:
            try:
                self.close()
            except Exception:
                logger.exception("Failed to clean up an HCU shared host allocation")
            raise
        logger.info(
            "Shared host allocation: backend=hsa_ipc, "
            "bytes=%d, numa_nodes=%s, rank=%d/%d",
            nbytes,
            self.nodes,
            self.rank,
            len(nodes),
        )

    def _stage(self, name, action):
        error = value = None
        try:
            value = action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        results = self.group.all_gather_object((error, value))
        failures = [
            f"rank {rank}: {err}" for rank, (err, _) in enumerate(results) if err
        ]
        if failures:
            raise RuntimeError(
                f"HCU shared host allocation {name} ({self.nbytes} bytes): "
                + "; ".join(failures)
            )
        return [value for _, value in results]

    def _prepare(self):
        if os.environ.get("HSA_USERPTR_FOR_PAGED_MEM") != "0":
            raise RuntimeError(
                "HCU shared host allocations require HSA_USERPTR_FOR_PAGED_MEM=0 "
                "in the launch environment before HIP/HSA initialization"
            )
        self.hip, self.hsa = _hcu_runtime()
        self.policy = _NumaPolicy()
        if self.nbytes <= 0:
            raise ValueError("Shared host allocation size must be positive")
        return self.policy.nodes

    def _allocate(self):
        if self.rank != 0:
            return
        # hipHostMallocNumaUser honors the temporary policy. Shared
        # allocations may exceed one NUMA node, so use the group's memory nodes.
        with self.policy.interleave(self.nodes):
            _check_hip(
                self.hip,
                self.hip.hipHostMalloc(
                    ctypes.byref(self.base), self.span, _HIP_HOST_MALLOC_NUMA_USER
                ),
                f"hipHostMalloc({self.span} bytes)",
            )
        _check_hsa(
            self.hsa,
            self.hsa.hsa_amd_ipc_memory_create(
                self.base, self.span, ctypes.byref(self.handle)
            ),
            "hsa_amd_ipc_memory_create",
        )
        device = ctypes.c_void_p()
        _check_hip(
            self.hip,
            self.hip.hipHostGetDevicePointer(ctypes.byref(device), self.base, 0),
            "hipHostGetDevicePointer",
        )
        self.device_ptr = device.value
        return bytes(self.handle)

    def _attach(self):
        if self.rank != 0:
            _check_hsa(
                self.hsa,
                self.hsa.hsa_amd_ipc_memory_attach(
                    ctypes.byref(self.handle),
                    self.span,
                    0,
                    None,
                    ctypes.byref(self.base),
                ),
                "hsa_amd_ipc_memory_attach",
            )
            self.device_ptr = self.base.value
        self._make_view()

    def _make_view(self):
        # frombuffer retains the ctypes view; this allocation owns the storage.
        buffer = (ctypes.c_ubyte * self.nbytes).from_address(self.base.value)
        self.bytes = torch.frombuffer(buffer, dtype=torch.uint8)

    def close(self):
        self.bytes = None

        def detach():
            if self.rank != 0 and self.base.value:
                _check_hsa(
                    self.hsa,
                    self.hsa.hsa_amd_ipc_memory_detach(self.base),
                    "hsa_amd_ipc_memory_detach",
                )
                self.base = ctypes.c_void_p()

        def free():
            if self.rank == 0 and self.base.value:
                _check_hip(self.hip, self.hip.hipHostFree(self.base), "hipHostFree")
                self.base = ctypes.c_void_p()

        # All imports detach before the allocation owner frees its backing.
        self._stage("IPC detach", detach)
        self._stage("allocation release", free)
