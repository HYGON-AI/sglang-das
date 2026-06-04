# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/custom_all_reduce.py

import ctypes
import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    can_use_custom_all_reduce_with_nvlink,
    is_weak_contiguous,
    is_full_nvlink,
)
from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_hip,
    is_musa,
    log_info_on_rank0,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_musa = is_musa()
_sglang_dcu_custom_cache = get_bool_env_var("DCU_CUSTOM_CACHE", default="true")

# try:
#     if ops.use_vllm_custom_allreduce and not _is_hip:
#         # Use vLLM custom allreduce
#         ops.meta_size()
#     elif ops.use_dcu_custom_allreduce:
#         ops.meta_size()
#     else:
#         # Use custom allreduce from sgl kernel (ROCM and TRT-LLM)
#         import sgl_kernel  # noqa: F401
#     custom_ar = True
# except Exception:
#     # For CPUs
#     custom_ar = False

logger = logging.getLogger(__name__)


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    _MAX_CAR_SIZE = 8192 * 1024
    if _is_hip:
        # crossover is at 16MB buffer size for ROCm
        _MAX_CAR_SIZE = 2 * 8192 * 1024
    if _is_musa:
        # crossover is at 128MB buffer size for MUSA
        _MAX_CAR_SIZE = 16 * 8196 * 1024

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size=_MAX_CAR_SIZE,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True  # This can be modified in-place by context manager in piecewise cuda graph runner
        self.original_disabled = True  # To store the original state
        self.use_amd_deterministic_impl = _use_amd_deterministic_impl()

        if not ops.IS_CUSTOM_AR_AVAILABLE:
            # disable because of missing custom allreduce library
            # e.g. in a non-cuda environment
            return

        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        full_nvlink = can_use_custom_all_reduce_with_nvlink(
            group=group,
            device=device,
            supported_world_size=self._SUPPORTED_WORLD_SIZES,
            cls_name="CustomAllreduce",
        )
        if full_nvlink is None:
            return  # fail to get nvlink status

        self.group = group
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.full_nvlink = full_nvlink

        if not _is_hip:
            # Buffers memory are owned by this Python class and passed to C++.
            # Meta data composes of two parts: meta data for synchronization and a
            # temporary buffer for storing intermediate allreduce results.
            self.meta_ptrs = self.create_shared_buffer(
                ops.meta_size() + max_size, group=group
            )
            # This is a pre-registered IPC buffer. In eager mode, input tensors
            # are first copied into this buffer before allreduce is performed
            self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
            # This is a buffer for storing the tuples of pointers pointing to
            # IPC buffers from all ranks. Each registered tuple has size of
            # 8*world_size bytes where world_size is at most 8. Allocating 8MB
            # is enough for 131072 such tuples. The largest model I've seen only
            # needs less than 10000 of registered tuples.
            self.rank_data = torch.empty(
                max_size, dtype=torch.uint8, device=self.device
            )
            self._ptr = ops.init_custom_ar(
                self.meta_ptrs, self.rank_data, rank, self.full_nvlink
            )
            ops.register_buffer(self._ptr, self.buffer_ptrs)
        else:
            # meta data buffers need to be "uncached" for signal on MI200
            self.meta = ops.allocate_meta_buffer(ops.meta_size() + max_size)
            self.buffer = torch.empty(max_size, dtype=torch.uint8, device=self.device)
            handle = ops.get_meta_buffer_ipc_handle(self.meta)
            shard_data = (
                bytes(handle),  # ipc handle to base ptr
                0,  # offset of base ptr
            )
            handles, offsets = self._gather_ipc_meta(shard_data)
            self.rank_data = torch.empty(
                max_size, dtype=torch.uint8, device=self.device
            )
            self._ptr = ops.init_custom_ar(
                self.meta, self.rank_data, handles, offsets, rank, self.full_nvlink
            )
            self.register_buffer(self.buffer)

        self.disabled = False
        self.original_disabled = False  # Ensure original_disabled == disabled
        self.tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int, group: Optional[ProcessGroup] = None
    ) -> List[int]:
        """
        Creates a shared buffer and returns a list of pointers
        representing the buffer on all processes in the group.
        """
        lib = CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        if _is_musa:
            lib.cudaMemset(pointer, 0, size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)  # type: ignore
            else:
                pointers.append(lib.cudaIpcOpenMemHandle(h).value)  # type: ignore

        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: List[int], group: Optional[ProcessGroup] = None
    ) -> None:
        rank = dist.get_rank(group=group)
        lib = CudaRTLibrary()
        lib.cudaFree(ctypes.c_void_p(pointers[rank]))

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def _get_ipc_meta(self, inp: torch.Tensor):
        # _share_cuda_() doesn't accept meta buffer not allocated from
        # PyTorch cache allocator, use direct HIP call to get IPC handle
        handle = ops.get_meta_buffer_ipc_handle(inp)
        shard_data = (
            bytes(handle),  # ipc handle to base ptr
            0,  # offset of base ptr
        )
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):
        # Note: don't use `[[None]] * self.world_size` here
        # because it will create a list of the same reference
        all_data: List[Optional[Any]] = [[None] for i in range(self.world_size)]
        all_data[self.rank][0] = shard_data

        ranks = dist.get_process_group_ranks(group=self.group)
        ranks.sort()
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )

        # we cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.

        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0][0])  # type: ignore
            offsets.append(all_data[i][0][1])  # type: ignore
        return handles, offsets

    def register_buffer(self, inp: torch.Tensor):
        handles, offsets = self._get_ipc_meta(inp)
        ops.register_buffer(self._ptr, inp, handles, offsets)

    def register_graph_buffers(self):
        if _is_hip:
            handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
            handles, offsets = self._gather_ipc_meta((bytes(handle), offset))
            log_info_on_rank0(logger, f"Registering {len(offset)} cuda graph addresses")
            ops.register_graph_buffers(self._ptr, handles, offsets)
        else:
            handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
            log_info_on_rank0(logger, f"Registering {len(offset)} cuda graph addresses")
            # We cannot directly use `dist.all_gather_object` here
            # because it is incompatible with `gloo` backend under inference mode.
            # see https://github.com/pytorch/pytorch/issues/126032 for details.
            all_data = [
                [None, None] for _ in range(dist.get_world_size(group=self.group))
            ]
            all_data[self.rank] = [handle, offset]
            ranks = sorted(dist.get_process_group_ranks(group=self.group))
            for i, rank in enumerate(ranks):
                dist.broadcast_object_list(
                    all_data[i], src=rank, group=self.group, device="cpu"
                )
            # Unpack list of tuples to tuple of lists.
            handles = [d[0] for d in all_data]  # type: ignore
            offsets = [d[1] for d in all_data]  # type: ignore
            ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if not _is_hip:
            if self.world_size == 2 or self.full_nvlink:
                return inp_size <= self.max_size
            return False

        if _is_hip:
            if self.use_amd_deterministic_impl:
                return True
            if self.full_nvlink:
                return inp_size <= self.max_size
            return False

        return False

    def _all_reduce_impl(self, inp: torch.Tensor, registered: bool):
        out = torch.empty_like(inp)
        if not _is_hip:  # CUDA-like
            if registered:
                ops.all_reduce(self._ptr, inp, out, 0, 0)
            else:
                ops.all_reduce(
                    self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
                )
        elif self.use_amd_deterministic_impl:
            inp_size = inp.numel() * inp.element_size()
            if inp_size < self.max_size:
                reg_buffer = self.buffer.view(inp.dtype)[: inp.numel()]
                ops.deterministic_all_reduce_unreg(self._ptr, inp, reg_buffer, out)
            else:
                self.register_buffer(inp)
                ops.deterministic_all_reduce_reg(self._ptr, inp, out)
        else:  # normal AMD ROCm path
            if registered:
                ops.all_reduce_reg(self._ptr, inp, out)
            else:
                ops.all_reduce_unreg(self._ptr, inp, self.buffer, out)
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self._all_reduce_impl(input, registered=not self.tms_cudagraph)
            else:
                # Could be warmup OR piecewise cuda graph split op execution.
                # In piecewise cuda graph, split ops run eagerly outside the graph
                # but _IS_CAPTURING is still True. We need to do real all-reduce.
                if is_in_piecewise_cuda_graph():
                    # Split op execution - do real all-reduce
                    return self._all_reduce_impl(input, registered=False)
                else:
                    # True warmup - mimic the allocation pattern since custom
                    # allreduce is out-of-place.
                    return torch.zeros_like(input)
        else:
            return self._all_reduce_impl(input, registered=False)

    def close(self):
        if not self.disabled and self._ptr:
            ops.dispose(self._ptr)
            if _is_cuda:
                self.free_shared_buffer(self.meta_ptrs)
                self.free_shared_buffer(self.buffer_ptrs)
            self._ptr = 0

    def __del__(self):
        self.close()

class DCUCustomAllreduce:
    
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8, 16]

    # max_size: max supported allreduce size
    def __init__(self,
                 group: ProcessGroup,
                 device: Union[int, str, torch.device],
                 max_size=8192 * 512) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True
        self.original_disabled = True  # To store the original state

        if not ops.IS_CUSTOM_AR_AVAILABLE:
            # disable because of missing custom allreduce library
            # e.g. in a non-GPU environment
            logger.info("Custom allreduce is disabled because "
                        "of missing custom allreduce library")
            return

        self.group = group

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group.")

        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom allreduce for multi-node case.
            logger.warning(
                "Custom allreduce is disabled because this process group"
                " spans across nodes.")
            return

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        
        # if world_size > envs.VLLM_CUSTOM_ALLREDUCE_SUPPORTED_WORLDSIZE_MAX:
        if world_size > 16:
            return

        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in DCUCustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size, str(DCUCustomAllreduce._SUPPORTED_WORLD_SIZES))
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            device_ids = list(range(torch.cuda.device_count()))

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id],
                              dtype=torch.int,
                              device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu")
            for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        # assert current_platform.is_cuda_alike()
        # fully_connected = current_platform.is_fully_connected(
        #     physical_device_ids)
        if _is_cuda or _is_hip:
            fully_connected = is_full_nvlink(physical_device_ids, world_size)
        
        # if world_size > 2 and not fully_connected:
        if not fully_connected:
            max_size = 32 * 8192 * 2
            # if not envs.VLLM_PCIE_USE_CUSTOM_ALLREDUCE:
            #     logger.warning(
            #         "Custom allreduce is disabled because it's not supported on"
            #         " more than two PCIe-only GPUs. To silence this warning, "
            #         "specify disable_custom_all_reduce=True explicitly.")
            #     return
            logger.warning(
                "We are using PCIe's custom allreduce."
                "If the performance is poor, we can add "
                "--disable-custom-all-reduce in the instruction.")
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        if not _is_hip and not _can_p2p(rank, world_size):
            logger.warning(
                "Custom allreduce is disabled because your platform lacks "
                "GPU P2P capability or P2P test failed. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.")
            return

        self.disabled = False
        self.original_disabled = False  # Ensure original_disabled == disabled
        # Buffers memory are owned by this Python class and passed to C++.
        # Meta data composes of two parts: meta data for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        self.meta_ptrs = self.create_shared_buffer(ops.meta_size() + max_size,
                                                   group=group,
                                                   uncached=True)
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer_ptrs = self.create_shared_buffer(max_size, group=group)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(8 * 1024 * 1024,
                                     dtype=torch.uint8,
                                     device=self.device)
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        self.fully_connected = fully_connected
        self._ptr = ops.init_custom_ar(self.meta_ptrs, self.rank_data, rank,
                                       self.fully_connected)
        ops.register_buffer(self._ptr, self.buffer_ptrs)

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the 
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def register_graph_buffers(self):
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        logger.info("Registering %d cuda graph addresses", len(offset))
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data = [[None, None]
                    for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(all_data[i],
                                       src=rank,
                                       group=self.group,
                                       device="cpu")
        # Unpack list of tuples to tuple of lists.
        handles = [d[0] for d in all_data]  # type: ignore
        offsets = [d[1] for d in all_data]  # type: ignore
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        return inp_size <= self.max_size

    def all_reduce(self,
                   inp: torch.Tensor,
                   *,
                   out: torch.Tensor = None,
                   registered: bool = False):
        """Performs an out-of-place all reduce.
        
        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(self._ptr, inp, out, self.buffer_ptrs[self.rank],
                           self.max_size)
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                if _sglang_dcu_custom_cache:
                    return self.all_reduce(input, registered=True)
                else:
                    return self.all_reduce(input, registered=False)
            else:
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.all_reduce(input, registered=False)

    def close(self):
        if not self.disabled and self._ptr:
            if ops is not None:
                ops.dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.buffer_ptrs, rank=self.rank)

    def __del__(self):
        self.close()


    @staticmethod
    def create_shared_buffer(size_in_bytes: int,
                             group: Optional[ProcessGroup] = None,
                             uncached: Optional[bool] = False) -> list[int]:
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    @staticmethod
    def free_shared_buffer(pointers: list[int],
                           group: Optional[ProcessGroup] = None,
                           rank: Optional[int] = 0) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        if ops is not None:
            ops.free_shared_buffer(pointers[rank])

def dispatch_custom_allreduce():
    """Return the CustomAllreduce class to use (aiter on ROCm if enabled).

    On AMD with 1-stage AR enabled, use sglang's CustomAllreduce.
    Otherwise use AiterCustomAllreduce if available.

    On CUDA, the JIT-compiled v2 implementation is used by default.
    Set SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0 to fall back to the legacy CustomAllreduce.
    Note: ``ServerArgs._handle_environment_variables`` forces this env to "0" when
    ``nnodes > 1`` since custom AR is intra-node only.
    """
    if _is_cuda and envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2.get():
        from .custom_all_reduce_v2 import CustomAllReduceV2

        logger.debug("[AR] Using CustomAllReduceV2 (JIT-compiled)")
        return CustomAllReduceV2

    if _is_cuda or _is_musa:
        return CustomAllreduce

    assert _is_hip

    if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
        if envs.SGLANG_USE_1STAGE_ALLREDUCE.get():
            logger.debug(
                "[AR] All-reduce: 1-stage kernel (SGLANG_USE_1STAGE_ALLREDUCE=1)"
            )
        else:
            logger.debug("[AR] All-reduce: default (SGLANG_USE_1STAGE_ALLREDUCE=0)")
    elif envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get():
        logger.debug(
            "[AR] All-reduce: 1-stage kernel (deterministic inference enabled)"
        )
    else:
        logger.debug("[AR] All-reduce: default")

    # On AMD with 1-stage AR, use sglang's CustomAllreduce
    # (AiterCustomAllreduce doesn't have deterministic_all_reduce method)
    if _use_amd_deterministic_impl():
        return CustomAllreduce

    if get_bool_env_var("SGLANG_USE_AITER_AR", default="true"):
        try:
            from aiter.dist.device_communicators.custom_all_reduce import (
                CustomAllreduce as AiterCustomAllreduce,
            )

            logger.info("[AR] Using AiterCustomAllreduce (DCU/HIP default)")
            tms_cudagraph = envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
            return partial(
                AiterCustomAllreduce,
                enable_register_for_capturing=not tms_cudagraph,
            )
        except ImportError as e:
            logger.warning(
                "[AR] Aiter custom all-reduce not available; "
                "falling back to sglang CustomAllreduce. Details: %s",
                e,
            )
            return CustomAllreduce

    return CustomAllreduce


def _use_amd_deterministic_impl() -> bool:
    if not _is_hip:  # CUDA is always deterministic
        return False
    if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
        return envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
    else:
        return envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
