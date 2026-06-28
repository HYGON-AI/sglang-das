import fcntl
import logging
import threading
import time
from multiprocessing import shared_memory
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

MM_FEATURE_CACHE_SIZE = envs.SGLANG_MM_FEATURE_CACHE_MB.get() * 1024 * 1024

MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL = (
    envs.SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC.get()
)

SHM_LOCK_FILE = "/tmp/shm_wr_lock.lock"


class ShmSyncBuffer:
    def __init__(self, byte_size: int = 4):
        self.buffer = shared_memory.SharedMemory(create=True, size=byte_size)
        self.buffer_wrapper = np.ndarray(1, dtype=np.float32, buffer=self.buffer.buf)
        self.buffer_wrapper *= 0
        self.meta_data = {
            "handle": self.buffer.name,
            "shape": self.buffer_wrapper.shape,
            "dtype": str(self.buffer_wrapper.dtype),
        }

    def __del__(self):
        if isinstance(self.buffer, shared_memory.SharedMemory):
            self.buffer.close()
            self.buffer.unlink()


class MmItemMemoryChunk:
    def __init__(self, area: Tuple, sync_buffer: ShmSyncBuffer):
        self.area = area
        self.sync_flag = sync_buffer

    @property
    def mem_size(self):
        return self.area[1] - self.area[0]

    @property
    def start(self):
        return self.area[0]

    @property
    def end(self):
        return self.area[1]

    def try_to_recycle(self, recycle_count: Optional[int] = None) -> bool:
        if recycle_count is None:
            try:
                recycle_count = get_global_server_args().tp_size
            except Exception:
                logger.info(
                    "get_global_server_args has not been inited , skip this turn 's recycle"
                )
                return False

        val = float(self.sync_flag.buffer_wrapper.item())
        logger.debug(
            f"[try_to_recycle] area={self.area}, flag={val}, "
            f"recycle_count={recycle_count}"
        )

        if val == float(recycle_count):
            self.sync_flag.buffer_wrapper *= 0.0
            return True

        return False


class MmItemMemoryPool:
    def __init__(
        self,
        memory_size,
        recycle_interval,
        device: Union[str, torch.device] = "cuda",
        recycle_count: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.recycle_count = recycle_count

        with torch.cuda.device(self.device):
            self.memory_pool = torch.empty(
                memory_size, dtype=torch.int8, device=self.device
            ).contiguous()

        self.sync_flag_list = []

        init_chunk = MmItemMemoryChunk((0, memory_size), self.pop_sync_buffer())
        self.available_chunks = [init_chunk]
        self.occupied_chunks = []

        self._lock = threading.Lock()

        self._recycle_interval = recycle_interval
        self._stop_recycler = False
        self._recycle_thread = threading.Thread(
            target=self._recycle_loop, name="MmItemMemoryPoolRecycler", daemon=True
        )
        self._recycle_thread.start()

        logger.debug(
            f"[MmItemMemoryPool] init: memory_size={memory_size}, "
            f"recycle_interval={recycle_interval}s, device={self.device}, "
            f"recycle_count={self.recycle_count}"
        )

    def shutdown(self):
        self._stop_recycler = True
        if self._recycle_thread.is_alive():
            self._recycle_thread.join(timeout=1.0)

    def _recycle_loop(self):
        while not self._stop_recycler:
            try:
                with self._lock:
                    self.recycle_chunks()
                    self.merge_chunks()
            except Exception as e:
                logger.warning(
                    f"[MmItemMemoryPool] recycle loop error: {e}", exc_info=True
                )

            time.sleep(self._recycle_interval)

    def clear_sync_flag_list(self):
        # call each chunk's __del__
        self.sync_flag_list.clear()

    def pop_sync_buffer(self):
        if len(self.sync_flag_list) == 0:
            try:
                new_sync_buffer = ShmSyncBuffer()
                return new_sync_buffer
            except:
                logger.info("allocate shm buffer failed")
                raise RuntimeError
        else:
            return self.sync_flag_list.pop()

    def push_sync_buffer(self, sync_buffer):
        self.sync_flag_list.append(sync_buffer)

    def get_available_chunk(self, src_tensor: torch.Tensor) -> MmItemMemoryChunk:
        # find currently available_chunks contain a available chunk or not
        # if not, return None
        src_tensor_size = src_tensor.numel() * src_tensor.element_size()
        min_size = self.memory_pool.numel() * self.memory_pool.element_size() + 1
        selected_chunk = None
        for chunk in self.available_chunks:
            if chunk.mem_size >= src_tensor_size:
                if chunk.mem_size < min_size:
                    min_size = chunk.mem_size
                    selected_chunk = chunk

        if selected_chunk:
            occupied_chunk_area = (
                selected_chunk.start,
                selected_chunk.start + src_tensor_size,
            )
            occupied_chunk_sync_flag = selected_chunk.sync_flag
            new_occupied_chunk = MmItemMemoryChunk(
                occupied_chunk_area, occupied_chunk_sync_flag
            )

            self.occupied_chunks.append(new_occupied_chunk)
            self.available_chunks.remove(selected_chunk)

            available_split_chunk_area = (new_occupied_chunk.end, selected_chunk.end)
            # add a new chunk
            if available_split_chunk_area[0] != available_split_chunk_area[1]:
                split_available_chunk = MmItemMemoryChunk(
                    available_split_chunk_area, self.pop_sync_buffer()
                )
                self.available_chunks.append(split_available_chunk)

            return new_occupied_chunk

        return None

    def return_a_slice_tensor_with_flag(self, src_tensor: torch.Tensor):
        sync_flag, slice_tensor, _ = self.return_a_slice_tensor_with_flag_and_chunk(
            src_tensor
        )
        return sync_flag, slice_tensor

    def return_a_slice_tensor_with_flag_and_chunk(self, src_tensor: torch.Tensor):
        with self._lock:
            available_chunk = self.get_available_chunk(src_tensor)
            if available_chunk is not None:
                return (
                    available_chunk.sync_flag.meta_data,
                    self.memory_pool[available_chunk.start : available_chunk.end],
                    available_chunk,
                )
        return None, None, None

    def recycle_unconsumed_chunk(self, chunk: MmItemMemoryChunk):
        chunk.sync_flag.buffer_wrapper *= 0.0
        with self._lock:
            if chunk in self.occupied_chunks:
                self.occupied_chunks.remove(chunk)
            self.available_chunks.append(chunk)
            self.merge_chunks()

    def recycle_chunks(self):

        new_occupied_chunks = []
        for chunk in self.occupied_chunks:
            if chunk.try_to_recycle(self.recycle_count):
                self.available_chunks.append(chunk)
            else:
                new_occupied_chunks.append(chunk)
        self.occupied_chunks = new_occupied_chunks

    def merge_chunks(self):
        # merge_all_available_chunks
        merged_chunks = []
        for chunk in sorted(self.available_chunks, key=lambda x: x.start):
            if len(merged_chunks) == 0:
                merged_chunks.append(chunk)
            else:
                if chunk.start == merged_chunks[-1].end:
                    to_merge_chunk = merged_chunks.pop()
                    to_merge_chunk_sync = to_merge_chunk.sync_flag
                    merged_chunk_area = (to_merge_chunk.start, chunk.end)
                    merged_chunks.append(
                        MmItemMemoryChunk(merged_chunk_area, to_merge_chunk_sync)
                    )
                    self.push_sync_buffer(chunk.sync_flag)
                else:
                    merged_chunks.append(chunk)

        self.available_chunks = merged_chunks


class MmItemMemoryPoolGroup:
    """One CUDA IPC source pool per TP rank/device."""

    def __init__(self, memory_size, recycle_interval):
        try:
            tp_size = get_global_server_args().tp_size
        except Exception:
            tp_size = torch.cuda.device_count()

        if tp_size > torch.cuda.device_count():
            raise RuntimeError(
                f"tp_size={tp_size} exceeds cuda device count={torch.cuda.device_count()}"
            )

        self.pools = {
            device_idx: MmItemMemoryPool(
                memory_size,
                recycle_interval,
                device=torch.device(f"cuda:{device_idx}"),
                recycle_count=1,
            )
            for device_idx in range(tp_size)
        }

    def return_slices_with_flags(self, src_tensor: torch.Tensor):
        allocated = []
        sync_flags = {}
        slice_tensors = {}
        src_bytes = src_tensor.view(torch.int8).view(-1)

        for device_idx, pool in self.pools.items():
            sync_flag, slice_tensor, chunk = pool.return_a_slice_tensor_with_flag_and_chunk(
                src_tensor
            )
            if not isinstance(slice_tensor, torch.Tensor):
                for allocated_pool, allocated_chunk in allocated:
                    allocated_pool.recycle_unconsumed_chunk(allocated_chunk)
                return None, None

            try:
                slice_tensor.copy_(src_bytes)
            except Exception:
                pool.recycle_unconsumed_chunk(chunk)
                for allocated_pool, allocated_chunk in allocated:
                    allocated_pool.recycle_unconsumed_chunk(allocated_chunk)
                raise

            allocated.append((pool, chunk))
            sync_flags[device_idx] = sync_flag
            slice_tensors[device_idx] = slice_tensor
            torch.cuda.synchronize(slice_tensor.device)

        return sync_flags, slice_tensors


class CudaIpcTensorTransportProxy:
    """
    A torch.tensor's proxy used to do inter-process data-sharing
    including:

    torch.tensor(on gpu)'s cuda-ipc-hande infos
    a shm sync buffer's meta data which is used to sync between different process
    """

    def __init__(
        self,
        data: Union[torch.Tensor, Dict[int, torch.Tensor]],
        info_data: torch.Tensor,
        sync_buffer_meta,
        pool_ipc_handle=None,
        pool_byte_offset: int = 0,
        pool_device_index: int = 0,
    ):

        if (not isinstance(data, (torch.Tensor, dict))) or (
            not isinstance(info_data, torch.Tensor)
        ):
            raise TypeError(
                "Input 'data' must be a torch.Tensor or dict[int, torch.Tensor], "
                f"but got {type(data)}"
            )

        if pool_ipc_handle is not None:
            self.proxy_state = {
                "ipc_extra": {
                    "pool_handle": pool_ipc_handle,
                    "pool_byte_offset": pool_byte_offset,
                    "pool_device_index": pool_device_index,
                    "shape": data.shape,
                    "dtype": data.dtype,
                    "stride": data.stride(),
                    "storage_offset": 0,
                    "nbytes": data.numel() * data.element_size(),
                    "recons_shape": info_data.shape,
                    "recons_dtype": info_data.dtype,
                },
                "tensor_data": None,
            }
        else:
            self.proxy_state = self.get_proxy_state(data, info_data)
        self.reconstruct_tensor = None
        self.sync_data_meta = sync_buffer_meta
        self.sync_buffers = {}

    @property
    def get_sync_flag(self):
        return self.get_sync_flag_from_meta(self.sync_data_meta)

    def get_sync_flag_from_meta(self, sync_data_meta):
        shm_name = sync_data_meta["handle"]
        if shm_name not in self.sync_buffers:
            self.sync_buffers[shm_name] = shared_memory.SharedMemory(name=shm_name)

        shape = sync_data_meta["shape"]
        dtype = sync_data_meta["dtype"]
        return np.ndarray(shape, dtype=dtype, buffer=self.sync_buffers[shm_name].buf)

    def close_shm(self):
        for sync_buffer in self.sync_buffers.values():
            sync_buffer.close()
        self.sync_buffers.clear()

    def get_proxy_state(self, data, info_data):
        # acquire all serialize metadata from _metadata
        state = {}

        if isinstance(data, dict):
            ipc_extra_by_device = {}
            try:
                for device_idx, tensor in data.items():
                    storage = tensor.untyped_storage()
                    handle = storage._share_cuda_()
                    ipc_extra_by_device[int(device_idx)] = {
                        "handle": handle,
                        "shape": tensor.shape,
                        "dtype": tensor.dtype,
                        "stride": tensor.stride(),
                        "device_index": tensor.device.index,
                        "storage_offset": tensor.storage_offset(),
                        "recons_shape": info_data.shape,
                        "recons_dtype": info_data.dtype,
                    }

                state["ipc_extra"] = None
                state["ipc_extra_by_device"] = ipc_extra_by_device
                state["tensor_data"] = None
                return state
            except Exception:
                state["ipc_extra"] = None
                state["ipc_extra_by_device"] = None
                state["tensor_data"] = info_data
                return state

        try:
            storage = data.untyped_storage()
            handle = storage._share_cuda_()

            state["ipc_extra"] = {
                "handle": handle,
                "shape": data.shape,
                "dtype": data.dtype,
                "stride": data.stride(),
                "device_index": data.device.index,
                "storage_offset": data.storage_offset(),
                "recons_shape": info_data.shape,
                "recons_dtype": info_data.dtype,
            }
            state["ipc_extra_by_device"] = None
            state["tensor_data"] = None
        except Exception as e:
            # Failed to get CUDA IPC handle (possibly tp). Falling back to default transport.
            state["ipc_extra"] = None
            state["ipc_extra_by_device"] = None
            state["tensor_data"] = data

        return state

    def _reconstruct_from_ipc_extra(
        self, ipc_extra, *, use_cache: bool, rebuild_device_idx: int
    ):
        shape = ipc_extra["shape"]
        dtype = ipc_extra["dtype"]
        stride = ipc_extra["stride"]
        # Redirect handle[0] to the consumer's device so _new_shared_cuda's
        # CUDAGuard stays there; peer access handles the cross-GPU open.
        pool_handle = ipc_extra["pool_handle"]
        redirected_handle = (rebuild_device_idx,) + tuple(pool_handle)[1:]
        target_device = torch.device(f"cuda:{rebuild_device_idx}")
        cache_key = _normalize_pool_cache_key(pool_handle, rebuild_device_idx)

        with torch.cuda.device(target_device):
            if use_cache:
                storage = _pool_handle_cache_get_or_open(cache_key, redirected_handle)
                storage_to_cache = None
            else:
                storage = _open_pooled_storage_uncached(redirected_handle)
                storage_to_cache = storage
            slice_storage = storage[
                ipc_extra["pool_byte_offset"] : ipc_extra["pool_byte_offset"]
                + ipc_extra["nbytes"]
            ]
            slice_tensor = torch.empty(0, dtype=dtype, device=target_device).set_(
                slice_storage,
                storage_offset=ipc_extra["storage_offset"],
                size=shape,
                stride=stride,
            )

        return slice_tensor, target_device, cache_key, storage_to_cache

    def _copy_slice_tensor_to_target(
        self,
        slice_tensor: torch.Tensor,
        rebuild_device: torch.device,
        recons_shape,
        recons_dtype,
    ):
        with torch.cuda.device(rebuild_device):
            reconstructed_tensor = torch.empty(
                recons_shape, dtype=recons_dtype, device=rebuild_device
            ).contiguous()
            reconstructed_tensor.view(torch.int8).view(-1).copy_(slice_tensor)

            open(SHM_LOCK_FILE, "a").close()
            # write the shm_sync_buffer with a file lock
            with open(SHM_LOCK_FILE, "w+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                sync_flag = self.get_sync_flag
                sync_flag += 1
                fcntl.flock(f, fcntl.LOCK_UN)

            self.close_shm()

        return reconstructed_tensor

    def reconstruct_on_target_device(self, rebuild_device_idx):
        rebuild_device = torch.device(f"cuda:{rebuild_device_idx}")
        if (
            isinstance(self.reconstruct_tensor, torch.Tensor)
            and self.reconstruct_tensor.device == rebuild_device
        ):
            return self.reconstruct_tensor

        ipc_extra_by_device = self.proxy_state.get("ipc_extra_by_device")
        if ipc_extra_by_device:
            ipc_extra = ipc_extra_by_device.get(rebuild_device_idx)
            if ipc_extra is None:
                ipc_extra = ipc_extra_by_device.get(str(rebuild_device_idx))
            if ipc_extra is None:
                raise RuntimeError(
                    f"Cannot find CUDA IPC source pool for device {rebuild_device_idx}"
                )
        else:
            ipc_extra = self.proxy_state.get("ipc_extra")

        if ipc_extra:
            (
                handle,
                shape,
                dtype,
                stride,
                source_device_index,
                s_offset,
                recons_shape,
                recons_dtype,
            ) = (
                ipc_extra["handle"],
                ipc_extra["shape"],
                ipc_extra["dtype"],
                ipc_extra["stride"],
                ipc_extra["device_index"],
                ipc_extra["storage_offset"],
                ipc_extra["recons_shape"],
                ipc_extra["recons_dtype"],
            )

            try:
                target_device = torch.device(f"cuda:{source_device_index}")
                with torch.cuda.device(target_device):
                    storage = torch.UntypedStorage._new_shared_cuda(*handle)
                    slice_tensor = torch.empty(
                        0, dtype=dtype, device=target_device
                    ).set_(storage, storage_offset=s_offset, size=shape, stride=stride)

                    reconstructed_tensor = torch.empty(
                        recons_shape, dtype=recons_dtype, device=rebuild_device
                    ).contiguous()
                    reconstructed_tensor.view(torch.int8).view(-1).copy_(slice_tensor)

                    open(SHM_LOCK_FILE, "a").close()
                    # write the shm_sync_buffer with a file lock
                    with open(SHM_LOCK_FILE, "w+") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        sync_data_meta = self.sync_data_meta
                        if isinstance(sync_data_meta, dict) and (
                            rebuild_device_idx in sync_data_meta
                            or str(rebuild_device_idx) in sync_data_meta
                        ):
                            sync_data_meta = sync_data_meta.get(
                                rebuild_device_idx,
                                sync_data_meta.get(str(rebuild_device_idx)),
                            )
                        sync_flag = self.get_sync_flag_from_meta(sync_data_meta)
                        sync_flag += 1
                        fcntl.flock(f, fcntl.LOCK_UN)

                    self.close_shm()

            except Exception as e:
                logger.info(f"Error: Failed to deserialize from CUDA IPC handle ({e}).")
                raise e
        elif isinstance(self.proxy_state["tensor_data"], torch.Tensor):
            reconstructed_tensor = self.proxy_state["tensor_data"].to(
                rebuild_device, non_blocking=True
            )
        else:
            raise TypeError("invalid proxy_state")

        self.reconstruct_tensor = reconstructed_tensor
        return self.reconstruct_tensor
