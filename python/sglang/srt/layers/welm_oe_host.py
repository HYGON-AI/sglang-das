"""WeLM over-encoding (OE) tables served from host memory.

The four OE n-gram tables are 65.5 GB in bf16 but every token reads only one
512-wide row per table. With WELM_OE_HOST_DIR set (files written by
welm_oe_host_src/export_oe.py, normally under /dev/shm), each rank maps the
shared files, registers them with hipHostRegister (mapped, read-only) and
gathers rows straight over the bus, so the tables use no HBM and need no
cross-rank reduction. One host copy serves every process on the node.
"""
import ctypes
import importlib.util
import json
import mmap
import os

import torch
from torch import nn

_mod = None
_hip = None


def _ext():
    global _mod
    if _mod is None:
        so = os.path.join(os.path.dirname(__file__), "oe_host_gather.so")
        spec = importlib.util.spec_from_file_location("oe_host_gather", so)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
    return _mod


def host_dir():
    return os.environ.get("WELM_OE_HOST_DIR") or None


def table_meta(directory):
    return json.load(open(os.path.join(directory, "meta.json")))["tables"]


class HostOEEmbedding(nn.Module):
    """Drop-in for the OE VocabParallelEmbedding: forward(ids) -> [n, dim] (full rows)."""

    def __init__(self, directory: str, index: int, num_embeddings: int, embedding_dim: int):
        super().__init__()
        meta = table_meta(directory)[str(index)]
        assert meta["shape"] == [num_embeddings, embedding_dim], (meta, num_embeddings, embedding_dim)
        self.path = os.path.join(directory, f"{index}.bin")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._dev_ptr = None
        self._keep = None

    def _ensure_mapped(self):
        global _hip
        if self._dev_ptr is not None:
            return
        if _hip is None:
            _hip = ctypes.CDLL("libamdhip64.so")
        size = self.num_embeddings * self.embedding_dim * 2
        fd = os.open(self.path, os.O_RDONLY)
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ, flags=mmap.MAP_SHARED)
        os.close(fd)
        buf = (ctypes.c_char * size).from_address(_mmap_address(mm))
        host_ptr = ctypes.c_void_p(ctypes.addressof(buf))
        # hipHostRegisterMapped (2) | hipHostRegisterReadOnly (8)
        err = _hip.hipHostRegister(host_ptr, ctypes.c_size_t(size), ctypes.c_uint(2 | 8))
        if err != 0:
            err = _hip.hipHostRegister(host_ptr, ctypes.c_size_t(size), ctypes.c_uint(2))
        assert err == 0, f"hipHostRegister({self.path}) failed: {err}"
        dptr = ctypes.c_void_p()
        assert _hip.hipHostGetDevicePointer(ctypes.byref(dptr), host_ptr, 0) == 0
        self._dev_ptr = dptr.value
        self._keep = (mm, buf)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._ensure_mapped()
        ids = input_ids.reshape(-1).to(torch.int64).contiguous()
        out = torch.empty((ids.numel(), self.embedding_dim), dtype=torch.bfloat16, device=input_ids.device)
        _ext().oe_host_gather(self._dev_ptr, self.num_embeddings, self.embedding_dim, ids, out)
        return out.view(*input_ids.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return f"host_table={self.path}, num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


def _mmap_address(mm) -> int:
    """Base address of an mmap object (read-only maps cannot use from_buffer)."""
    import numpy as np

    return np.frombuffer(mm, dtype=np.uint8).ctypes.data
