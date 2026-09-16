# SPDX-License-Identifier: Apache-2.0
"""Opt-in MiniMax-H3 channel INT8 loading, Linear, and fused modulation."""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch.nn import Parameter
from triton.language.extra import libdevice

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.linear import LinearMethodBase
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs

FORMAT = "sglang-minimax-h3-int8-v1"
MANIFEST = "minimax_h3_int8.json"


def backbone_layers(num_layers=50):
    return {
        f"blocks.{i}.{suffix}"
        for i in range(num_layers)
        for suffix in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    }


def inspect_int8_checkpoint(paths):
    """Return metadata only for this format; leave other formats to their loaders."""
    from safetensors import safe_open

    if not paths:
        return None
    roots = {Path(path).resolve().parent for path in paths}
    if len(roots) != 1:
        return None
    root = roots.pop()
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        for path in paths:
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                if (checkpoint.metadata() or {}).get("format") == FORMAT:
                    raise ValueError(f"MiniMax-H3 INT8 is missing {MANIFEST}")
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT or manifest.get("status") != "complete":
        raise ValueError("Invalid or incomplete MiniMax-H3 INT8 export")
    if manifest.get("profile") not in ("backbone", "backbone-adaln"):
        raise ValueError("Unsupported MiniMax-H3 INT8 profile")
    if manifest.get("layout") != "native-qkv" or manifest.get("variant") not in (
        "fl2va",
        "ref2va",
    ):
        raise ValueError("Invalid MiniMax-H3 INT8 layout or model variant")
    targets = backbone_layers()
    if set(manifest.get("quantized_layers", [])) != targets:
        raise ValueError(
            "MiniMax-H3 INT8 must cover exactly the 200 backbone Linear layers"
        )
    shards = manifest.get("shards", [])
    if len(shards) != len(set(shards)) or set(shards) != {Path(p).name for p in paths}:
        raise ValueError("MiniMax-H3 INT8 checkpoint shards do not match the manifest")
    tensors = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            if (checkpoint.metadata() or {}).get("format") != FORMAT:
                raise ValueError("Mixed MiniMax-H3 INT8 checkpoint formats")
            for name in checkpoint.keys():
                if name in tensors:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                tensor = checkpoint.get_slice(name)
                tensors[name] = (tensor.get_dtype(), tensor.get_shape())
    actual_int8 = {
        n[:-7] for n, (d, _) in tensors.items() if d == "I8" and n.endswith(".weight")
    }
    if actual_int8 != targets:
        raise ValueError(
            "MiniMax-H3 INT8 checkpoint coverage does not match its manifest"
        )
    for name, (dtype, _) in tensors.items():
        if dtype not in ("I8", "BF16", "F32") or (
            dtype == "I8" and name[:-7] not in targets
        ):
            raise ValueError(f"Unexpected tensor dtype: {name} ({dtype})")
        if (
            name.endswith(".weight_scale_inv")
            and name.removesuffix(".weight_scale_inv") not in targets
        ):
            raise ValueError(f"Unexpected quantization scale: {name}")
    for prefix in targets:
        _, shape = tensors[prefix + ".weight"]
        if len(shape) != 2 or tensors.get(prefix + ".weight_scale_inv") != (
            "F32",
            [shape[0], 1],
        ):
            raise ValueError(f"Invalid per-channel weight scale: {prefix}")
    compressed = manifest["profile"] == "backbone-adaln"
    if compressed:
        if tensors.get("adaln_t_table") != ("F32", [1025, 8]):
            raise ValueError("Compressed AdaLN requires an FP32 [1025,8] table")
        for prefix in [f"blocks.{i}" for i in range(50)] + ["final_layer"]:
            name = prefix + ".adaln_proj.linear.weight"
            dtype, shape = tensors.get(name, (None, []))
            if dtype != "F32" or len(shape) != 2 or shape[1] != 8:
                raise ValueError(f"Invalid compressed AdaLN projection: {name}")
        if any(n.startswith("time_embedder.") for n in tensors):
            raise ValueError("Compressed AdaLN must not contain the original time MLP")
    elif "adaln_t_table" in tensors:
        raise ValueError("Full AdaLN profile must not contain a compressed table")
    return manifest


class MiniMaxH3Int8Config(QuantizationConfig):
    checkpoint_uses_native_qkv_layout = True

    def __init__(self, *, targets=None, compressed_adaln=False):
        if not envs.SGLANG_MINIMAX_H3_INT8_ENABLE:
            raise ValueError("MiniMax-H3 INT8 requires SGLANG_MINIMAX_H3_INT8_ENABLE=1")
        if not targets:
            raise ValueError(
                "MiniMax-H3 INT8 requires a validated checkpoint from "
                "the standalone export_int8.py; use --transformer-weights-path"
            )
        self.targets = frozenset(targets)
        self.compressed_adaln = compressed_adaln
        self.fused = envs.SGLANG_MINIMAX_H3_INT8_FUSION

    @classmethod
    def get_name(cls):
        return "minimax_h3_int8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        # A generic model config must not enable this model-specific adapter.
        return cls()

    def get_quant_method(self, layer, prefix):
        from sglang.multimodal_gen.runtime.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )

        if not isinstance(layer, LinearBase):
            return None
        if prefix in self.targets:
            return MiniMaxH3Int8LinearMethod(fused=self.fused)
        return UnquantizedLinearMethod()


class MiniMaxH3Int8LinearMethod(LinearMethodBase):
    def __init__(self, *, fused=True):
        self.fused = fused

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        rows = sum(output_partition_sizes)
        weight = Parameter(
            torch.empty(rows, input_size_per_partition, dtype=torch.int8),
            requires_grad=False,
        )
        scale = Parameter(
            torch.empty(rows, 1, dtype=torch.float32), requires_grad=False
        )
        set_weight_attrs(
            weight, {"input_dim": 1, "output_dim": 0, "missing_param_init": "error"}
        )
        set_weight_attrs(scale, {"output_dim": 0, "missing_param_init": "error"})
        set_weight_attrs(weight, extra_weight_attrs)
        set_weight_attrs(scale, extra_weight_attrs)
        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale_inv", scale)

    def process_weights_after_loading(self, layer):
        if layer.weight.dtype != torch.int8:
            raise ValueError("MiniMax-H3 INT8 weight dtype must be int8")
        scale = layer.weight_scale_inv
        if scale.dtype != torch.float32 or scale.shape != (layer.weight.shape[0], 1):
            raise ValueError("MiniMax-H3 INT8 needs one FP32 scale per output channel")
        if not bool(torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError("MiniMax-H3 INT8 scales must be finite and positive")

    def apply(self, layer, x, bias=None):
        shape = x.shape
        rows = x.reshape(-1, shape[-1]).contiguous()
        result = linear(
            rows,
            layer.weight,
            layer.weight_scale_inv,
            bias,
            mode="both" if self.fused else "reference",
        )
        return result.reshape(*shape[:-1], layer.weight.shape[0])


@triton.jit
def _quant_rows(
    X, Q, S, M: tl.constexpr, K: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * STRIDE + col, (row < M) & (col < K), 0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-12)
    # Match PyTorch's float scalar division: multiply by the FP32 reciprocal.
    scale = amax * (1.0 / 127.0)
    q = libdevice.nearbyint(tl.div_rn(x, scale))
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
    tl.store(Q + row * K + col, q, col < K)
    tl.store(S + row, scale)


@triton.jit
def _restore_output(
    A,
    SX,
    SW,
    BIAS,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < M * N
    row, col = idx // N, idx % N
    a = tl.load(A + idx, mask, 0).to(tl.float32)
    sx = tl.load(SX + row, mask, 0)
    sw = tl.load(SW + col, mask, 0)
    y = (a * sx) * sw
    if HAS_BIAS:
        y = y + tl.load(BIAS + col, mask, 0).to(tl.float32)
    tl.store(OUT + idx, y, mask)


def quant_rows(x):
    """Per-token absmax/127, ties-to-even; pad M for _int_mm, not K."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda or x.stride(1) != 1:
        raise ValueError("Expected GPU BF16 [M,K], contiguous last dimension")
    m, k = x.shape
    if m < 1 or k < 1 or k > 8192:
        raise ValueError("Experimental supported domain: M>0, 0<K<=8192")
    padded_m = max(32, triton.cdiv(m, 16) * 16)
    q = torch.empty((padded_m, k), device=x.device, dtype=torch.int8)
    scale = torch.empty((padded_m, 1), device=x.device, dtype=torch.float32)
    _quant_rows[(padded_m,)](
        x,
        q,
        scale,
        m,
        k,
        x.stride(0),
        triton.next_power_of_2(k),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q, scale


def restore_output(accum, sx, sw, out, bias=None):
    """FP32 accumulate -> two ordered scale multiplies -> optional bias -> BF16."""
    m, n = out.shape
    tensors = (accum, sx, sw, out) + (() if bias is None else (bias,))
    if not all(
        t.is_cuda and t.device == out.device and t.is_contiguous() for t in tensors
    ):
        raise ValueError("Expected contiguous tensors on one GPU")
    if accum.dtype != torch.int32 or out.dtype != torch.bfloat16:
        raise ValueError("Expected INT32 accum and BF16 out")
    if sx.dtype != torch.float32 or sw.dtype != torch.float32:
        raise ValueError("Expected FP32 scales")
    if (
        accum.ndim != 2
        or accum.shape[0] < m
        or accum.shape[1] != n
        or sx.numel() < m
        or sw.numel() != n
    ):
        raise ValueError("Incompatible accumulator/scale/output shapes")
    if bias is not None and (
        bias.numel() != n or bias.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError("Expected BF16/FP32 bias with N elements")
    _restore_output[(triton.cdiv(m * n, 1024),)](
        accum,
        sx,
        sw,
        out if bias is None else bias,
        out,
        m,
        n,
        bias is not None,
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def linear(x, qw, sw, bias=None, *, mode="both", chunk=4096):
    """Dynamic per-token quantization, INT8 GEMM, and ordered FP32 epilogue."""
    if mode not in ("reference", "epilogue", "both"):
        raise ValueError(mode)
    if x.ndim != 2 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        raise ValueError("Expected BF16 [M,K] with contiguous last dimension")
    if not x.is_cuda or qw.device != x.device or sw.device != x.device:
        raise ValueError("Expected one GPU")
    if (
        qw.dtype != torch.int8
        or not qw.is_contiguous()
        or sw.dtype != torch.float32
        or not sw.is_contiguous()
    ):
        raise ValueError("Expected contiguous INT8 weights and FP32 scales")
    m, k = x.shape
    if (
        qw.ndim != 2
        or qw.shape[1] != k
        or k <= 16
        or k % 16
        or qw.shape[0] <= 16
        or qw.shape[0] % 8
        or sw.numel() != qw.shape[0]
    ):
        raise ValueError("Unsupported _int_mm dimensions or scale shape")
    if m < 1 or chunk < 16 or chunk % 16:
        raise ValueError("Expected M>0 and chunk a positive multiple of 16")
    out = torch.empty((m, qw.shape[0]), device=x.device, dtype=x.dtype)
    for lo in range(0, m, chunk):
        part = x[lo : lo + chunk]
        if mode == "both" and k <= 8192:
            qx, sx = quant_rows(part)
        else:
            r = part.float()
            sx = r.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127.0
            qx = (r / sx).round().clamp_(-127, 127).to(torch.int8)
            padding = max(32, triton.cdiv(len(part), 16) * 16) - len(part)
            if padding:
                qx = torch.nn.functional.pad(qx, (0, 0, 0, padding))
        acc = torch._int_mm(qx.contiguous(), qw.t())
        if mode == "reference":
            y = acc[: len(part)].float().mul_(sx[: len(part)]).mul_(sw.reshape(1, -1))
            if bias is not None:
                y.add_(bias)
            out[lo : lo + len(part)] = y.to(x.dtype)
        else:
            restore_output(acc, sx, sw, out[lo : lo + len(part)], bias)
    return out


@triton.jit
def _affine(
    X,
    S,
    B,
    I,
    Y,
    M: tl.constexpr,
    H: tl.constexpr,
    SS: tl.constexpr,
    BS: tl.constexpr,
    T: tl.constexpr,
):
    idx = tl.program_id(0) * T + tl.arange(0, T)
    mask = idx < M * H
    row = idx // H
    col = idx % H
    cond = tl.load(I + row, mask, 0)
    x = tl.load(X + idx, mask, 0).to(tl.float32)
    s = tl.load(S + cond * SS + col, mask, 0)
    b = tl.load(B + cond * BS + col, mask, 0)
    y = x * (1.0 + s) + b
    tl.store(Y + idx, y, mask)


@triton.jit
def _gate(
    X, G, O, I, Y, M: tl.constexpr, H: tl.constexpr, GS: tl.constexpr, T: tl.constexpr
):
    idx = tl.program_id(0) * T + tl.arange(0, T)
    mask = idx < M * H
    row = idx // H
    col = idx % H
    cond = tl.load(I + row, mask, 0)
    x = tl.load(X + idx, mask, 0).to(tl.float32)
    o = tl.load(O + idx, mask, 0).to(tl.float32)
    g = tl.load(G + cond * GS + col, mask, 0)
    tl.store(Y + idx, x + g * o, mask)


def affine(x, shift, scale, indices):
    m, h = x.shape
    y = torch.empty_like(x)
    _affine[(triton.cdiv(m * h, 1024),)](
        x,
        scale,
        shift,
        indices,
        y,
        m,
        h,
        scale.stride(0),
        shift.stride(0),
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y


def gate(x, g, other, indices):
    m, h = x.shape
    y = torch.empty_like(x)
    _gate[(triton.cdiv(m * h, 1024),)](
        x,
        g,
        other,
        indices,
        y,
        m,
        h,
        g.stride(0),
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y


def supports_fp32_modulation(x, params, indices, *, dtype, other=None):
    if not (
        x.is_cuda
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and dtype == torch.bfloat16
        and x.is_contiguous()
        and indices.device == x.device
        and indices.ndim == 1
        and indices.numel() == x.shape[0]
        and indices.is_contiguous()
        and indices.dtype in (torch.int32, torch.int64)
    ):
        return False
    if not all(
        p.device == x.device
        and p.dtype == torch.float32
        and p.ndim == 2
        and p.shape[1] == x.shape[1]
        and p.stride(1) == 1
        and p.shape[0] == params[0].shape[0]
        for p in params
    ):
        return False
    return other is None or (
        other.device == x.device
        and other.dtype == x.dtype
        and other.shape == x.shape
        and other.is_contiguous()
    )
