# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 channel INT8/FP8 loading, Linear, and fused modulation."""

import json
import os
import warnings
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch.nn import Parameter
from triton.language.extra import libdevice

from sglang.multimodal_gen import envs
from sglang.srt.utils import is_hcu

_is_hcu = is_hcu()
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


def _quant_optimize(precision):
    name = "SGLANG_MINIMAX_H3_QUANT_OPTIMIZE"
    legacy = (
        "SGLANG_MINIMAX_H3_INT8_FUSION"
        if precision == "int8"
        else "SGLANG_MINIMAX_H3_FP8_OPTIMIZE"
    )
    enabled = envs.SGLANG_MINIMAX_H3_QUANT_OPTIMIZE
    if legacy in os.environ:
        old = getattr(envs, legacy)
        if name in os.environ and old != enabled:
            raise ValueError(f"Conflicting {name} and deprecated {legacy}")
        warnings.warn(
            f"{legacy} is deprecated; use {name} instead", FutureWarning, stacklevel=3
        )
        if name not in os.environ:
            enabled = old
    return enabled


class MiniMaxH3QuantConfig(QuantizationConfig):
    checkpoint_uses_native_qkv_layout = True
    precision = ""

    def __init__(self, *, targets=None, compressed_adaln=False):
        if not targets:
            raise ValueError(
                "MiniMax-H3 quantization requires a validated checkpoint; "
                "use --transformer-weights-path"
            )
        self.targets = frozenset(targets)
        self.compressed_adaln = compressed_adaln
        self.fused = _quant_optimize(self.precision)

    @classmethod
    def get_name(cls):
        return "minimax_h3_" + cls.precision

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
            method = (
                MiniMaxH3Int8LinearMethod
                if self.precision == "int8"
                else MiniMaxH3Fp8LinearMethod
            )
            return method(fused=self.fused)
        return UnquantizedLinearMethod()


class MiniMaxH3Int8Config(MiniMaxH3QuantConfig):
    precision = "int8"

    def __init__(self, *, targets=None, compressed_adaln=False):
        # Honor an explicit legacy opt-out; new commands need no enable flag.
        if not envs.SGLANG_MINIMAX_H3_INT8_ENABLE:
            raise ValueError(
                "MiniMax-H3 INT8 explicitly disabled; unset SGLANG_MINIMAX_H3_INT8_ENABLE or set ENABLE=1"
            )
        super().__init__(targets=targets, compressed_adaln=compressed_adaln)


class MiniMaxH3QuantLinearMethod(LinearMethodBase):
    weight_dtype = torch.int8

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
            torch.empty(rows, input_size_per_partition, dtype=self.weight_dtype),
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
        if layer.weight.dtype != self.weight_dtype:
            raise ValueError(f"MiniMax-H3 weight dtype must be {self.weight_dtype}")
        scale = layer.weight_scale_inv
        if scale.dtype != torch.float32 or scale.shape != (layer.weight.shape[0], 1):
            raise ValueError(
                "MiniMax-H3 quantization needs one FP32 scale per output channel"
            )
        if not bool(torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError(
                "MiniMax-H3 quantization scales must be finite and positive"
            )


class MiniMaxH3Int8LinearMethod(MiniMaxH3QuantLinearMethod):
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


FP8_RECIPE = (
    "main-block-only symmetric per-output-channel FP8, no rotation, no calibration"
)
FP8_FORMAT = "h3_channel_fp8_main_blocks_v1"


def inspect_fp8_checkpoint(paths):
    """Recognize the standalone exporter's full/curve-AdaLN Diffusers format."""
    from safetensors import safe_open

    roots = {Path(path).resolve().parent for path in paths}
    if len(roots) != 1:
        return None
    manifest_path = roots.pop() / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    recognized = manifest.get("recipe") == FP8_RECIPE
    explicit_variant = manifest.get("variant")
    source_component = (
        manifest.get("source_component") or Path(manifest.get("source", "")).name
    )
    if recognized:
        checkpoint_variant = explicit_variant or "fl2va"
        expected_component = {
            "fl2va": "transformer",
            "ref2va": "transformer_ref",
        }.get(checkpoint_variant)
        if expected_component is None or source_component != expected_component:
            raise ValueError(
                "MiniMax-H3 FP8 manifest variant/source component mismatch"
            )
        if explicit_variant is None and source_component != "transformer":
            raise ValueError("Only the legacy FL2VA FP8 export may omit its variant")
        profile = manifest.get("profile", "backbone")
        if profile not in ("backbone", "backbone-adaln"):
            raise ValueError("Unsupported MiniMax-H3 FP8 profile")
    tensors = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            tagged = metadata.get("recipe") == FP8_FORMAT
            if not recognized:
                if tagged:
                    raise ValueError(
                        "MiniMax-H3 FP8 requires its matching manifest.json"
                    )
                continue
            if not tagged:
                raise ValueError("Mixed MiniMax-H3 FP8 checkpoint formats")
            shard_variant = metadata.get("variant")
            if explicit_variant is not None and shard_variant != checkpoint_variant:
                raise ValueError("MiniMax-H3 FP8 shard variant mismatch")
            if explicit_variant is None and shard_variant not in (None, "fl2va"):
                raise ValueError("Invalid variant tag in legacy MiniMax-H3 FP8 shard")
            if profile == "backbone-adaln" and metadata.get("profile") != profile:
                raise ValueError("MiniMax-H3 FP8 shard AdaLN profile mismatch")
            for name in checkpoint.keys():
                if name in tensors:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                tensor = checkpoint.get_slice(name)
                tensors[name] = (tensor.get_dtype(), tensor.get_shape())
    if not recognized:
        return None
    if manifest.get("status") != "complete":
        raise ValueError("Incomplete MiniMax-H3 FP8 export")
    shards = [entry["file"] for entry in manifest.get("shards", [])]
    if len(shards) != len(set(shards)) or set(shards) != {Path(p).name for p in paths}:
        raise ValueError("MiniMax-H3 FP8 checkpoint shards do not match the manifest")
    entries = manifest.get("tensors", [])
    expected = {entry["name"]: entry for entry in entries}
    if len(expected) != len(entries):
        raise ValueError("Duplicate MiniMax-H3 FP8 manifest entries")
    targets = {
        f"transformer_blocks.{i}.{suffix}.weight"
        for i in range(50)
        for suffix in (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        )
    }
    if {name for name, entry in expected.items() if entry.get("quantized")} != targets:
        raise ValueError("MiniMax-H3 FP8 requires exactly the 300 backbone matrices")
    scales = {name.removesuffix("weight") + "weight_scale_inv" for name in targets}
    if set(tensors) != set(expected) | scales:
        raise ValueError("MiniMax-H3 FP8 tensor inventory does not match the manifest")
    for name, entry in expected.items():
        dtype, shape = tensors[name]
        if shape != entry["shape"]:
            raise ValueError(f"MiniMax-H3 FP8 tensor shape mismatch: {name}")
        if name in targets:
            scale = name.removesuffix("weight") + "weight_scale_inv"
            if (
                dtype != "F8_E4M3"
                or len(shape) != 2
                or tensors[scale] != ("F32", [shape[0], 1])
            ):
                raise ValueError(f"Invalid FP8 weight or per-channel scale: {name}")
        elif dtype not in ("BF16", "F16", "F32") or name.endswith("weight_scale_inv"):
            raise ValueError(f"Unexpected unquantized FP8 checkpoint tensor: {name}")
    if profile == "backbone-adaln":
        if tensors.get("adaln_t_table") != ("F32", [1025, 8]):
            raise ValueError("Compressed AdaLN requires an FP32 [1025,8] table")
        projections = {
            **{f"transformer_blocks.{i}.adaln_proj.linear": 96768 for i in range(50)},
            "norm_out.linear": 10752,
        }
        for prefix, rows in projections.items():
            if tensors.get(prefix + ".weight") != ("F32", [rows, 8]) or tensors.get(
                prefix + ".bias"
            ) != ("F32", [rows]):
                raise ValueError(f"Invalid compressed FP8 AdaLN projection: {prefix}")
        if any(name.startswith("time_embedder.") for name in tensors):
            raise ValueError("Compressed AdaLN must not contain the original time MLP")
    elif "adaln_t_table" in tensors:
        raise ValueError("Full AdaLN profile must not contain a compressed table")
    manifest["_validated_variant"] = checkpoint_variant
    manifest["_validated_source_component"] = source_component
    return manifest


class MiniMaxH3Fp8Config(MiniMaxH3QuantConfig):
    precision = "fp8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.float16]


class MiniMaxH3Fp8LinearMethod(MiniMaxH3QuantLinearMethod):
    weight_dtype = torch.float8_e4m3fn

    def apply(self, layer, x, bias=None):
        shape = x.shape
        rows = x.reshape(-1, shape[-1]).contiguous()
        result = linear_fp8(
            rows, layer.weight, layer.weight_scale_inv, bias, fused=self.fused
        )
        return result.reshape(*shape[:-1], layer.weight.shape[0])


def inspect_quant_checkpoint(paths, selected_variant):
    """Return config, source layout and curve shape for our validated exports."""
    manifest = inspect_int8_checkpoint(paths)
    if manifest is not None:
        if manifest["variant"] != selected_variant.lower():
            raise ValueError("MiniMax-H3 INT8 checkpoint model variant mismatch")
        compressed = manifest["profile"] == "backbone-adaln"
        config = MiniMaxH3Int8Config(
            targets=manifest["quantized_layers"], compressed_adaln=compressed
        )
        return config, False, (1025, 8) if compressed else None
    manifest = inspect_fp8_checkpoint(paths)
    if manifest is not None:
        if selected_variant.lower() != manifest["_validated_variant"]:
            raise ValueError("MiniMax-H3 FP8 checkpoint model variant mismatch")
        compressed = manifest.get("profile", "backbone") == "backbone-adaln"
        config = MiniMaxH3Fp8Config(
            targets=backbone_layers(), compressed_adaln=compressed
        )
        return config, True, (1025, 8) if compressed else None
    return None


@triton.jit
def _quant_rows_fp8(
    X, Q, S, M: tl.constexpr, K: tl.constexpr, STRIDE: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * STRIDE + col, (row < M) & (col < K), 0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-12)
    scale = amax * (1.0 / 448.0)
    q = tl.div_rn(x, scale)
    q = tl.minimum(tl.maximum(q, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * K + col, q, col < K)
    tl.store(S + row, scale)


def quant_rows_fp8(x):
    if (
        x.ndim != 2
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not x.is_cuda
        or x.stride(1) != 1
    ):
        raise ValueError("Expected GPU BF16/FP16 [M,K], contiguous last dimension")
    m, k = x.shape
    if m < 1 or k < 1 or k > 8192:
        raise ValueError("Supported domain: M>0, 0<K<=8192")
    padded_m = max(16, triton.cdiv(m, 16) * 16)
    q = torch.empty((padded_m, k), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((padded_m, 1), device=x.device, dtype=torch.float32)
    _quant_rows_fp8[(padded_m,)](
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


def linear_fp8(x, weight, scale, bias=None, *, fused=True):
    """Optimized full-row GEMM or the chunked, unfused W8A8 reference path."""
    if (
        x.ndim != 2
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.stride(1) != 1
    ):
        raise ValueError("Expected BF16/FP16 [M,K] with contiguous last dimension")
    if not x.is_cuda or weight.device != x.device or scale.device != x.device:
        raise ValueError("Expected one GPU")
    m, k = x.shape
    if (
        m < 1
        or k < 16
        or k % 16
        or weight.ndim != 2
        or weight.shape[1] != k
        or weight.shape[0] < 16
        or weight.shape[0] % 16
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
        or scale.dtype != torch.float32
        or scale.shape != (weight.shape[0], 1)
        or not scale.is_contiguous()
    ):
        raise ValueError("Unsupported FP8 GEMM weight, scale, or dimensions")
    if bias is not None and (
        bias.device != x.device
        or bias.shape != (weight.shape[0],)
        or bias.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        raise ValueError("Expected a BF16/FP16/FP32 bias vector on the same GPU")
    out = torch.empty((m, weight.shape[0]), device=x.device, dtype=x.dtype)
    weight_t, scale_w = weight.t(), scale.t()
    chunk = m if fused else 4096
    for start in range(0, m, chunk):
        part = x[start : start + chunk]
        count = part.shape[0]
        if fused and k <= 8192:
            qx, scale_x = quant_rows_fp8(part)
        else:
            r = part.float()
            scale_x = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
            qx = (r / scale_x).clamp_(-448, 448).to(torch.float8_e4m3fn)
            padding = (-count) % 16
            if padding:
                padded = torch.zeros(
                    (count + padding, k), device=x.device, dtype=x.dtype
                ).to(qx.dtype)
                padded[:count] = qx
                qx = padded
                scale_x = torch.nn.functional.pad(scale_x, (0, 0, 0, padding), value=1)
        target = out[start : start + count]
        if _is_hcu and x.dtype == torch.float16:
            # HCU hipBLASLt rowwise _scaled_mm only accepts BF16 output.
            # Preserve FP16 rounding using the existing Triton scaled GEMM.
            from sglang.kernels.ops.quantization.fp8_kernel import (
                vllm_triton_scaled_mm_fp8,
            )

            y = vllm_triton_scaled_mm_fp8(
                qx.contiguous(), weight_t, scale_x, scale, torch.float16
            )[:count]
            if bias is not None:
                y = y + bias
            target.copy_(y)
        elif fused and qx.shape[0] == count:
            torch.ops.aten._scaled_mm.out(
                qx.contiguous(),
                weight_t,
                scale_x,
                scale_w,
                None,
                None,
                x.dtype,
                False,
                out=target,
            )
            if bias is not None:
                target.add_(bias)
        else:
            y = torch._scaled_mm(
                qx.contiguous(),
                weight_t,
                scale_a=scale_x,
                scale_b=scale_w,
                out_dtype=x.dtype,
                use_fast_accum=False,
            )[:count]
            if bias is not None:
                y = y + bias
            target.copy_(y)
    return out
