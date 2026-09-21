# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 channel FP8 checkpoints and dynamic per-token Linear."""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch.nn import Parameter

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.linear import LinearMethodBase
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.minimax_h3_int8 import (
    backbone_layers,
)
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs

FP8_RECIPE = (
    "main-block-only symmetric per-output-channel FP8, no rotation, no calibration"
)
FP8_FORMAT = "h3_channel_fp8_main_blocks_v1"


def inspect_fp8_checkpoint(paths):
    """Recognize only the standalone exporter's full-AdaLN Diffusers format."""
    from safetensors import safe_open

    roots = {Path(path).resolve().parent for path in paths}
    if len(roots) != 1:
        return None
    manifest_path = roots.pop() / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    recognized = manifest.get("recipe") == FP8_RECIPE
    tensors = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            tagged = (checkpoint.metadata() or {}).get("recipe") == FP8_FORMAT
            if not recognized:
                if tagged:
                    raise ValueError(
                        "MiniMax-H3 FP8 requires its matching manifest.json"
                    )
                continue
            if not tagged:
                raise ValueError("Mixed MiniMax-H3 FP8 checkpoint formats")
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
    return manifest


class MiniMaxH3Fp8Config(QuantizationConfig):
    checkpoint_uses_native_qkv_layout = True

    def __init__(self, *, targets=None):
        if not targets:
            raise ValueError("MiniMax-H3 FP8 requires a validated checkpoint")
        self.targets = frozenset(targets)
        self.fused = envs.SGLANG_MINIMAX_H3_FP8_OPTIMIZE

    @classmethod
    def get_name(cls):
        return "minimax_h3_fp8"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_quant_method(self, layer, prefix):
        from sglang.multimodal_gen.runtime.layers.linear import (
            LinearBase,
            UnquantizedLinearMethod,
        )

        if not isinstance(layer, LinearBase):
            return None
        if prefix in self.targets:
            return MiniMaxH3Fp8LinearMethod(fused=self.fused)
        return UnquantizedLinearMethod()


class MiniMaxH3Fp8LinearMethod(LinearMethodBase):
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
            torch.empty(rows, input_size_per_partition, dtype=torch.float8_e4m3fn),
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
        scale = layer.weight_scale_inv
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise ValueError("MiniMax-H3 FP8 requires E4M3FN weights")
        if scale.dtype != torch.float32 or scale.shape != (layer.weight.shape[0], 1):
            raise ValueError("MiniMax-H3 FP8 needs one FP32 scale per output channel")
        if not bool(torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError("MiniMax-H3 FP8 scales must be finite and positive")

    def apply(self, layer, x, bias=None):
        shape = x.shape
        rows = x.reshape(-1, shape[-1]).contiguous()
        result = linear_fp8(
            rows, layer.weight, layer.weight_scale_inv, bias, fused=self.fused
        )
        return result.reshape(*shape[:-1], layer.weight.shape[0])


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
        if fused and qx.shape[0] == count:
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
