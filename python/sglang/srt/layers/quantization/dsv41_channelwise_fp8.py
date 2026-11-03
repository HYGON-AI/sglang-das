"""Automatically selected HCU channelwise FP8 projections for DeepSeek V4.1.

Weights stay in checkpoint row-major FP8 layout. Activations are quantized
per token; DeepGEMM consumes FP8 operands and FP32 row scales directly.

The default requires all three special projections in a local safetensors
checkpoint to have E4M3 weights and channel scales. Set
SGLANG_DSV41_CHANNELWISE_FP8_RUNTIME=0 to retain the previous paths, or =1
to explicitly enable this method with custom checkpoint loaders.
LM-head LoRA is not supported by this method.
"""

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.quantization.dsv41_channelwise_fp8_checkpoint import (
    checkpoint_supports_channelwise_fp8,
)


def configure_checkpoint(quant_config, hf_config):
    """Keep separate target/draft models bound to their own checkpoint."""
    if quant_config is not None:
        quant_config._dsv41_channelwise_fp8_checkpoint = getattr(
            hf_config, "_name_or_path", None
        )


def enabled(quant_config, *, model_path=None):
    mode = envs.SGLANG_DSV41_CHANNELWISE_FP8_RUNTIME.get()
    if mode == "0":
        return False
    if mode not in ("auto", "1"):
        raise ValueError("SGLANG_DSV41_CHANNELWISE_FP8_RUNTIME must be auto, 0 or 1")
    from sglang.srt.utils import is_hcu

    supported = False
    if quant_config is not None and quant_config.get_name() in (
        "compressed-tensors",
        "compressed_tensors",
    ):
        # DeepseekV41Config translates is_per_channel FP8 checkpoints to this
        # scheme before model construction. Match the normalized config.
        scheme = quant_config.target_scheme_map.get("Linear", {})
        weight = scheme.get("weights")
        activation = scheme.get("input_activations")
        supported = (
            quant_config.quant_format == "float-quantized"
            and weight is not None
            and activation is not None
            and weight.num_bits == 8
            and weight.type == "float"
            and weight.strategy == "channel"
            and not weight.dynamic
            and activation.num_bits == 8
            and activation.type == "float"
            and activation.strategy == "token"
            and activation.dynamic
        )
    elif quant_config is not None and quant_config.get_name() == "fp8":
        supported = (
            quant_config.is_checkpoint_fp8_serialized
            and not quant_config.weight_block_size
            and not quant_config.use_mxfp8
            and quant_config.activation_scheme == "dynamic"
        )
    if not (is_hcu() and supported):
        if mode == "auto":
            return False
        raise ValueError(
            "DSV4.1 channelwise FP8 runtime requires serialized HCU channelwise FP8"
        )
    if mode == "1":
        return True
    if model_path is None:
        model_path = getattr(quant_config, "_dsv41_channelwise_fp8_checkpoint", None)
    if not model_path:
        return False
    return checkpoint_supports_channelwise_fp8(str(model_path))


def linear(x, weight, scale, *, output_dtype=None):
    import deepgemm
    from lightop.quant import per_token_quant_fp8

    if weight.dtype != torch.float8_e4m3fn or scale.shape != weight.shape[:-1] + (1,):
        raise ValueError(
            "Expected row-major e4m3fn weights and one scale per output channel"
        )
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError("FP8 projection input/weight dimensions differ")
    dtype = output_dtype or x.dtype
    output = torch.empty(
        (*x.shape[:-1], weight.shape[-2]), device=x.device, dtype=dtype
    )
    if x.numel() == 0:
        return output
    x = x.contiguous()
    xq, xs = per_token_quant_fp8(x)
    # The installed hipBLASLt dispatch is extremely slow for full-vocabulary
    # N=129280. Tile output channels within this device, without changing TP.
    tiles = (weight.shape[-2] + 16383) // 16384
    if weight.ndim == 2 and tiles > 1 and weight.shape[0] % tiles == 0:
        n = weight.shape[0] // tiles
        tiled = torch.empty((tiles, x.shape[0], n), device=x.device, dtype=dtype)
        if dtype == torch.float32:
            for tile in range(tiles):
                deepgemm.fp8_gemm(
                    (xq, xs),
                    (
                        weight[tile * n : (tile + 1) * n],
                        scale[tile * n : (tile + 1) * n],
                    ),
                    tiled[tile],
                )
        else:
            deepgemm.fp8_gemm(
                (
                    xq.unsqueeze(0).expand(tiles, -1, -1).contiguous(),
                    xs.unsqueeze(0).expand(tiles, -1, -1).contiguous(),
                ),
                (weight.view(tiles, n, -1), scale.view(tiles, n, 1)),
                tiled,
            )
        return tiled.transpose(0, 1).reshape(x.shape[0], weight.shape[0])
    # DTK 2604's batched GEMM gives incorrect FP32 outputs for batch > 1.
    # Individual FP32 GEMMs and batched BF16 GEMMs pass the same oracle.
    if output.dtype == torch.float32 and weight.ndim == 3:
        for group in range(weight.shape[0]):
            deepgemm.fp8_gemm(
                (xq[group], xs[group]),
                (weight[group], scale[group]),
                output[group],
            )
    else:
        deepgemm.fp8_gemm((xq, xs), (weight, scale), output)
    return output


def grouped(x, layer, groups, rank):
    weight = layer.weight.view(groups, rank, -1)
    scale = layer.weight_scale_inv.view(groups, rank, 1)
    y = linear(x.transpose(0, 1).contiguous(), weight, scale)
    return y.transpose(0, 1).contiguous()


class ChannelwiseFp8Method(QuantizeMethodBase):
    supports_lm_head_lora = False

    def process_weights_after_loading(self, layer):
        for name in ("weight", "weight_scale_inv"):
            if not getattr(getattr(layer, name), "_channelwise_loaded", False):
                raise ValueError(f"Missing channelwise FP8 {name}")

    def apply(self, layer, x, bias=None):
        return self.apply_with_output_dtype(layer, x, bias, output_dtype=x.dtype)

    def apply_with_output_dtype(self, layer, x, bias=None, *, output_dtype):
        """Honor FP32 logits without converting the FP8 weight to FP32."""
        original_shape = x.shape
        y = linear(
            x.reshape(-1, original_shape[-1]),
            layer.weight,
            layer.weight_scale_inv,
            output_dtype=output_dtype,
        )
        if bias is not None:
            y = y + bias
        return y.view(*original_shape[:-1], y.shape[-1])


def install(layer):
    """Retain the layer's existing TP/padded-vocabulary loader geometry."""
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    old_weight = layer.weight
    loader = getattr(old_weight, "weight_loader", default_weight_loader)
    output_dim = getattr(old_weight, "output_dim", None)
    for name, shape, dtype in (
        ("weight", old_weight.shape, torch.float8_e4m3fn),
        ("weight_scale_inv", (old_weight.shape[0], 1), torch.float32),
    ):
        param = nn.Parameter(
            torch.empty(shape, device=old_weight.device, dtype=dtype),
            requires_grad=False,
        )
        if output_dim is not None:
            param.output_dim = output_dim
        if name == "weight":
            param.input_dim = 1

        def load(param, value, *, _loader=loader, _name=name):
            if _name == "weight" and value.dtype != torch.float8_e4m3fn:
                raise ValueError(
                    "Channelwise FP8 runtime requires an FP8 checkpoint weight"
                )
            if _name == "weight_scale_inv" and (value.ndim != 2 or value.shape[1] != 1):
                raise ValueError("Channelwise FP8 runtime requires [N, 1] scales")
            _loader(param, value)
            param._channelwise_loaded = True

        param.weight_loader = load
        setattr(layer, name, param)
    layer.quant_method = ChannelwiseFp8Method()
