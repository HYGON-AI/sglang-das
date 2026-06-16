# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, Optional, cast
import torch
from compressed_tensors.config import SparsityCompressionConfig
from compressed_tensors.quantization import QuantizationArgs
import logging

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig, CompressedTensorsLinearMethod, CompressedTensorsKVCacheMethod
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_moe_marlin import CompressedTensorsMarlinMoEMethod
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
import os

# if TYPE_CHECKING:
#     from vllm.model_executor.models.utils import WeightsMapper

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsLinearMethod"]

SPARSITY_CONFIG_NAME: Literal["sparsity_config"] = "sparsity_config"
QUANTIZATION_SCHEME_MAP_TYPE = dict[str, Optional[dict[str, QuantizationArgs]]]


class SlimQuantCompressedTensorsMarlinConfig(CompressedTensorsConfig):
    def __init__(
        self,
        target_scheme_map: dict[str, Any],
        ignore: list[str],
        quant_format: str,
        sparsity_scheme_map: dict[str, SparsityCompressionConfig],
        sparsity_ignore_list: list[str],
        kv_cache_scheme: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
        packed_modules_mapping: Optional[dict[str, list[str]]] = None,
        linear_fp8_config: Optional[Any] = None,
    ):
        super().__init__(
            target_scheme_map,
            ignore,
            quant_format,
            sparsity_scheme_map,
            sparsity_ignore_list,
            kv_cache_scheme,
            config,
            packed_modules_mapping,
            linear_fp8_config,
        )


    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if hf_quant_cfg.get("quant_method") == "compressed-tensors" \
                and user_quant == "slimquant_marlin":
            return cls.get_name()
        return None
    @classmethod
    def get_name(cls) -> str:
        return "slimquant_marlin"

    def get_quant_method(
            self,
            layer: torch.nn.Module,
            prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # Avoid circular import
        from sglang.srt.layers.radix_attention import RadixAttention
        # Check if the layer is skipped for quantization.
        if isinstance(layer, RadixAttention):
            return CompressedTensorsKVCacheMethod(self)
        if should_ignore_layer(prefix,
                               ignore=self.ignore,
                               fused_mapping=self.packed_modules_mapping):
            return UnquantizedEmbeddingMethod()#UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            scheme = self.get_linear_scheme(layer=layer, layer_name=prefix)
            if scheme is None:
                return UnquantizedEmbeddingMethod()#UnquantizedLinearMethod()
            layer.scheme = scheme
            return CompressedTensorsLinearMethod(self)
        if isinstance(layer, FusedMoE):
            return CompressedTensorsMarlinMoEMethod.get_moe_method(self, layer)
        return None
