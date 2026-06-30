from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmFp8MoeQuantInfo,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
    get_moe_weight_sizes,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz, scaled_fp8_quant
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.layers.quantization.utils import (
    all_close_1d,
    per_tensor_dequantize,
    swap_w13_to_w31,
)
from sglang.srt.utils import get_bool_env_var, is_hip, is_dcu, set_weight_attrs
from sglang.srt.server_args import get_global_server_args
if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

__all__ = ["CompressedTensorsW8A8Fp8MoE"]

_is_hip = is_hip()
_is_dcu = is_dcu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_use_fp8_w8a8_moe = get_bool_env_var("SGLANG_USE_FP8_W8A8_MOE")
_use_deepgemm_moe = get_bool_env_var("SGLANG_USE_DEEPGEMM_MOE")
_use_aiter_fp8_w8a8_moe = get_bool_env_var("SGLANG_ROCM_USE_AITER_MOE")
if _use_aiter_fp8_w8a8_moe:
    import aiter
    from aiter.moe import (
        get_aiter_moe_config,
        aiter_moe,
        MoeSolutionType,
        MoeQuantType,
            )
if _use_aiter and not _is_dcu:
    from aiter.ops.shuffle import shuffle_weight


logger = logging.getLogger(__name__)

def is_moe_prefill_or_normal():
    args = get_global_server_args()
    return (args.disaggregation_mode == "prefill" or args.deepep_mode == "normal")

class CompressedTensorsW8A8Fp8MoE(CompressedTensorsMoEScheme):

    def __init__(self, weight_quant, input_quant):
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.use_flashinfer_trtllm = get_moe_runner_backend().is_flashinfer_trtllm()

        per_tensor = (
            self.weight_quant.strategy == QuantizationStrategy.TENSOR
            and self.input_quant.strategy == QuantizationStrategy.TENSOR
        )
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not (per_tensor or per_channel):
            assert self.weight_quant.strategy == QuantizationStrategy.BLOCK
            self.weight_block_size = self.weight_quant.block_structure
            assert self.weight_quant.dynamic is not None
        else:
            self.weight_block_size = None
        self.block_quant = self.weight_block_size is not None
        self.use_deepep = get_moe_a2a_backend().is_deepep()

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales and per_channel:
            raise ValueError(
                "For FP8 Fused MoE layer, we require either per tensor or "
                "channelwise, dynamic per token quantization."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        params_dtype = torch.float8_e4m3fn

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
            intermediate_size_per_partition,
            is_aiter_moe=_use_aiter,
            is_concat=True,
            is_packed=False,
        )

        extra_weight_attrs.update(
            {"weight_padded": weight_padded},
        )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                w2_down_dim,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # per-tensor quantization
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            weight_quant_method = FusedMoeWeightScaleSupported.TENSOR.value
        elif self.weight_quant.strategy == QuantizationStrategy.CHANNEL:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    w13_up_dim,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
                requires_grad=False,
            )
            weight_quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
        elif self.weight_quant.strategy == QuantizationStrategy.BLOCK:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            weight_quant_method = FusedMoeWeightScaleSupported.BLOCK.value
        else:
            raise ValueError(
                f"Unsupported weight quantization strategy: {self.weight_quant.strategy}"
            )

        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update({"quant_method": weight_quant_method})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            assert (
                self.input_quant.strategy == QuantizationStrategy.TENSOR
            ), "Only per-tensor quantization is supported for static input scales"
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    @staticmethod
    def _register_runtime_buffer(
        layer: torch.nn.Module, name: str, value: torch.Tensor
    ) -> None:
        if name in layer._buffers:
            layer._buffers[name] = value
            layer._non_persistent_buffers_set.add(name)
            return

        if hasattr(layer, name):
            delattr(layer, name)
        layer.register_buffer(name, value, persistent=False)

    def _prepare_dsv4_channel_fp8_deepgemm_weights(
        self, layer: torch.nn.Module
    ) -> None:
        if getattr(layer, "_dsv4_channel_fp8_deepgemm_repacked", False):
            return

        w13 = layer.w13_weight
        w2 = layer.w2_weight

        from deepgemm.m_group_gemm import pack_int8_weight_enk_to_w6_low_latency

        with torch.no_grad():
            w13_deepgemm = pack_int8_weight_enk_to_w6_low_latency(w13).detach()
            w2_deepgemm = pack_int8_weight_enk_to_w6_low_latency(w2).detach()

        self._register_runtime_buffer(layer, "w13_weight_deepgemm", w13_deepgemm)
        self._register_runtime_buffer(layer, "w2_weight_deepgemm", w2_deepgemm)
        layer._dsv4_channel_fp8_deepgemm_repacked = True

        # Save original weight shapes before deletion — needed by
        # DeepEPMoE.forward_groupgemm_w8a8_fp8_contiguous for N dimension
        layer._dsv4_w13_weight_shape = tuple(w13.shape)
        layer._dsv4_w2_weight_shape = tuple(w2.shape)

        # Clean up weights that won't be used
        del layer.w13_weight
        del layer.w2_weight
        torch.cuda.empty_cache()

    def process_weights_after_loading(self, layer: torch.nn.Module | FusedMoE) -> None:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.static_input_scales:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(
                layer.w2_input_scale
            ):
                logger.warning(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer."
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        if is_fp8_fnuz():
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = (
                normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                )
            )
            w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
            )
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(
                    w13_input_scale, requires_grad=False
                )
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(
                    w2_input_scale, requires_grad=False
                )
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.num_local_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    (
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        _,
                    ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])

                    start += shard_size

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )

        if self.weight_quant.strategy == QuantizationStrategy.CHANNEL and _use_aiter:
            with torch.no_grad():
                # Pre-shuffle weights
                layer.w13_weight = torch.nn.Parameter(
                    shuffle_weight(layer.w13_weight.data, (16, 16)),
                    requires_grad=False,
                )
                torch.cuda.empty_cache()
                layer.w2_weight = torch.nn.Parameter(
                    shuffle_weight(layer.w2_weight.data, (16, 16)),
                    requires_grad=False,
                )
                torch.cuda.empty_cache()
        if self.weight_quant.strategy == QuantizationStrategy.CHANNEL and _use_deepgemm_moe and _is_dcu:
            self._prepare_dsv4_channel_fp8_deepgemm_weights(layer)

        elif (_use_fp8_w8a8_moe and _is_dcu
            and not getattr(layer, "_w8a8_fp8_packed", False)):
            w1 = layer.w13_weight
            w2 = layer.w2_weight
            w1_shape = w1.shape
            w2_shape = w2.shape
            if (w1.is_cuda and w2.is_cuda):
                if w1.dim() != 3 or w2.dim() != 3 or w1.size(0) != w2.size(
                    0):
                    raise RuntimeError("Unexpected MoE weight shapes")
                twoN, K = w1.size(1), w1.size(2)
                if w2.size(1) != K:
                    raise RuntimeError("Unexpected MoE w2 layout")
                N = w2.size(2)
                if twoN != 2 * N:
                    raise RuntimeError("Unexpected MoE hidden dims")
                if (K % 16 != 0 or K % 32 != 0 or N % 16 != 0
                    or twoN % 32 != 0):
                    raise RuntimeError("Marlin packing requires alignment")

                from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
                    w8a8_2_marlin_weight, weight8bit_nt_kpack2_marlin,weight8bit_nt_kpack2_marlin1)
                from torch.nn.parameter import Parameter

                def _pack_per_expert(weight: torch.Tensor) -> torch.Tensor:
                    num_experts = weight.shape[0]
                    for i in range(num_experts):
                        new_expert = w8a8_2_marlin_weight(
                            weight[i]).contiguous()
                        weight.data[i].view(-1).copy_(new_expert.view(-1))
                    weight = weight.reshape((-1,) + new_expert.shape)

                    return weight

                def _pack_per_expert_deepep(weight: torch.Tensor) -> torch.Tensor:
                    num_experts = weight.shape[0]
                    for i in range(num_experts):
                        if is_moe_prefill_or_normal():
                            new_expert = weight8bit_nt_kpack2_marlin(
                                weight[i]).contiguous()
                        else:
                            new_expert = weight8bit_nt_kpack2_marlin1(
                                weight[i]).contiguous()
                        weight.data[i].view(-1).copy_(new_expert.view(-1))
                    weight = weight.reshape((-1,) + new_expert.shape)
                    return weight

                with torch.no_grad():
                    if self.use_deepep:
                        w1_packed = _pack_per_expert_deepep(w1)
                        w2_packed = _pack_per_expert_deepep(w2)
                    else:
                        w1_packed = _pack_per_expert(w1)
                        w2_packed = _pack_per_expert(w2)

                    new_w1 = Parameter(w1_packed, requires_grad=False)
                    new_w2 = Parameter(w2_packed, requires_grad=False)

                    if hasattr(w1, "__dict__"):
                        for k, v in w1.__dict__.items():
                            setattr(new_w1, k, v)
                    if hasattr(w2, "__dict__"):
                        for k, v in w2.__dict__.items():
                            setattr(new_w2, k, v)

                    setattr(new_w1, "_w8a8_fp8_packed", True)
                    setattr(new_w1, "w1_shape", w1_shape)
                    setattr(new_w2, "_w8a8_fp8_packed", True)
                    setattr(new_w2, "w2_shape", w2_shape)

                    layer.w13_weight = new_w1
                    layer.w2_weight = new_w2
                    layer._w8a8_fp8_packed = True

        if (
            self.weight_quant.strategy == QuantizationStrategy.BLOCK
            and self.use_flashinfer_trtllm
        ):
            layer.w13_weight = torch.nn.Parameter(
                swap_w13_to_w31(layer.w13_weight.data),
                requires_grad=False,
            )
            layer.w13_weight_scale = torch.nn.Parameter(
                swap_w13_to_w31(layer.w13_weight_scale.data),
                requires_grad=False,
            )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_auto():
            if (
                _use_aiter
                and self.weight_quant.strategy == QuantizationStrategy.CHANNEL
                and get_moe_a2a_backend().is_none()
            ):
                moe_runner_backend = MoeRunnerBackend.AITER
            else:
                moe_runner_backend = MoeRunnerBackend.TRITON

        if (
            moe_runner_backend.is_aiter()
            or moe_runner_backend.is_triton()
            or moe_runner_backend.is_flashinfer_trtllm()
            or moe_runner_backend.is_flashinfer_trtllm_routed()
        ):
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
        else:
            # TODO(cwan): refactor other backends
            pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
        bias: Optional[torch.Tensor] = None,
        i_q: Optional[torch.Tensor] = None,
        i_s: Optional[torch.Tensor] = None,
    ) -> CombineInput:

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        if (
            _is_dcu
            and self.runner.runner_backend.is_aiter()
            and self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        ):
            from sglang.srt.layers.moe.moe_runner.aiter import (
                get_aiter_w8a8_fp8_quant_info,
            )

            assert not moe_runner_config.no_combine, "unsupported"
            quant_info = get_aiter_w8a8_fp8_quant_info(layer)
            combine_input = self.runner.run(dispatch_output, quant_info)
            if bias is not None:
                from sglang.srt.layers.moe.token_dispatcher import (
                    StandardCombineInput,
                )

                return StandardCombineInput(
                    hidden_states=combine_input.hidden_states + bias
                )
            return combine_input
        elif self.runner.runner_backend.is_aiter():
            from sglang.srt.layers.moe.moe_runner.aiter import (
                AiterMoeQuantInfo,
                AiterQuantType,
            )

            assert not moe_runner_config.no_combine, "unsupported"
            quant_info = AiterMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                quant_type=AiterQuantType.PER_TOKEN,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.weight_quant.strategy == QuantizationStrategy.BLOCK:
            if self.use_flashinfer_trtllm:
                from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                    get_activation_type,
                )

                activation_type = get_activation_type(moe_runner_config.activation)
                quant_info = FlashInferTrtllmFp8MoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    global_num_experts=layer.num_experts,
                    local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                    local_num_experts=layer.num_local_experts,
                    intermediate_size=layer.w2_weight.shape[2],
                    routing_method_type=layer.routing_method_type,
                    block_quant=self.block_quant,
                    weight_block_k=self.weight_block_size[1],
                    w13_weight_scale_inv=layer.w13_weight_scale,
                    w2_weight_scale_inv=layer.w2_weight_scale,
                    activation_type=activation_type,
                )
            else:
                quant_info = TritonMoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    use_fp8_w8a8=True,
                    w13_scale=layer.w13_weight_scale,
                    w2_scale=layer.w2_weight_scale,
                    a13_scale=layer.w13_input_scale,
                    a2_scale=layer.w2_input_scale,
                    block_shape=self.weight_block_size,
                )
            return self.runner.run(dispatch_output, quant_info)
        elif _is_dcu and _use_fp8_w8a8_moe:
            if (getattr(layer.w13_weight, "_w8a8_fp8_packed", False)
                or getattr(layer.w2_weight, "_w8a8_fp8_packed", False)):
                topk_weights, topk_ids, _ = topk_output
                use_prequant_input = i_q is not None and i_s is not None
                if moe_runner_config.apply_router_weight_on_input:
                    assert topk_weights.dim() == 2, "`topk_weights` should be (num_tokens, topk)"
                    _, tk = topk_weights.shape
                    assert tk == 1, "DCU marlin path: apply_router_weight_on_input requires topk=1"
                    x = x * topk_weights.to(x.dtype)
                    topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)
                    # Router-weighted input no longer matches precomputed rms-quant activations.
                    use_prequant_input = False
                # from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe_fp8_w8a8
                from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe_fp8_w8a8

                origin_w1_shape = getattr(layer.w13_weight, "w1_shape", None)
                origin_w2_shape = getattr(layer.w2_weight, "w2_shape", None)
                output = fused_moe_fp8_w8a8(
                    hidden_states=x,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    w1_scale=layer.w13_weight_scale,
                    w2_scale=layer.w2_weight_scale,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    global_num_experts=self.moe_runner_config.num_experts,
                    inplace=True,
                    origin_w1_shape=origin_w1_shape,
                    origin_w2_shape=origin_w2_shape,
                    routed_scaling_factor=moe_runner_config.routed_scaling_factor,
                    bias=bias,
                    hidden_states_fp8_input=i_q if use_prequant_input else None,
                    hidden_states_scale_fp8_input=i_s if use_prequant_input else None,
                )
                from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

                return StandardCombineInput(hidden_states=output)
        elif  _is_dcu and not _use_fp8_w8a8_moe and _use_aiter_fp8_w8a8_moe:
            if isinstance(layer.w13_weight,tuple):
                w1=layer.w13_weight[0]
                w2=layer.w2_weight[0]
            else:
                w1=layer.w13_weight
                w2=layer.w2_weight
            topk_weights, topk_ids, _ = topk_output
            if moe_runner_config.apply_router_weight_on_input:
                assert topk_weights.dim() == 2, "`topk_weights` should be (num_tokens, topk)"
                _, tk = topk_weights.shape
                assert tk == 1, "DCU marlin path: apply_router_weight_on_input requires topk=1"
                x = x * topk_weights.to(x.dtype)
                topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)
                # Router-weighted input no longer matches precomputed rms-quant activations.
                use_prequant_input = False
            if isinstance(layer.w13_weight_scale, tuple):
                w1_scale=layer.w13_weight_scale[0]
                w2_scale=layer.w2_weight_scale[0]
            else:
                w1_scale=layer.w13_weight_scale
                w2_scale=layer.w2_weight_scale
            M, K = x.shape 
            E = self.moe_runner_config.num_experts
            top_k = topk_ids.shape[1]
            N1 = w1.shape[1]
            N2 = w2.shape[1]
            activation="silu" if moe_runner_config.activation == "silu" else "gelu"
            # -----------------------------------------------------------------
            # Query backend config
            # ------------------------------------------------------------------
            status, moe_cfg = get_aiter_moe_config(
                M=M,
                E=E,
                N1=N1,
                N2=N2,
                K=K,
                top_k=top_k,
                block_size=0,
                dtype=x.dtype,
                quant_type=MoeQuantType.FP8_W8A8,
            )
            if not status:
                raise RuntimeError(
                "[aiter_moe_fp8_w8a8] no suitable backend found: "
                f"M={M}, N1={N1}, N2={N2}, K={K}, "
                f"E={E}, topk={top_k}")
            if moe_cfg.solution_type not in {
                MoeSolutionType.MOE_C,
                MoeSolutionType.ASM,
                MoeSolutionType.TRITON,
                MoeSolutionType.CK,
            }:
                raise RuntimeError(
                    f"Unsupported solution_type: {moe_cfg.solution_type}"
                )
            # for debug：
            # aiter.logger.info(f"moe_cfg.solution_type {moe_cfg.solution_type}")
            #在 moe_c 后端需要对weight做shuffle，如果不做此 shuffle 直接传入 moe_c_fused_experts，
            #底层 Marlin/CUDA kernel 会按错误的内存偏移读取权重，导致结果错误。
            if moe_cfg.solution_type==MoeSolutionType.MOE_C:
                from aiter.ops.shuffle import moe_layout_shuffle_gemm1,moe_layout_shuffle_gemm2
                if not getattr(layer, "_moec_shuffled", False):
                    layer.w13_weight.data = moe_layout_shuffle_gemm1(layer.w13_weight.data).view(*layer.w13_weight.data.shape)
                    layer.w2_weight.data = moe_layout_shuffle_gemm2(layer.w2_weight.data).view(*layer.w2_weight.data.shape)
                    layer._moec_shuffled = True
                w1 = layer.w13_weight
                w2 = layer.w2_weight
            if moe_cfg.quant_type != MoeQuantType.FP8_W8A8:
                raise RuntimeError(
                    f"Unexpected quant_type: {moe_cfg.quant_type}"
                )
            if moe_runner_config.routed_scaling_factor is None:
                moe_runner_config.routed_scaling_factor=1.0
            output=aiter_moe(
                hidden_states=x,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                moe_config=moe_cfg,
                inplace=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                activation=activation,
                block_shape=None,
                global_num_experts=E,
                routed_scaling_factor=moe_runner_config.routed_scaling_factor,
            )
            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

            return StandardCombineInput(hidden_states=output)
        else:
            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_fp8_w8a8=True,
                per_channel_quant=self.weight_quant.strategy
                == QuantizationStrategy.CHANNEL,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
            return self.runner.run(dispatch_output, quant_info)
