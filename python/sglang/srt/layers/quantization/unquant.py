from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.layers.amx_utils import (
    CPUQuantMethod,
    _amx_process_weight_after_loading,
)
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
    get_moe_a2a_backend
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.utils import MultiPlatformOp, copy_or_rebind_param
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_hip,
    is_npu,
    next_power_of_2,
    set_weight_attrs,
    use_intel_amx_backend,
    use_intel_xpu_backend,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DispatchOutput,
        StandardDispatchOutput,
    )
from sglang.srt.utils import is_dcu
_is_dcu = is_dcu()

_use_marlin_w16a16_moe = get_bool_env_var("SGLANG_USE_MARLIN_W16A16_MOE")
_use_aiter_w16a16_moe = get_bool_env_var("SGLANG_ROCM_USE_AITER_MOE")
if _use_aiter_w16a16_moe :
    import aiter
    from aiter.moe import (
        get_aiter_moe_config,
        aiter_moe,
        MoeSolutionType,
        MoeQuantType,
            )

_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.tuned_gemm import tgemm

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import npu_format_cast

try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType
except ImportError:
    flashinfer_cutlass_fused_moe = None


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """Unquantized method for embeddings."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for embedding layer."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["weight"])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_intel_amx_backend(layer):
            x_shapes = x.shape
            if len(x_shapes) == 3:
                x = x.view(-1, x.shape[-1])
            output = torch.ops.sgl_kernel.weight_packed_linear(
                x,
                layer.weight,
                bias,
                True,  # is_vnni
            )
            if len(x_shapes) == 3:
                output = output.view(x_shapes[0], x_shapes[1], -1)
            return output

        elif _use_aiter and type(layer.weight.data) is torch.Tensor:
            return tgemm.mm(x, layer.weight, bias, otype=x.dtype)

        if x.dtype != layer.weight.dtype:
            x = x.to(layer.weight.dtype)
        return F.linear(x, layer.weight, bias)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, MultiPlatformOp):
    """MoE method without quantization."""

    def __init__(
        self,
        use_triton_kernels: bool = False,
        use_flashinfer_trtllm_moe: bool = False,
        use_deep_gemm: bool = False,
    ):
        super().__init__()
        self.use_flashinfer_cutlass = get_moe_runner_backend().is_flashinfer_cutlass()
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = False
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        self.use_deep_gemm = use_deep_gemm
        self._cache_permute_indices = dict({})
        self.use_deepep = get_moe_a2a_backend().is_deepep()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.with_bias = with_bias

        # Fused gate_up_proj (column parallel)
        w13_up_dim = (
            2 * intermediate_size_per_partition
            if layer.moe_runner_config.is_gated
            else intermediate_size_per_partition
        )
        w13_weight_n, w13_weight_k = (w13_up_dim, hidden_size)
        if self.use_triton_kernels:
            w13_weight_n, w13_weight_k = w13_weight_k, w13_weight_n
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_weight_n, w13_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        if self.with_bias:
            w13_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, w13_up_dim, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_bias", w13_weight_bias)
            set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight_n, w2_weight_k = (
            hidden_size,
            intermediate_size_per_partition,
        )
        if self.use_triton_kernels:
            w2_weight_n, w2_weight_k = w2_weight_k, w2_weight_n
        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, w2_weight_n, w2_weight_k, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.with_bias:
            w2_weight_bias = torch.nn.Parameter(
                torch.empty(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_bias", w2_weight_bias)
            set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _should_use_aiter_moe = _use_aiter and (
            get_moe_runner_backend().is_auto() or get_moe_runner_backend().is_aiter()
        )
        if _should_use_aiter_moe:
            copy_or_rebind_param(
                layer, "w13_weight", shuffle_weight(layer.w13_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()
            copy_or_rebind_param(
                layer, "w2_weight", shuffle_weight(layer.w2_weight.data, (16, 16))
            )
            torch.cuda.empty_cache()

        # Pack weight for get better performance on CPU
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])

        if (
            self.use_deep_gemm
            and layer.w13_weight.dtype == torch.bfloat16
            and get_moe_a2a_backend().is_deepep()
            and get_deepep_mode().enable_low_latency()
            and not _is_npu
            and not _is_hip
            and hasattr(layer, "dispatcher")
        ):
            layer.dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})

        # Reorder rows of W1 for fused gated activation
        if self.use_flashinfer_trtllm_moe:
            from flashinfer.fused_moe.core import (
                _maybe_get_cached_w3_w1_permute_indices,
                convert_to_block_layout,
                get_w2_permute_indices_with_cache,
            )

            # w1 and w3 have been swapped, so we don't need do that here
            epilogue_tile_m = 128
            block_k = 128
            old_shape_w13 = layer.w13_weight.data[0].shape
            old_shape_w2 = layer.w2_weight.data[0].shape
            new_shape_w13 = None
            new_shape_w2 = None
            for i in range(layer.num_local_experts):
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                    self._cache_permute_indices,
                    layer.w13_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights1 = (
                    layer.w13_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w13_weight.data.device)]
                    .contiguous()
                )

                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    layer.w2_weight.data[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights2 = (
                    layer.w2_weight.data[i]
                    .clone()
                    .view(torch.uint8)[permute_indices.to(layer.w2_weight.data.device)]
                    .contiguous()
                )

                tmp_weights1 = convert_to_block_layout(
                    tmp_weights1.view(torch.uint8), block_k
                )
                tmp_weights2 = convert_to_block_layout(
                    tmp_weights2.view(torch.uint8), block_k
                )

                new_shape_w13 = tmp_weights1.view(torch.bfloat16).shape
                new_shape_w2 = tmp_weights2.view(torch.bfloat16).shape
                layer.w13_weight.data[i] = (
                    tmp_weights1.view(torch.bfloat16)
                    .contiguous()
                    .reshape(old_shape_w13)
                )
                layer.w2_weight.data[i] = (
                    tmp_weights2.view(torch.bfloat16).contiguous().reshape(old_shape_w2)
                )

            layer.w13_weight.data = layer.w13_weight.data.reshape(
                layer.num_local_experts, *new_shape_w13
            )
            layer.w2_weight.data = layer.w2_weight.data.reshape(
                layer.num_local_experts, *new_shape_w2
            )
        if (_is_dcu and _use_marlin_w16a16_moe and not _use_aiter_w16a16_moe
            and not self.use_deepep
            and not getattr(layer, "use_nn_moe", False)
            and not getattr(layer, "_marlin_w16a16_moe_packed", False)):
            w1 = layer.w13_weight
            w2 = layer.w2_weight
            N = w1.shape[1]
            if (w1.is_cuda and w2.is_cuda
                    and w1.dtype in (torch.float16, torch.bfloat16)
                    and w2.dtype in (torch.float16, torch.bfloat16)):
                try:
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
                        w16a16_marlin_weight)
                    from torch.nn.parameter import Parameter

                    # def _pack_per_expert(weight: torch.Tensor) -> torch.Tensor:
                    #     num_experts = weight.shape[0]
                    #     packed0 = w16a16_marlin_weight(
                    #         weight[0]).contiguous()
                    #     packed = packed0.new_empty((num_experts, ) +
                    #                                packed0.shape)
                    #     packed[0].copy_(packed0)
                    #     del packed0
                    #     for i in range(1, num_experts):
                    #         tmp = w16a16_marlin_weight(
                    #             weight[i]).contiguous()
                    #         packed[i].copy_(tmp)
                    #         del tmp
                    #     return packed
                    def _pack_per_expert(weight: torch.Tensor) -> torch.Tensor:
                        num_experts = weight.shape[0]
                        for i in range(num_experts):
                            new_expert = w16a16_marlin_weight(
                                weight[i]).contiguous()
                            weight.data[i].view(-1).copy_(new_expert.view(-1))
                        weight = weight.reshape((-1,) + new_expert.shape)
                        return weight

                    with torch.no_grad():
                        w1_packed = _pack_per_expert(w1)
                        w2_packed = _pack_per_expert(w2)

                        new_w1 = Parameter(w1_packed, requires_grad=False)
                        new_w2 = Parameter(w2_packed, requires_grad=False)

                        # Preserve any custom weight attributes (e.g. loaders).
                        if hasattr(w1, "__dict__"):
                            for k, v in w1.__dict__.items():
                                setattr(new_w1, k, v)
                        if hasattr(w2, "__dict__"):
                            for k, v in w2.__dict__.items():
                                setattr(new_w2, k, v)

                        setattr(new_w1, "marlin_w16a16_packed", True)
                        setattr(new_w1, "N", N)
                        setattr(new_w2, "marlin_w16a16_packed", True)

                        layer.w13_weight = new_w1
                        layer.w2_weight = new_w2
                        layer._marlin_w16a16_moe_packed = True
                        return
                except Exception:
                    # If packing dependencies are unavailable, fall back to the
                    # standard (non-Marlin) layouts.
                    pass
        if _is_npu:
            for weight_name in ["w13_weight", "w2_weight"]:
                weight = getattr(layer, weight_name)
                origin_weight = weight.data.transpose(1, 2)
                new_weight = origin_weight.contiguous()
                origin_weight.untyped_storage().resize_(0)
                weight.data = npu_format_cast(new_weight)

        return

    def maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
        self,
        layer: torch.nn.Module,
        param: torch.nn.Parameter,
        weight_name: str,
    ) -> None:
        """Restore canonical BF16 MoE load shapes before hot weight copy.

        The flashinfer TRT-LLM BF16 postprocess reshapes expert weights into
        block layout. During weight update, checkpoint tensors are in
        canonical layout and need a temporary shape restore for copy.
        """
        if not get_moe_runner_backend().is_flashinfer_trtllm_routed():
            return

        expected_shape = None
        if weight_name.endswith(".experts.w13_weight"):
            w13_rows = (
                2 * layer.intermediate_size_per_partition
                if layer.moe_runner_config.is_gated
                else layer.intermediate_size_per_partition
            )
            expected_shape = (layer.num_local_experts, w13_rows, layer.hidden_size)
        elif weight_name.endswith(".experts.w2_weight"):
            expected_shape = (
                layer.num_local_experts,
                layer.hidden_size,
                layer.intermediate_size_per_partition,
            )

        if expected_shape is None or tuple(param.data.shape) == expected_shape:
            return

        expected_numel = expected_shape[0] * expected_shape[1] * expected_shape[2]
        if param.data.numel() != expected_numel:
            raise RuntimeError(
                f"Cannot restore flashinfer TRT-LLM BF16 MoE weight shape for {weight_name}: "
                f"current shape={tuple(param.data.shape)}, expected shape={expected_shape}."
            )

        param.data = param.data.reshape(expected_shape)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        if self.use_flashinfer_trtllm_moe:
            backend = (
                MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
                if get_moe_runner_backend().is_flashinfer_trtllm_routed()
                else MoeRunnerBackend.FLASHINFER_TRTLLM
            )
        elif self.use_deep_gemm:
            backend = MoeRunnerBackend.DEEP_GEMM
        elif self.use_triton_kernels:
            backend = MoeRunnerBackend.TRITON_KERNELS
        else:
            backend = MoeRunnerBackend.TRITON
        self.runner = MoeRunner(backend, moe_runner_config)

        # Separate runner so CK-shape errors fall back to self.runner on every call.
        self._aiter_runner: Optional[MoeRunner] = None
        if (
            _use_aiter
            and (
                get_moe_runner_backend().is_auto()
                or get_moe_runner_backend().is_aiter()
            )
            and get_moe_a2a_backend().is_none()
        ):
            self._aiter_runner = MoeRunner(MoeRunnerBackend.AITER, moe_runner_config)

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        return self.use_flashinfer_cutlass

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return self.forward(
            layer=layer,
            dispatch_output=dispatch_output,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        moe_runner_config = self.moe_runner_config

        backend = self.runner.runner_backend
        if backend.is_triton_kernels():
            from sglang.srt.layers.moe.moe_runner.triton_kernels import (
                TritonKernelsQuantInfo,
            )

            quant_info = TritonKernelsQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_bias=getattr(layer, "w13_weight_bias", None),
                w2_bias=getattr(layer, "w2_weight_bias", None),
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.runner.runner_backend.is_deep_gemm():
            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight
            from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo

            # Only use_fp8=False when SGLANG_DEEPEP_BF16_DISPATCH is true,
            # otherwise use_fp8=True for FP8 dispatch path
            use_fp8 = not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=w13_weight,
                w2_weight=w2_weight,
                use_fp8=use_fp8,
            )
            return self.runner.run(dispatch_output, quant_info)
        elif self.use_flashinfer_cutlass:
            topk_output = dispatch_output.topk_output
            output = flashinfer_cutlass_fused_moe(
                input=x,
                token_selected_experts=topk_output.topk_ids,
                token_final_scales=topk_output.topk_weights,
                fc1_expert_weights=layer.w13_weight,
                fc2_expert_weights=layer.w2_weight,
                output_dtype=x.dtype,
                quant_scales=None,
                ep_size=layer.moe_ep_size,
                ep_rank=layer.moe_ep_rank,
                tp_size=layer.moe_tp_size,
                tp_rank=layer.moe_tp_rank,
                tune_max_num_tokens=next_power_of_2(x.shape[0]),
                activation_type=(
                    ActivationType.Relu2
                    if moe_runner_config.activation == "relu2"
                    else ActivationType.Swiglu
                ),
            )[0]
            return StandardCombineInput(hidden_states=output)
        elif self.use_flashinfer_trtllm_moe:
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmBf16MoeQuantInfo,
            )

            quant_info = FlashInferTrtllmBf16MoeQuantInfo(
                gemm1_weights=layer.w13_weight,
                gemm2_weights=layer.w2_weight,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            )
            return self.runner.run(dispatch_output, quant_info)
        else:
            # Skip aiter fused_moe when using non-auto MoE backend (e.g., triton, triton_kernels)
            # because aiter CK kernels don't support all GEMM dimensions
            _should_use_aiter_moe = _use_aiter and get_moe_runner_backend().is_auto()
            if _should_use_aiter_moe:
                assert not moe_runner_config.no_combine, "unsupported"
                topk_weights, topk_ids, _ = topk_output
                if moe_runner_config.apply_router_weight_on_input:
                    assert (
                        topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
                    _, topk = topk_weights.shape
                    assert (
                        topk == 1
                    ), "Only support topk=1 when `apply_router_weight_on_input` is True"
                    x = x * topk_weights.to(x.dtype)
                    topk_weights = torch.ones_like(
                        topk_weights, dtype=torch.float32
                    )  # topk_weights must be FP32 (float32)
                output = fused_moe(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    activation=(
                        ActivationType.Silu
                        if moe_runner_config.activation == "silu"
                        else ActivationType.Gelu
                    ),
                    expert_mask=layer.expert_mask_gpu,
                )
                return StandardCombineInput(hidden_states=output)
            elif _is_dcu and _use_marlin_w16a16_moe and not _use_aiter_w16a16_moe:
                    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe_w16a16
                    K = x.size(1)

                    def _is_marlin_w16a16_packed(w1: torch.Tensor,
                                                 w2: torch.Tensor) -> bool:
                        if w1.dim() != 3 or w2.dim() != 3:
                            return False
                        if w1.size(0) != w2.size(0):
                            return False
                        k_div16 = w1.size(1)
                        if k_div16 * 16 != K:
                            return False
                        if w1.size(2) % 16 != 0:
                            return False
                        twoN = w1.size(2) // 16
                        if twoN % 2 != 0:
                            return False
                        N = twoN // 2
                        if w2.size(2) != K * 16:
                            return False
                        if w2.size(1) * 16 != N:
                            return False
                        return True

                    if (getattr(layer.w13_weight, "marlin_w16a16_packed", False)
                        or getattr(layer.w2_weight, "marlin_w16a16_packed", False)
                        or _is_marlin_w16a16_packed(layer.w13_weight, layer.w2_weight)):
                        topk_weights, topk_ids, _ = topk_output
                        origin_w1_shape = getattr(layer.w13_weight, "N", None)
                        output = fused_marlin_moe_w16a16(
                            hidden_states=x,
                            w1=layer.w13_weight,
                            w2=layer.w2_weight,
                            topk_weights=topk_weights,
                            topk_ids=topk_ids,
                            global_num_experts=self.moe_runner_config.num_experts,
                            origin_w1_shape=origin_w1_shape,
                            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
                            inplace=True,
                        )
                        return StandardCombineInput(hidden_states=output)
            elif _is_dcu and _use_aiter_w16a16_moe and not _use_marlin_w16a16_moe:
                w1 = layer.w13_weight[0] if isinstance(layer.w13_weight, tuple) else layer.w13_weight
                w2 = layer.w2_weight[0] if isinstance(layer.w2_weight, tuple) else layer.w2_weight
                topk_weights, topk_ids, _ = topk_output
                if moe_runner_config.apply_router_weight_on_input:
                    assert (
                        topk_weights.dim() == 2
                    ), "`topk_weights` should be (num_tokens, topk)"
                    _, tk = topk_weights.shape
                    assert (
                        tk == 1
                    ), "DCU AITER W16A16 path: apply_router_weight_on_input requires topk=1"
                    x = x * topk_weights.to(x.dtype)
                    topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)

                if x.dim() != 2:
                    raise RuntimeError(
                        f"AITER W16A16 MoE expects 2D hidden_states, got shape={tuple(x.shape)}"
                    )
                if w1.dim() != 3 or w2.dim() != 3:
                    raise RuntimeError(
                        "AITER W16A16 MoE expects 3D expert weights, "
                        f"got w1.shape={tuple(w1.shape)}, w2.shape={tuple(w2.shape)}"
                    )

                M, K = x.shape
                E = self.moe_runner_config.num_experts
                if w1.shape[0] != E or w2.shape[0] != E:
                    raise RuntimeError(
                        "AITER W16A16 MoE expert count mismatch: "
                        f"E={E}, w1.shape={tuple(w1.shape)}, w2.shape={tuple(w2.shape)}"
                    )
                if w1.shape[2] != K or w2.shape[1] != K:
                    raise RuntimeError(
                        "AITER W16A16 MoE K dimension mismatch: "
                        f"K={K}, w1.shape={tuple(w1.shape)}, w2.shape={tuple(w2.shape)}"
                    )

                top_k = topk_ids.shape[1]
                N1 = w1.shape[1]
                N2 = w2.shape[2]
                activation = "silu" if moe_runner_config.activation == "silu" else "gelu"
                routed_scaling_factor = (
                    1.0
                    if moe_runner_config.routed_scaling_factor is None
                    else moe_runner_config.routed_scaling_factor
                )

                status, moe_cfg = get_aiter_moe_config(
                    M=M,
                    E=E,
                    N1=N1,
                    N2=N2,
                    K=K,
                    top_k=top_k,
                    block_size=0,
                    dtype=x.dtype,
                    quant_type=MoeQuantType.W16A16,
                )
                if not status:
                    raise RuntimeError(
                        "[aiter_moe_w16a16] no suitable backend found: "
                        f"M={M}, N1={N1}, N2={N2}, K={K}, E={E}, topk={top_k}"
                    )
                if moe_cfg.solution_type not in {
                    MoeSolutionType.ASM,
                    MoeSolutionType.TRITON,
                    MoeSolutionType.CK,
                }:
                    raise RuntimeError(
                        f"Unsupported solution_type: {moe_cfg.solution_type}"
                    )
                if moe_cfg.quant_type != MoeQuantType.W16A16:
                    raise RuntimeError(f"Unexpected quant_type: {moe_cfg.quant_type}")

                output = aiter_moe(
                    hidden_states=x,
                    w1=w1,
                    w2=w2,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    moe_config=moe_cfg,
                    inplace=True,
                    w1_scale=None,
                    w2_scale=None,
                    activation=activation,
                    block_shape=None,
                    global_num_experts=E,
                    routed_scaling_factor=routed_scaling_factor,
                )
                return StandardCombineInput(hidden_states=output)

            else:
                quant_info = TritonMoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    b13=getattr(layer, "w13_weight_bias", None),
                    b2=getattr(layer, "w2_weight_bias", None),
                )
                return self.runner.run(dispatch_output, quant_info)

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        assert (
            moe_runner_config.activation == "silu"
        ), f"activation = {moe_runner_config.activation} is not supported."

        if use_intel_amx_backend(layer):
            from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

            topk_weights, topk_ids, _ = topk_output
            x, topk_weights = apply_topk_weights_cpu(
                moe_runner_config.apply_router_weight_on_input, topk_weights, x
            )
            output = torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                CPUQuantMethod.UNQUANT,
                None,  # w1_scale
                None,  # w2_scale
                None,  # w1_zp
                None,  # w2_zp
                None,  # block_size
                True,  # is_vnni
            )
            return StandardCombineInput(hidden_states=output)
        else:
            from sglang.srt.layers.moe.fused_moe_native import moe_forward_native

            output = moe_forward_native(
                layer,
                x,
                topk_output,
                moe_runner_config,
            )
            return StandardCombineInput(hidden_states=output)

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            b13=getattr(layer, "w13_weight_bias", None),
            b2=getattr(layer, "w2_weight_bias", None),
        )

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config
        assert moe_runner_config.activation in [
            "silu",
            "gelu",
        ], f"activation = {moe_runner_config.activation} is not supported."

        backend = self.runner.runner_backend
        if use_intel_xpu_backend():
            # sgl-kernel-xpu path
            from sgl_kernel import fused_experts

            topk_weights, topk_ids, _ = topk_output
            if moe_runner_config.apply_router_weight_on_input:
                x = x * topk_weights.to(x.dtype)
                topk_weights = torch.ones_like(topk_weights)
            output = fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                b1=getattr(layer, "w13_weight_bias", None),
                b2=getattr(layer, "w2_weight_bias", None),
                activation=moe_runner_config.activation,
                gemm1_alpha=moe_runner_config.gemm1_alpha,
                gemm1_limit=moe_runner_config.gemm1_clamp_limit,
            )
            return StandardCombineInput(hidden_states=output)
        else:
            assert backend.is_triton()
            assert (
                moe_runner_config.activation == "silu"
            ), f"activation = {moe_runner_config.activation} is not supported \
            for Triton PATH, please set ENV SGLANG_USE_SGL_XPU=1."

            quant_info = self.get_triton_quant_info(layer)
            return self.runner.run(dispatch_output, quant_info)

    def forward_npu(
        self,
        layer: torch.nn.Module,
        dispatch_output: "DispatchOutput",
    ) -> CombineInput:

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker

        if DispatchOutputChecker.format_is_deepep(dispatch_output):
            return self._forward_npu_deepep(layer, dispatch_output)

        # x.shape = [B*S, H]
        x = dispatch_output.hidden_states
        # topk_weights.shape = [B*S, K]; topk_ids.shape = [B*S, K]
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        original_dtype = x.dtype
        num_tokens = x.shape[0]
        topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.to(torch.int32)
        num_experts = layer.num_experts
        top_k = layer.top_k or topk_ids.shape[1]  # in case layer.top_k is not set

        hidden_states, expanded_row_idx, expert_tokens, _ = (
            torch.ops.npu.npu_moe_init_routing_v2(
                x,
                topk_ids,
                active_num=num_tokens * top_k,
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=-1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        w13_bias = [layer.w13_weight_bias] if self.with_bias else None
        w2_bias = [layer.w2_weight_bias] if self.with_bias else None

        # gmm1: gate_up_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            bias=w13_bias,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        # act_fn:
        if self.moe_runner_config.activation == "npu_swiglu_oai":
            from sgl_kernel_npu.activation.swiglu_oai import swiglu_oai

            hidden_states = swiglu_oai(layer, hidden_states)
        elif self.moe_runner_config.activation == "silu":
            hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
        else:
            from sglang.srt.layers.activation import GeluAndMul

            hidden_states = GeluAndMul()(hidden_states)

        # gmm2: down_proj
        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            bias=w2_bias,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=2,
        )

        return StandardCombineInput(hidden_states=final_hidden_states)

    def _forward_npu_deepep(
        self,
        layer: torch.nn.Module,
        dispatch_output: "DispatchOutput",
    ) -> CombineInput:
        from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
            npu_fused_moe_without_routing_weights_bf16,
        )
        from sglang.srt.layers.moe.token_dispatcher import (
            DeepEPLLCombineInput,
            DeepEPNormalCombineInput,
        )
        from sglang.srt.layers.moe.token_dispatcher.base import DispatchOutputChecker

        # NOTE: Ascend's Dispatch & Combine does not support FP16
        output_dtype = torch.bfloat16
        group_list_type = 1

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            hidden_states, _, _, _, num_recv_tokens_per_expert = dispatch_output
            group_list = torch.tensor(
                num_recv_tokens_per_expert,
                dtype=torch.int64,
                device=hidden_states.device,
            )
            combine_cls = DeepEPNormalCombineInput
        else:
            hidden_states, _, _, _, group_list, _ = dispatch_output
            group_list = group_list.to(torch.int64)
            combine_cls = DeepEPLLCombineInput

        hidden_states = npu_fused_moe_without_routing_weights_bf16(
            layer, hidden_states, group_list_type, group_list, output_dtype
        )
        return combine_cls(
            hidden_states=hidden_states,
            topk_ids=dispatch_output.topk_ids,
            topk_weights=dispatch_output.topk_weights,
        )

    def forward_tpu(self, *args, **kwargs) -> CombineInput:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    def forward_musa(self, *args, **kwargs) -> CombineInput:
        return self.forward_cuda(*args, **kwargs)

    forward_native = forward_cpu
