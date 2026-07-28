# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import numpy as np
from lightop import moe_gemm_marlin_w16a16, get_moe_cuda_marlin_config_w16a16
import torch
import torch.nn.functional as F

from sglang.srt.utils import direct_register_custom_op, is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import moe_sum_reduce

    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm

from lightop import fuse_silu_and_mul
from lightop import op as op
from vllm.platforms import current_platform
device_name = current_platform.get_device_name().replace(" ", "_")
num_cus= torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

def get_scalar_type(
    num_bits: int,
    has_zp: bool,
    scales: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
):
    from sgl_kernel.scalar_type import scalar_types

    if (
        not has_zp
        and num_bits == 4
        and scales is not None
        and (scales.dtype == torch.float8_e8m0fnu or global_scale is not None)
    ):
        return scalar_types.float4_e2m1f
    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def swiglu_limit_func(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    swiglu_limit: float = 0.0,
) -> None:
    d = input.shape[1] // 2
    gate = input[:, :d]
    up = input[:, d:]

    if swiglu_limit > 0:
        gate = torch.clamp(gate, max=swiglu_limit)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)

    output.copy_(F.silu(gate) * up)


def swiglu_gpt_oss_sigmoid_alpha_contiguous(
    output: torch.Tensor,
    input: torch.Tensor,  # first half is gate, second half is up
    gemm1_alpha: float,
    gemm1_limit: float,
) -> None:
    d = input.shape[1] // 2
    gate = input[:, :d].clamp(max=gemm1_limit)
    up = input[:, d:].clamp(min=-gemm1_limit, max=gemm1_limit)
    output.copy_(gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1))


@register_custom_op(out_shape="hidden_states")
def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    w1_global_scale: Optional[torch.Tensor] = None,
    w2_global_scale: Optional[torch.Tensor] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
    clamp_limit: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    activation: str = "silu",
    is_gated: bool = True,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (int): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2
    ), "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    is_mxfp4_marlin = (
        num_bits == 4
        and w1_zeros is None
        and w2_zeros is None
        and w1_scale.dtype == torch.float8_e8m0fnu
        and w2_scale.dtype == torch.float8_e8m0fnu
    )
    is_nvfp4_marlin = (
        num_bits == 4
        and w1_zeros is None
        and w2_zeros is None
        and w1_global_scale is not None
        and w2_global_scale is not None
    )
    if is_mxfp4_marlin:
        assert hidden_states.dtype == torch.bfloat16, (
            "MXFP4 Marlin with E8M0 scales is only instantiated for bfloat16 "
            f"activations, got {hidden_states.dtype}"
        )
    elif not is_nvfp4_marlin:
        assert (
            hidden_states.dtype == w1_scale.dtype
        ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w1_scale.dtype ({w1_scale.dtype})"
        assert (
            hidden_states.dtype == w2_scale.dtype
        ), f"moe_wna16_marlin_gemm assumes hidden_states.dtype ({hidden_states.dtype}) == w2_scale.dtype ({w2_scale.dtype})"
    assert num_bits in [4, 8]

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]
    gemm1_n = 2 * N if is_gated else N

    # M block size selection logic
    # TODO: tune this further for specific models
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, global_num_experts
    )

    if workspace is None:
        max_workspace_size = (max(2 * N, K) // 64) * (
            sorted_token_ids.size(0) // block_size_m
        )
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms * 4)
        workspace = torch.zeros(
            max_workspace_size, dtype=torch.int, device=device, requires_grad=False
        )

    scalar_type1 = get_scalar_type(
        num_bits, w1_zeros is not None, w1_scale, w1_global_scale
    )
    scalar_type2 = get_scalar_type(
        num_bits, w2_zeros is not None, w2_scale, w2_global_scale
    )

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    # Marlin skips masked expert rows, so their shared cache must start at zero.
    intermediate_cache13 = torch.zeros(
        (M * topk_ids.shape[1] * max(gemm1_n, K),),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[: M * topk_ids.shape[1] * gemm1_n]
    intermediate_cache1 = intermediate_cache1.view(-1, gemm1_n)
    intermediate_cache3 = intermediate_cache13[: M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)

    use_atomic_add = (
        hidden_states.dtype == torch.half
        or torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
    ) and (not is_mxfp4_marlin)

    intermediate_cache1 = moe_wna16_marlin_gemm(
        hidden_states,
        intermediate_cache1,
        w1,
        w1_bias,
        w1_scale,
        w1_global_scale,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=expert_map is not None,
        b_q_type=scalar_type1,
        size_m=M,
        size_n=gemm1_n,
        size_k=K,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    if activation == "silu" and is_gated and gemm1_alpha is not None:
        if clamp_limit is None:
            raise ValueError("GPT-OSS Marlin activation requires clamp_limit.")
        swiglu_gpt_oss_sigmoid_alpha_contiguous(
            intermediate_cache2,
            intermediate_cache1.view(-1, gemm1_n),
            gemm1_alpha,
            clamp_limit,
        )
    elif activation == "silu" and is_gated and clamp_limit is not None:
        swiglu_limit_func(
            intermediate_cache2,
            intermediate_cache1.view(-1, gemm1_n),
            clamp_limit,
        )
    elif activation == "silu" and is_gated:
        silu_and_mul(intermediate_cache1.view(-1, gemm1_n), intermediate_cache2)
    elif activation == "silu" and not is_gated:
        intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
    elif activation == "relu2" and not is_gated:
        intermediate_cache2 = torch.square(F.relu(intermediate_cache1.view(-1, N)))
    else:
        raise ValueError(f"Unsupported activation: {activation=}, with {is_gated=}")

    if expert_map is not None:
        intermediate_cache3.zero_()

    intermediate_cache3 = moe_wna16_marlin_gemm(
        intermediate_cache2,
        intermediate_cache3,
        w2,
        w2_bias,
        w2_scale,
        w2_global_scale,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=expert_map is not None,
        b_q_type=scalar_type2,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    ).view(-1, topk, K)

    output = hidden_states if inplace else torch.empty_like(hidden_states)

    if is_mxfp4_marlin:
        return torch.sum(intermediate_cache3, dim=1, out=output)
    else:
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

    moe_sum_reduce(
        intermediate_cache3,
        output,
        routed_scaling_factor,
    )
    return output


def fused_marlin_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="fused_marlin_moe",
    op_func=fused_marlin_moe,
    mutates_args=[],
    fake_impl=fused_marlin_moe_fake,
)


def get_weight_perms(interleave: bool = False):
    # ================== 4条mmac 指令进行拼接的结果 ============
    perm = []
    for i in range(64):  # 遍历64个线程，因为是针对一个warp内的

        for col in range(2):  # 遍历列方向2次， 代表2次mmac指令 具体是行还是列还不知道

            cur_col = (i % 16) * 2 + col  # 计算当前线程在哪个列 这里是占据4列

            for row in range(4):  # 每个线程在 每个mmac中需要取8个uint4数据 占据8行
                cur_row = (i // 16) * 4 + row
                # 计算在整个 [32, 64]范围内的实际偏移
                cur_idx = cur_row * 32 + cur_col
                perm.append(cur_idx)

    perm = np.array(perm)
    if interleave:
        # =================  加入混排策略 =================
        # # interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        # # interleave = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        # QQQ 类似的 pack混排策略
        interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        # 按照 interleave 重排后展成 一维数组
        perm = perm.reshape((-1, 8))[:, interleave].ravel()

    perm = torch.from_numpy(perm)

    return perm


# npack重排 //512大小
def marlin_weights_npack2(
    q_w,
    weight_perm,
    k_tile=16,
    n_tile=32):
    # 2048, 768
    size_k, size_n = q_w.shape

    # [7168, 512] ==> [128, 16, 24，32]
    q_w = q_w.reshape((size_k // k_tile, k_tile, size_n // n_tile, n_tile))
    # [128, 16, 24，32] ==> [128, 24, 16，32]
    q_w = q_w.permute((0, 2, 1, 3))
    # [128, 24, 16，32] ==> [128, 12288]
    q_w = q_w.reshape((size_k // k_tile, size_n * k_tile))
    # 按照指定的 perm进行重排
    q_w = q_w.reshape((-1, weight_perm.numel()))[:, weight_perm].reshape(q_w.shape)

    # orig_device = q_w.device

    # q_w = q_w.cpu().numpy()
    # q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    # for i in range(pack_factor):
    #     q_packed |= q_w[:, i::pack_factor] << 4 * i
    # q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_w


def w16a16_marlin_weight(full_w16a16_w  # [size_n, size_k]
                         ):
    # import pdb
    # pdb.set_trace()
    # [size_n, size_k] == > [size_k, size_n] 此时已经是默认NN的 k * n 基于这个进行重排
    full_w16a16_w = full_w16a16_w.T
    # 获取 [16, 32]的权重数据块中，需要重排的顺序
    weight_perm = get_weight_perms()
    # 按照索引进行重排
    marlin_q_w = marlin_weights_npack2(full_w16a16_w, weight_perm, k_tile=16, n_tile=32)
    return marlin_q_w


def get_weight_perms_fp8(interleave: bool = False):
    # ================== 4条mmac 指令进行拼接的结果 ============
    perm = []
    for i in range(64):  # 遍历64个线程

        for col in range(4):  # 遍历列方向4次， 代表4次mmac指令
            cur_col = (i % 16) * 4 + col
            for row in range(8):  # 每个线程在 每个mmac中需要取8个uint4数据
                cur_row = (i // 16) * 8 + row
                # 计算在整个 [32, 64]范围内的实际偏移
                cur_idx = cur_row * 64 + cur_col
                perm.append(cur_idx)

    perm = np.array(perm)
    if interleave:
        # =================  加入混排策略 =================
        # # interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        # # interleave = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        # QQQ 类似的 pack混排策略
        interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        # 按照 interleave 重排后展成 一维数组
        perm = perm.reshape((-1, 8))[:, interleave].ravel()

    perm = torch.from_numpy(perm)

    return perm


def marlin_weights(
    q_w,
    weight_perm,
    k_tile=64,
    n_tile=32,
    pack_factor=8):
    # 7168, 512
    size_k, size_n = q_w.shape

    q_w = q_w.reshape(size_k // k_tile, k_tile, size_n)
    q_w = q_w.transpose(1, 2)
    q_w = q_w.reshape(size_k // k_tile, size_n * k_tile)

    # [7168, 512] ==> [224, 32, 8，64]
    # q_w = q_w.reshape((size_k // k_tile, k_tile, size_n // n_tile, n_tile))
    # # [224, 32, 8，64] ==> [224, 8, 32, 64]
    # q_w = q_w.permute((0, 2, 1, 3))
    # # [224, 8, 32, 64] ==> [224, 16384]
    # q_w = q_w.reshape((size_k // k_tile, size_n * k_tile))
    # w4的压缩 w8不需要
    # 按照指定的 perm进行重排
    # q_w = q_w.reshape((-1, weight_perm.numel()))[:, weight_perm].reshape(q_w.shape)
    # orig_device = q_w.device
    # q_w = q_w.cpu().numpy().astype(np.uint32)
    # q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    # for i in range(pack_factor):
    #     q_packed |= q_w[:, i::pack_factor] << 4 * i
    # q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_w


def w8a8_2_marlin_weight(w4a8_w  # [size_n, size_k// 2 ]
                         ):
    # 将 w4a8 的现有权重 拆开 # [size_n, size_k// 2 ] --> [size_n, size_k]
    full_w4a8_w = w4a8_w
    # [size_n, size_k] == > [size_k, size_n]
    full_w4a8_w = full_w4a8_w.T
    # 获取 [32, 64]的权重数据块中，需要重排的 顺序
    weight_perm = get_weight_perms_fp8()
    # 按照索引进行重排
    marlin_q_w = marlin_weights(full_w4a8_w, weight_perm, k_tile=64, n_tile=32, pack_factor=8)
    return marlin_q_w

def weight8bit_nt_kpack2_marlin1(weight, # [size_n, size_k// 2 ]
                                k_tile=16,
                                k_tile1=4,
                                n_tile=16, 
                                n_tile1=16):
    assert weight.element_size() == 1, "weight 必须是 8 bit 类型"
    if weight.dim() == 2:
        size_n, size_k = weight.shape
        assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((size_n // (n_tile*n_tile1), n_tile1, n_tile, size_k // (k_tile*k_tile1), k_tile1, k_tile))
        # q = q.permute((0, 2, 1, 3)).contiguous()
        q = q.permute((0, 3, 1, 4, 2, 5)).contiguous()
        q = q.reshape((size_n // k_tile, size_k * k_tile))
    elif weight.dim() == 3:
        E, size_n, size_k = weight.shape
        assert size_n % n_tile == 0 and size_k % k_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((E, size_n // (n_tile*n_tile1), n_tile1, n_tile, size_k // (k_tile*k_tile1), k_tile1, k_tile))
        q = q.permute((0, 1, 4, 2, 5, 3, 6)).contiguous()
        q = q.reshape((E, size_n // k_tile, size_k * k_tile))
    return q

def weight8bit_nt_kpack2_marlin(weight, # [size_n, size_k// 2 ]
                                k_tile=16,
                                n_tile=16, ):
    assert weight.element_size() == 1, "weight 必须是 8 bit 类型"
    if weight.dim() == 2:
        size_n, size_k = weight.shape
        assert size_n % k_tile == 0 and size_k % n_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((size_n // n_tile,  n_tile, size_k // k_tile, k_tile))
        q = q.permute((0, 2, 1, 3)).contiguous()
        q = q.reshape((size_n // k_tile, size_k * k_tile))
    elif weight.dim() == 3:
        E, size_n, size_k = weight.shape
        assert size_n % n_tile == 0 and size_k % k_tile == 0, "k_tile / n_tile 必须能整除对应维度"

        q = weight.reshape((E, size_n // n_tile,  n_tile, size_k // k_tile, k_tile))
        q = q.permute((0, 1, 3, 2, 4)).contiguous()
        q = q.reshape((E, size_n // k_tile, size_k * k_tile))
    return q


def fused_marlin_moe_w16a16(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    origin_w1_shape: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 8,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - w1_scale (torch.Tensor): Scale to be used for w1.
    - w2_scale (torch.Tensor): Scale to be used for w2.
    - g_idx1 (Optional[torch.Tensor]): The first set of act_order indices.
    - g_idx2 (Optional[torch.Tensor]): The second set of act_order indices.
    - sort_indices1 (Optional[torch.Tensor]): The first act_order input
        permutation.
    - sort_indices2 (Optional[torch.Tensor]): The second act_order input
        permutation.
    - topk_weights (torch.Tensor): Top-k weights.
    - topk_ids (torch.Tensor): Indices of topk-k elements.
    - w1_zeros (Optional[torch.Tensor]): Optional zero points to be used for w1.
    - w2_zeros (Optional[torch.Tensor]): Optional zero points to be used for w2.
    - num_bits (int): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import hcu_moe_align_block_size

    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]

    M, K = hidden_states.shape
    E = w1.shape[0]  # 128
    N = origin_w1_shape  # 768

    topk = topk_ids.shape[1]

    config_marlin_0, config_marlin_1, status = get_moe_cuda_marlin_config_w16a16(
        E,
        M,
        2 * N,
        K,
        K,
        N,
        topk,
        device_name,
        num_cus,
        hidden_states.dtype)      
    block_size_m = config_marlin_0["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E

    sorted_token_ids, expert_ids, num_tokens_post_padded = hcu_moe_align_block_size(topk_ids, block_size_m, global_num_experts)

    # TODO: tune this further for specific models
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        (M * topk_ids.shape[1] * max(2 * N, K),),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[:M * topk_ids.shape[1] * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N)  # [M * topk, 2 * N]  [32*8, 512]
    intermediate_cache3 = intermediate_cache13[:M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)

    intermediate_cache1 = moe_gemm_marlin_w16a16(
        hidden_states,
        w1,
        intermediate_cache1,
        None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk,
        config_marlin_0
    )
    fuse_silu_and_mul(intermediate_cache1, intermediate_cache2)
    intermediate_cache3 = moe_gemm_marlin_w16a16(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        1,
        config_marlin_1,
    ).view(-1, topk, K)
    output = hidden_states if inplace else torch.empty_like(hidden_states)

    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    op.moe_sum(
        input=intermediate_cache3,
        output=output,
        bias=None,
        expert_mask=None,
        num_local_tokens=None,
        factor=routed_scaling_factor
    )
    return output
