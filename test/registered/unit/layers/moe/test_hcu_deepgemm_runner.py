from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.deep_gemm_wrapper.entrypoint as deep_gemm_entrypoint
import sglang.srt.layers.moe.ep_moe.layer as ep_moe_layer
import sglang.srt.layers.moe.moe_runner.aiter as aiter_runner
import sglang.srt.layers.moe.moe_runner.deep_gemm as deep_gemm_runner
import sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph as breakable_graph
from sglang.srt.layers.moe.moe_runner.aiter import AiterMoeQuantInfo, AiterQuantType
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def _quant_info(**overrides):
    kwargs = {
        "w13_weight": torch.empty((2, 1, 1, 1, 1, 1), dtype=torch.int8),
        "w2_weight": torch.empty((2, 1, 1, 1, 1, 1), dtype=torch.int8),
        "use_fp8": True,
        "w13_scale": torch.ones((2, 8, 1), dtype=torch.float32),
        "w2_scale": torch.ones((2, 4, 1), dtype=torch.float32),
        "logical_w13_shape": (2, 8, 4),
        "logical_w2_shape": (2, 4, 4),
        "hcu_packed": True,
    }
    kwargs.update(overrides)
    return DeepGemmMoeQuantInfo(**kwargs)


def test_hcu_deepgemm_masked_wrapper_uses_vendor_abi(monkeypatch):
    captured = {}

    def masked(lhs, rhs, out, masked_m, expected_m, **kwargs):
        captured.update(
            lhs=lhs,
            rhs=rhs,
            out=out,
            masked_m=masked_m,
            expected_m=expected_m,
            kwargs=kwargs,
        )
        return out

    monkeypatch.setattr(deep_gemm_entrypoint, "ENABLE_HCU_DEEPGEMM", True)
    monkeypatch.setattr(
        deep_gemm_entrypoint,
        "deepgemm",
        SimpleNamespace(m_grouped_fp8_gemm_nt_masked_ll=masked),
    )
    lhs = (torch.empty((2, 3, 4)), torch.empty((2, 3)))
    rhs = (torch.empty((2, 1, 1, 1, 1, 1)), torch.empty((2, 8)))
    out = torch.empty((2, 3, 8))
    masked_m = torch.tensor([2, 3], dtype=torch.int32)
    signal = torch.empty((1,), dtype=torch.int32)

    result = deep_gemm_entrypoint.grouped_gemm_nt_f8f8bf16_masked(
        lhs,
        rhs,
        out,
        masked_m,
        expected_m=3,
        overlap_args=SimpleNamespace(signal=signal),
    )

    assert result is out
    assert captured["lhs"] is lhs
    assert captured["rhs"] is rhs
    assert captured["masked_m"] is masked_m
    assert captured["expected_m"] == 3
    assert captured["kwargs"] == {"enable_overlap": True, "signal": signal}


def test_hcu_deepgemm_contiguous_wrapper_uses_vendor_abi(monkeypatch):
    captured = {}

    def contiguous(lhs, rhs, out, m_indices):
        captured.update(lhs=lhs, rhs=rhs, out=out, m_indices=m_indices)
        return out

    monkeypatch.setattr(deep_gemm_entrypoint, "ENABLE_HCU_DEEPGEMM", True)
    monkeypatch.setattr(
        deep_gemm_entrypoint,
        "deepgemm",
        SimpleNamespace(m_grouped_fp8_gemm_nt_contiguous=contiguous),
    )
    lhs = (torch.empty((5, 4)), torch.empty((5, 1)))
    rhs = (torch.empty((2, 1, 1, 1, 1, 1)), torch.empty((2, 8)))
    out = torch.empty((5, 8))
    m_indices = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)

    result = deep_gemm_entrypoint.grouped_gemm_nt_f8f8bf16_contig(
        lhs, rhs, out, m_indices
    )

    assert result is out
    assert captured == {
        "lhs": lhs,
        "rhs": rhs,
        "out": out,
        "m_indices": m_indices,
    }


def test_hcu_deepgemm_problem_shape_uses_logical_packed_shapes():
    assert DeepGemmRunnerCore._hcu_problem_shape(_quant_info()) == (2, 8, 4)

    with pytest.raises(RuntimeError, match="split gated"):
        DeepGemmRunnerCore._hcu_problem_shape(_quant_info(logical_w13_shape=(2, 7, 4)))


def test_hcu_deepgemm_uses_modern_deepep_runner_stack(monkeypatch):
    monkeypatch.setattr(ep_moe_layer.deep_gemm_wrapper, "ENABLE_HCU_DEEPGEMM", True)
    monkeypatch.setattr(
        ep_moe_layer,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.DEEP_GEMM,
    )

    assert ep_moe_layer._should_use_hcu_deepgemm_runner()


def test_hcu_deepgemm_breakable_capture_stub_preserves_ll_layout():
    ep_moe_layer._HCU_LL_GRAPH_BRIDGE_BUFFERS.clear()
    layer = object.__new__(ep_moe_layer.DeepEPMoE)
    layer.params_dtype = torch.bfloat16
    dispatch_output = DeepEPLLDispatchOutput(
        hidden_states=torch.empty((2, 3, 4), dtype=torch.float8_e4m3fn),
        hidden_states_scale=torch.ones((2, 3), dtype=torch.float32),
        topk_ids=torch.zeros((2, 3), dtype=torch.int64),
        topk_weights=torch.ones((2, 3), dtype=torch.float32),
        masked_m=torch.tensor([2, 3], dtype=torch.int32),
        expected_m=3,
    )

    output = layer._hcu_ll_moe_core_capture_stub(dispatch_output)

    assert isinstance(output, DeepEPLLCombineInput)
    assert output.hidden_states.shape == dispatch_output.hidden_states.shape
    assert output.hidden_states.dtype == torch.bfloat16
    assert output.topk_ids is dispatch_output.topk_ids
    assert output.topk_weights is dispatch_output.topk_weights

    second = layer._hcu_ll_moe_core_capture_stub(dispatch_output)
    assert second.hidden_states is output.hidden_states


def test_hcu_deepgemm_breakable_routes_only_moe_core(monkeypatch):
    sentinel = object()
    dispatch_output = SimpleNamespace()
    layer = SimpleNamespace(
        deprecate_flag=True,
        hcu_ll_moe_core=lambda value: sentinel if value is dispatch_output else None,
    )
    monkeypatch.setattr(ep_moe_layer, "is_in_breakable_cuda_graph", lambda: True)
    monkeypatch.setattr(
        ep_moe_layer, "_should_break_only_hcu_deepgemm_core", lambda: True
    )

    assert ep_moe_layer.DeepEPMoE.run_moe_core(layer, dispatch_output) is sentinel


def test_breakable_graph_helpers_preserve_namedtuple_protocol():
    record = DeepEPLLCombineInput("hidden", "ids", "weights")
    weak_record = breakable_graph._weak_ref_if_tensor(record)
    assert isinstance(weak_record, DeepEPLLCombineInput)
    assert weak_record.format.value == "deepep_ll"

    dst = DeepEPLLCombineInput(torch.zeros(2), torch.zeros(2), torch.zeros(2))
    src = DeepEPLLCombineInput(torch.ones(2), torch.ones(2), torch.ones(2))
    copied = breakable_graph._copy_output(dst, src)
    assert isinstance(copied, DeepEPLLCombineInput)
    assert copied.format.value == "deepep_ll"
    assert torch.equal(dst.hidden_states, src.hidden_states)


def test_hcu_deepgemm_masked_forwards_minimax_activation_parameters(monkeypatch):
    gemm_calls = []
    activation_call = {}

    def grouped_gemm(lhs, rhs, out, masked_m, expected_m, **kwargs):
        gemm_calls.append(
            {
                "lhs": lhs,
                "rhs": rhs,
                "out": out,
                "masked_m": masked_m,
                "expected_m": expected_m,
            }
        )
        return out

    def activation(gateup_output, masked_m, **kwargs):
        activation_call.update(kwargs)
        experts, capacity, gateup = gateup_output.shape
        intermediate = gateup // 2
        return (
            torch.empty((experts, capacity, intermediate), dtype=torch.float8_e4m3fn),
            torch.ones((experts, capacity), dtype=torch.float32),
        )

    monkeypatch.setattr(
        deep_gemm_runner.deep_gemm_wrapper,
        "grouped_gemm_nt_f8f8bf16_masked",
        grouped_gemm,
    )
    monkeypatch.setattr(
        deep_gemm_runner, "_varlen_deep_gemm_silu_mul_quant", activation
    )
    monkeypatch.setattr(deep_gemm_runner, "dispose_tensor", lambda _: None)

    config = MoeRunnerConfig(
        activation="silu",
        is_gated=True,
        top_k=4,
        gemm1_alpha=1.702,
        gemm1_clamp_limit=7.0,
    )
    core = DeepGemmRunnerCore(config)
    hidden_states = torch.empty((2, 3, 4), dtype=torch.float8_e4m3fn)
    hidden_states_scale = torch.ones((2, 3), dtype=torch.float32)
    masked_m = torch.tensor([2, 3], dtype=torch.int32)
    runner_input = DeepGemmRunnerInput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=3,
    )

    output = core._run_hcu_masked_gemm(runner_input, _quant_info(), {})

    assert output.shape == (2, 3, 4)
    assert len(gemm_calls) == 2
    assert all(call["expected_m"] == 3 for call in gemm_calls)
    assert activation_call == {
        "group_size": 4,
        "topk": 4,
        "gemm1_alpha": 1.702,
        "gemm1_clamp_limit": 7.0,
    }

    bridge = torch.empty((2, 3, 4), dtype=torch.bfloat16)
    gemm_calls.clear()
    with deep_gemm_runner.use_hcu_masked_output_buffer(bridge):
        bridged_output = core._run_hcu_masked_gemm(runner_input, _quant_info(), {})
    assert bridged_output is bridge
    assert gemm_calls[-1]["out"] is bridge


def test_deepep_ll_aiter_compacts_minimax_grouped_input(monkeypatch):
    captured = {}

    def compact(hidden_states, hidden_states_scale, masked_m, max_tokens, sink):
        captured.update(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            masked_m=masked_m,
            max_tokens=max_tokens,
            sink=sink,
        )
        return (
            torch.zeros((max_tokens, 4), dtype=torch.float8_e4m3fn),
            torch.ones((max_tokens, 1), dtype=torch.float32),
            torch.full((max_tokens, 1), sink, dtype=torch.int32),
            torch.zeros((max_tokens, 1), dtype=torch.float32),
            torch.zeros(2, dtype=torch.int32),
        )

    import sglang.kernels.ops.moe.ep_moe_kernels as ep_kernels

    monkeypatch.setattr(ep_kernels, "compact_deepep_ll_for_aiter", compact)
    grouped = torch.empty((2, 3, 4), dtype=torch.float8_e4m3fn)
    scales = torch.ones((2, 3), dtype=torch.float32)
    masked_m = torch.tensor([2, 1], dtype=torch.int32)
    dispatch_output = SimpleNamespace(
        format=SimpleNamespace(is_deepep_ll=lambda: True),
        hidden_states=grouped,
        hidden_states_scale=scales,
        masked_m=masked_m,
        topk_ids=torch.zeros((2, 2), dtype=torch.int64),
        topk_weights=torch.ones((2, 2), dtype=torch.float32),
    )
    quant_info = AiterMoeQuantInfo(
        w13_weight=torch.empty((2, 8, 4)),
        w2_weight=torch.empty((2, 4, 4)),
        quant_type=AiterQuantType.PER_TOKEN,
    )
    config = MoeRunnerConfig(
        activation="silu",
        num_experts=4,
        num_local_experts=2,
        gemm1_alpha=1.702,
        gemm1_clamp_limit=7.0,
    )
    running_state = {}

    result = aiter_runner._pre_permute_deepep_to_aiter(
        dispatch_output, quant_info, config, running_state
    )

    assert result.hidden_states.shape == (8, 4)
    assert result.hidden_states.dtype == torch.float8_e4m3fn
    assert result.a1_scale.shape == (8, 1)
    assert result.input_is_dequantized is False
    assert captured["max_tokens"] == 8
    assert captured["sink"] == 0
    assert result.uses_local_expert_ids is True
    assert running_state["aiter_deepep_ll_compact"] is True
