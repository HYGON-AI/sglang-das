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
        SimpleNamespace(m_grouped_fp8_gemm_nt_masked=masked),
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


def test_hcu_deepgemm_runner_capability(monkeypatch):
    monkeypatch.setattr(ep_moe_layer.deep_gemm_wrapper, "ENABLE_HCU_DEEPGEMM", True)
    monkeypatch.setattr(
        ep_moe_layer,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.DEEP_GEMM,
    )

    assert ep_moe_layer._should_use_hcu_deepgemm_runner()


@pytest.mark.parametrize("is_minimax_m3", [False, True])
@pytest.mark.parametrize(
    "backend,quant_name,hcu,jit,aiter,npu,bf16_dispatch,expected_unmarked,expected_marked",
    [
        ("deep_gemm", "fp8", True, False, False, False, False, False, True),
        ("humming", "fp8", False, False, False, False, False, True, True),
        ("auto", "humming", False, False, False, False, False, True, True),
        ("deep_gemm", "fp8", False, False, True, False, False, True, True),
        ("deep_gemm", "fp8", False, False, False, True, False, True, True),
        ("deep_gemm", "fp8", False, True, False, False, False, True, True),
        ("deep_gemm", "generic", False, True, False, False, True, True, True),
        (
            "flashinfer_cutedsl",
            "modelopt_fp4",
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ),
        (
            "flashinfer_cutedsl",
            "modelopt_mixed",
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ),
        ("deep_gemm", "mxfp4", False, True, False, False, False, True, True),
        ("deep_gemm", None, False, True, False, False, False, True, True),
        ("deep_gemm", "generic", False, False, False, False, False, False, False),
        ("triton", "fp8", False, False, False, False, False, False, False),
        ("triton", "fp8", True, False, False, False, False, False, False),
        ("deep_gemm", "slim_w4a8", True, False, False, False, False, False, False),
        ("deep_gemm", "slim_w8a8", True, False, False, False, False, False, False),
    ],
)
def test_minimax_hcu_opt_in_preserves_backend_selection(
    monkeypatch,
    is_minimax_m3,
    backend,
    quant_name,
    hcu,
    jit,
    aiter,
    npu,
    bf16_dispatch,
    expected_unmarked,
    expected_marked,
):
    class FakeFp8Config:
        pass

    def init_fused_moe(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.moe_runner_config = SimpleNamespace(is_minimax_m3=False)
        self.quant_method = SimpleNamespace(block_quant=False)
        self.w13_weight = SimpleNamespace(dtype=torch.bfloat16)
        self.dispatcher = SimpleNamespace()

    runner = SimpleNamespace(
        **{
            "is_" + name: (lambda name=name: backend == name)
            for name in ["humming", "auto", "deep_gemm", "flashinfer_cutedsl"]
        }
    )
    mode = SimpleNamespace(enable_low_latency=lambda: True)
    monkeypatch.setattr(ep_moe_layer.FusedMoE, "__init__", init_fused_moe)
    monkeypatch.setattr(ep_moe_layer, "Fp8Config", FakeFp8Config)
    monkeypatch.setattr(ep_moe_layer, "get_moe_runner_backend", lambda: runner)
    monkeypatch.setattr(ep_moe_layer, "get_deepep_mode", lambda: mode)
    monkeypatch.setattr(
        ep_moe_layer,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_deepep=lambda: True, is_pplx=lambda: False),
    )
    monkeypatch.setattr(ep_moe_layer, "should_use_aiter_runner", lambda: aiter)
    monkeypatch.setattr(ep_moe_layer, "_use_aiter", False)
    monkeypatch.setattr(ep_moe_layer, "_is_npu", npu)
    monkeypatch.setattr(ep_moe_layer, "_is_hip", hcu)
    monkeypatch.setattr(ep_moe_layer, "_use_fp8_w8a8_moe", False)
    monkeypatch.setattr(ep_moe_layer, "_use_marlin_w16a16_moe", False)
    monkeypatch.setattr(ep_moe_layer, "_use_marlin_w4a16_moe", False)
    monkeypatch.setattr(ep_moe_layer.deep_gemm_wrapper, "ENABLE_HCU_DEEPGEMM", hcu)
    monkeypatch.setattr(ep_moe_layer.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", jit)
    monkeypatch.setattr(
        ep_moe_layer.envs.SGLANG_DEEPEP_BF16_DISPATCH, "get", lambda: bf16_dispatch
    )
    quant = (
        FakeFp8Config()
        if quant_name == "fp8"
        else (
            None if quant_name is None else SimpleNamespace(get_name=lambda: quant_name)
        )
    )
    if quant_name == "slim_w4a8":
        quant = object.__new__(ep_moe_layer.SlimQuantW4A8Int8MarlinConfig)
    elif quant_name == "slim_w8a8":
        quant = object.__new__(ep_moe_layer.SlimQuantCompressedTensorsMarlinConfig)
    layer = ep_moe_layer.DeepEPMoE(
        num_experts=128,
        top_k=8,
        hidden_size=4096,
        intermediate_size=2048,
        layer_id=3,
        quant_config=quant,
        is_minimax_m3=is_minimax_m3,
    )
    # Only the HCU-specific opt-in depends on the model marker. All other
    # backend conditions must continue selecting FusedMoE for other models.
    expected = expected_marked if is_minimax_m3 else expected_unmarked
    assert layer.deprecate_flag is expected
    if not expected:
        assert not layer.use_w4a16_marlin
        assert layer.deepep_mode is mode
        if quant_name == "fp8":
            assert layer.use_fp8_w8a8


@pytest.mark.parametrize("is_minimax_m3", [False, True])
def test_breakable_forward_limits_hcu_special_case_to_minimax(
    monkeypatch, is_minimax_m3
):
    hidden = torch.ones((2, 4))
    sentinel = object()
    calls = []

    def legacy_forward(hidden, weights, ids, logits, output):
        calls.append("legacy")
        output.copy_(hidden)

    layer = SimpleNamespace(
        deprecate_flag=True,
        moe_runner_config=SimpleNamespace(is_minimax_m3=is_minimax_m3),
        quant_config=None,
        forward_impl=lambda *args: sentinel,
        a2a_forward_with_output=legacy_forward,
    )
    topk = SimpleNamespace(topk_weights=None, topk_ids=None, router_logits=None)
    monkeypatch.setattr(ep_moe_layer, "is_in_breakable_cuda_graph", lambda: True)
    monkeypatch.setattr(
        ep_moe_layer, "_should_break_only_hcu_deepgemm_core", lambda quant: True
    )
    monkeypatch.setattr(
        ep_moe_layer.TopKOutputChecker, "format_is_standard", lambda value: True
    )
    output = ep_moe_layer.DeepEPMoE.forward(layer, hidden, topk)
    if is_minimax_m3:
        assert output is sentinel
        assert not calls
    else:
        assert torch.equal(output, hidden)
        assert calls == ["legacy"]


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


@pytest.mark.parametrize("is_minimax_m3", [False, True])
def test_hcu_core_break_preserves_other_model_delegation(monkeypatch, is_minimax_m3):
    hcu_result, default_result = object(), object()
    dispatch_output = SimpleNamespace()
    layer = object.__new__(ep_moe_layer.DeepEPMoE)
    torch.nn.Module.__init__(layer)
    layer.deprecate_flag = True
    layer.quant_config = None
    layer.moe_runner_config = SimpleNamespace(is_minimax_m3=is_minimax_m3)
    layer.hcu_ll_moe_core = lambda value: hcu_result
    monkeypatch.setattr(ep_moe_layer, "is_in_breakable_cuda_graph", lambda: True)
    monkeypatch.setattr(
        ep_moe_layer, "_should_break_only_hcu_deepgemm_core", lambda quant: True
    )
    monkeypatch.setattr(
        ep_moe_layer.FusedMoE, "run_moe_core", lambda *args, **kwargs: default_result
    )
    result = layer.run_moe_core(dispatch_output)
    assert result is (hcu_result if is_minimax_m3 else default_result)


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
