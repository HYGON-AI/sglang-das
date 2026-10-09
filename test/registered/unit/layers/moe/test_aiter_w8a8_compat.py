"""CPU regression for the channel-W8A8 callers surviving runner refactoring."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.moe_runner import aiter as runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


@pytest.fixture(autouse=True)
def isolate_config_cache(monkeypatch):
    monkeypatch.setattr(runner, "_AITER_UNIFIED_MOE_CONFIG_CACHE", {})


def layer(fp8=False):
    value = torch.nn.Module()
    dtype = torch.float8_e4m3fn if fp8 else torch.int8
    value.w13_weight = torch.nn.Parameter(
        torch.zeros((2, 16, 8), dtype=dtype), requires_grad=False
    )
    value.w2_weight = torch.nn.Parameter(
        torch.zeros((2, 8, 8), dtype=dtype), requires_grad=False
    )
    value.w13_weight_scale = torch.ones((2, 16, 1))
    value.w2_weight_scale = torch.ones((2, 8, 1))
    value.w13_input_scale = None
    value.w2_input_scale = None
    value._sglang_hcu_aiter_int8_noshuffle = not fp8
    return value


@pytest.mark.parametrize("fp8", [False, True])
def test_prepare_retains_weights_and_quantization(fp8):
    value = layer(fp8)
    weights = value.w13_weight, value.w2_weight
    state_keys = set(value.state_dict())
    prepare = (
        runner.process_weights_after_loading_aiter_w8a8_fp8
        if fp8
        else runner.process_weights_after_loading_aiter_w8a8_int8
    )
    get_info = (
        runner.get_aiter_w8a8_fp8_quant_info
        if fp8
        else runner.get_aiter_w8a8_int8_quant_info
    )
    prepare(value)
    info = get_info(value)
    assert info.w13_weight is weights[0] and info.w2_weight is weights[1]
    assert info.use_fp8_w8a8 is fp8 and info.use_int8_w8a8 is not fp8
    assert info.quant_type == runner.AiterQuantType.PER_TOKEN
    assert info.w8a8_no_shuffle is not fp8
    assert info.a13_scale is None and info.a2_scale is None
    assert set(value.state_dict()) == state_keys
    assert not list(value.buffers())


def test_prepare_rejects_input_router_weight():
    value = layer()
    value.apply_router_weight_on_input = True
    with pytest.raises(RuntimeError, match="apply_router_weight_on_input"):
        runner.process_weights_after_loading_aiter_w8a8_int8(value)


def test_int8_get_info_rejects_wrong_weight_dtype():
    with pytest.raises(ValueError, match="torch.int8"):
        runner.get_aiter_w8a8_int8_quant_info(layer(fp8=True))


def test_quant_info_preserves_global_to_local_expert_map():
    value = layer()
    mapping = torch.tensor([-1, 0, -1, 1], dtype=torch.int32)
    value.dispatcher = SimpleNamespace(
        expert_mask_gpu=torch.ones(4), local_expert_mapping=mapping
    )
    info = runner.get_aiter_w8a8_int8_quant_info(value)
    assert info.expert_mask is mapping


@pytest.mark.parametrize("fp8", [False, True])
def test_w8a8_routes_to_unified_without_activation_override(monkeypatch, fp8):
    value = layer(fp8)
    info = (
        runner.get_aiter_w8a8_fp8_quant_info
        if fp8
        else runner.get_aiter_w8a8_int8_quant_info
    )(value)
    core = runner.AiterRunnerCore(MoeRunnerConfig())
    sentinel = object()
    monkeypatch.setattr(core, "_run_unified_moe", lambda x, q: sentinel)
    inputs = runner.AiterRunnerInput(
        torch.ones((1, 8)),
        torch.zeros((1, 1), dtype=torch.int32),
        torch.ones((1, 1)),
        runner.AiterQuantType.PER_TOKEN,
    )
    assert core.run(inputs, info, {}) is sentinel


@pytest.mark.parametrize("fp8", [False, True])
def test_real_unified_call_keeps_dtype_scales_and_raw_layout(monkeypatch, fp8):
    import aiter.moe as api

    value = layer(fp8)
    info = (
        runner.get_aiter_w8a8_fp8_quant_info
        if fp8
        else runner.get_aiter_w8a8_int8_quant_info
    )(value)
    core = runner.AiterRunnerCore(MoeRunnerConfig(inplace=False, swiglu_limit=10.0))
    calls = {}

    def config(**kwargs):
        calls["config"] = kwargs
        return True, SimpleNamespace(
            solution_type=api.MoeSolutionType.ASM,
            quant_type=kwargs["quant_type"],
            need_shuffle=False,
            config={},
        )

    def execute(*, gemm1_alpha=None, gemm1_limit=None, **kwargs):
        kwargs.update(gemm1_alpha=gemm1_alpha, gemm1_limit=gemm1_limit)
        calls["execute"] = kwargs
        return kwargs["hidden_states"]

    monkeypatch.setattr(api, "get_aiter_moe_config", config)
    monkeypatch.setattr(api, "aiter_moe", execute)
    inputs = runner.AiterRunnerInput(
        torch.ones((1, 8), dtype=torch.bfloat16),
        torch.zeros((1, 1), dtype=torch.int32),
        torch.ones((1, 1)),
        info.quant_type,
    )
    core.run(inputs, info, {})
    expected = api.MoeQuantType.FP8_W8A8 if fp8 else api.MoeQuantType.W8A8
    assert calls["config"]["quant_type"] == expected
    assert calls["execute"]["w1"] is value.w13_weight
    assert calls["execute"]["w2"] is value.w2_weight
    assert calls["execute"]["w1_scale"] is value.w13_weight_scale
    assert calls["execute"]["w2_scale"] is value.w2_weight_scale
    assert calls["execute"]["gemm1_limit"] == 10.0
    assert calls["execute"]["gemm1_alpha"] is None
    if not fp8:
        assert calls["config"]["spec_sol_type"] == api.MoeSolutionType.ASM
        assert calls["config"]["use_shuffle"] == 0


def test_noshuffle_probe_does_not_admit_shuffled_config(monkeypatch):
    import aiter.moe as api

    core = runner.AiterRunnerCore(MoeRunnerConfig())
    info = runner.get_aiter_w8a8_int8_quant_info(layer())
    inputs = runner.AiterRunnerInput(
        torch.ones((1, 8), dtype=torch.bfloat16),
        torch.zeros((1, 1), dtype=torch.int32),
        torch.ones((1, 1)),
        info.quant_type,
    )
    monkeypatch.setattr(
        api,
        "get_aiter_moe_config",
        lambda **kw: (
            True,
            api.AiterMoeConfig(
                solution_type=api.MoeSolutionType.MOE_C,
                quant_type=api.MoeQuantType.W8A8,
                config={},
                need_shuffle=True,
            ),
        ),
    )
    result = core._get_unified_moe_config(inputs, info, api.MoeQuantType.W8A8, 0)
    assert result.solution_type == api.MoeSolutionType.TRITON
    assert not result.need_shuffle


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
