"""MiniMax keeps cached layouts; legacy models retain one weight storage."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.moe_runner import aiter as runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-c-test-cpu")


@pytest.fixture
def layout_case(monkeypatch):
    monkeypatch.setenv("SGLANG_FORCE_AITER_MOE_C", "0")
    monkeypatch.setattr(runner, "_AITER_UNIFIED_MOE_CONFIG_CACHE", {})
    monkeypatch.setattr(runner, "_AITER_UNIFIED_MOE_FALLBACK_WARNINGS", set())
    monkeypatch.setattr(runner, "_aiter_moe_c_has_exact_m", lambda *args: True)

    state = SimpleNamespace(
        solution="moe_c", need_shuffle=True, config_calls=[], shuffle_calls=0
    )
    api = ModuleType("aiter.moe")
    api.MoeSolutionType = SimpleNamespace(ASM="asm", MOE_C="moe_c", TRITON="triton")
    api.AiterMoeConfig = lambda **kwargs: SimpleNamespace(**kwargs)

    def get_config(**kwargs):
        state.config_calls.append(kwargs)
        return True, api.AiterMoeConfig(
            solution_type=state.solution,
            quant_type=kwargs["quant_type"],
            config={},
            need_shuffle=state.need_shuffle,
        )

    def shuffle(w1, w2, config):
        state.shuffle_calls += 1
        return w1.clone(), w2.clone()

    api.get_aiter_moe_config = get_config
    api.aiter_moe_shfl_weight = shuffle
    parent = ModuleType("aiter")
    parent.__path__ = []
    chip = ModuleType("aiter.jit.utils.chip_info")
    chip.get_gfx = lambda: "gfx938"
    monkeypatch.setitem(sys.modules, "aiter", parent)
    monkeypatch.setitem(sys.modules, "aiter.moe", api)
    monkeypatch.setitem(sys.modules, "aiter.jit.utils.chip_info", chip)

    core = runner.AiterRunnerCore(
        MoeRunnerConfig(
            activation="silu",
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
            gate_up_interleaved=False,
        )
    )
    return core, state


def tensors(dtype=torch.bfloat16, tokens=1):
    quant_type = (
        runner.AiterQuantType.NONE
        if dtype == torch.bfloat16
        else runner.AiterQuantType.PER_TOKEN
    )
    info = runner.AiterMoeQuantInfo(
        w13_weight=torch.zeros((2, 8, 4), dtype=dtype),
        w2_weight=torch.zeros((2, 4, 4), dtype=dtype),
        quant_type=quant_type,
    )
    inputs = runner.AiterRunnerInput(
        hidden_states=torch.zeros((tokens, 4), dtype=torch.bfloat16),
        topk_ids=torch.zeros((tokens, 1), dtype=torch.int32),
        topk_weights=torch.ones((tokens, 1), dtype=torch.float32),
        quant_type=quant_type,
    )
    name = {
        torch.bfloat16: "w16a16",
        torch.int8: "int8_w8a8",
        torch.float8_e4m3fn: "fp8_w8a8",
    }[dtype]
    return inputs, info, name


@pytest.mark.parametrize("solution", ["moe_c", "asm"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8, torch.float8_e4m3fn])
@pytest.mark.parametrize("minimax", [False, True])
def test_weight_ownership_is_selected_by_model(layout_case, solution, dtype, minimax):
    core, state = layout_case
    state.solution = solution
    core.config.is_minimax_m3 = minimax
    inputs, info, quant_type = tensors(dtype)
    config = core._get_unified_moe_config(inputs, info, quant_type, 0)
    weights = core._get_unified_weights(info, config)

    assert config.solution_type == solution and config.need_shuffle
    assert "use_shuffle" not in state.config_calls[0]
    assert state.shuffle_calls == 1
    again = core._get_unified_weights(info, config)
    assert again[0].data_ptr() == weights[0].data_ptr()
    assert state.shuffle_calls == 1
    if minimax:
        assert weights[0].data_ptr() != info.w13_weight.data_ptr()
        assert len(core._unified_weight_cache) == 1
        assert core._legacy_weight_layout is None
    else:
        assert weights[0].data_ptr() == info.w13_weight.data_ptr()
        assert weights[1].data_ptr() == info.w2_weight.data_ptr()
        assert not core._unified_weight_cache


def test_automatic_selection_keeps_unshuffled_asm(layout_case):
    core, state = layout_case
    state.solution = "asm"
    state.need_shuffle = False
    inputs, info, quant_type = tensors(torch.float8_e4m3fn)
    config = core._get_unified_moe_config(inputs, info, quant_type, 0)

    assert config.solution_type == "asm"
    assert core._get_unified_weights(info, config)[0] is info.w13_weight
    assert state.shuffle_calls == 0


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_ordinary_w8a8_unified_calls_reuse_model_storage(layout_case, dtype):
    core, state = layout_case
    core.config = MoeRunnerConfig(swiglu_limit=10.0)
    inputs, info, quant_type = tensors(dtype)
    info.use_int8_w8a8 = dtype == torch.int8
    info.use_fp8_w8a8 = dtype == torch.float8_e4m3fn
    api = sys.modules["aiter.moe"]
    api.MoeQuantType = SimpleNamespace(
        W8A8="int8_w8a8", FP8_W8A8="fp8_w8a8", W16A16="w16a16"
    )
    calls = {}

    def execute(*, gemm1_alpha=None, gemm1_limit=None, **kwargs):
        calls.update(kwargs, gemm1_alpha=gemm1_alpha, gemm1_limit=gemm1_limit)
        return kwargs["hidden_states"]

    api.aiter_moe = execute
    output = core.run(inputs, info, {})

    assert output.hidden_states is inputs.hidden_states
    assert calls["w1"].data_ptr() == info.w13_weight.data_ptr()
    assert calls["w2"].data_ptr() == info.w2_weight.data_ptr()
    assert calls["moe_config"].solution_type == "moe_c"
    assert calls["moe_config"].quant_type == quant_type
    assert calls["gemm1_alpha"] is None and calls["gemm1_limit"] == 10.0
    assert state.shuffle_calls == 1
    assert not core._unified_weight_cache


def test_minimax_can_switch_between_cached_and_raw_layouts(layout_case):
    core, state = layout_case
    core.config.is_minimax_m3 = True
    inputs, info, quant_type = tensors()
    for tokens, solution, shuffled in (
        (1, "asm", False),
        (8, "moe_c", True),
        (16, "triton", False),
        (1, "asm", False),
    ):
        inputs, _, _ = tensors(tokens=tokens)
        state.solution, state.need_shuffle = solution, shuffled
        config = core._get_unified_moe_config(inputs, info, quant_type, 0)
        weights = core._get_unified_weights(info, config)
        if config.need_shuffle:
            assert weights[0].data_ptr() != info.w13_weight.data_ptr()
        else:
            assert weights[0] is info.w13_weight and weights[1] is info.w2_weight

    assert len(state.config_calls) == 3
    assert state.shuffle_calls == 1
    assert len(core._unified_weight_cache) == 1


def test_explicit_moe_c_retains_opt_in_and_reuses_shuffle(monkeypatch, layout_case):
    monkeypatch.setenv("SGLANG_FORCE_AITER_MOE_C", "1")
    core, state = layout_case
    inputs, info, quant_type = tensors(torch.float8_e4m3fn)
    config = core._get_unified_moe_config(inputs, info, quant_type, 0)
    first = core._get_unified_weights(info, config)
    second = core._get_unified_weights(info, config)

    assert config.solution_type == "moe_c"
    assert state.config_calls[0]["spec_sol_type"] == "moe_c"
    assert first[0].data_ptr() == second[0].data_ptr() == info.w13_weight.data_ptr()
    assert first[1].data_ptr() == second[1].data_ptr() == info.w2_weight.data_ptr()
    assert state.shuffle_calls == 1
    assert not core._unified_weight_cache


def test_int8_no_shuffle_contract_overrides_explicit_moe_c(monkeypatch, layout_case):
    monkeypatch.setenv("SGLANG_FORCE_AITER_MOE_C", "1")
    core, state = layout_case
    inputs, info, quant_type = tensors(torch.int8)
    info.w8a8_no_shuffle = True
    config = core._get_unified_moe_config(inputs, info, quant_type, 0)

    assert state.config_calls[0]["spec_sol_type"] == "asm"
    assert config.solution_type == "triton"
    assert core._get_unified_weights(info, config)[0] is info.w13_weight
    assert state.shuffle_calls == 0


def test_legacy_shuffle_updates_existing_parameter_and_attributes(layout_case):
    core, state = layout_case
    core.config = MoeRunnerConfig()
    inputs, info, quant_type = tensors()
    info.w13_weight.copy_(torch.arange(64).reshape(2, 8, 4))
    before = info.w13_weight.clone()
    pointer = info.w13_weight.data_ptr()
    loader = object()
    info.w13_weight.weight_loader = loader
    api = sys.modules["aiter.moe"]
    api.aiter_moe_shfl_weight = lambda w1, w2, cfg: (w1.flip(1), w2.clone())
    cfg = SimpleNamespace(
        need_shuffle=True, solution_type="moe_c", quant_type=quant_type, config={}
    )

    weights = core._get_unified_weights(info, cfg)

    assert weights[0].data_ptr() == pointer
    assert info.w13_weight.weight_loader is loader
    torch.testing.assert_close(info.w13_weight, before.flip(1))
    assert not core._unified_weight_cache


def test_legacy_scale_is_interleaved_once_without_a_cached_copy(layout_case):
    core, state = layout_case
    inputs, info, quant_type = tensors(torch.float8_e4m3fn)
    info.w13_scale = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8, 1)
    pointer = info.w13_scale.data_ptr()
    cfg = SimpleNamespace(
        need_shuffle=True, solution_type="moe_c", quant_type=quant_type, config={}
    )
    core._get_unified_weights(info, cfg)
    first = core._get_unified_w1_scale(info, cfg)
    core._get_unified_weights(info, cfg)
    second = core._get_unified_w1_scale(info, cfg)

    assert first is second is info.w13_scale
    assert first.data_ptr() == pointer
    assert first.flatten().tolist() == [
        1,
        5,
        2,
        6,
        3,
        7,
        4,
        8,
        9,
        13,
        10,
        14,
        11,
        15,
        12,
        16,
    ]
    assert not core._unified_scale_cache


@pytest.mark.parametrize(
    ("solution", "shuffled", "params"),
    [("triton", False, {}), ("asm", True, {}), ("moe_c", True, {"PADDED_K": 8})],
)
def test_legacy_rejects_incompatible_config_after_conversion(
    layout_case, solution, shuffled, params
):
    core, state = layout_case
    inputs, info, quant_type = tensors()
    cfg = SimpleNamespace(
        need_shuffle=True, solution_type="moe_c", quant_type=quant_type, config={}
    )
    core._get_unified_weights(info, cfg)
    incompatible = SimpleNamespace(
        need_shuffle=shuffled,
        solution_type=solution,
        quant_type=quant_type,
        config=params,
    )
    with pytest.raises(RuntimeError, match="incompatible weight layout"):
        core._get_unified_weights(info, incompatible)
    assert state.shuffle_calls == 1


def test_legacy_preserves_logical_dimensions_for_packed_views(layout_case, monkeypatch):
    core, state = layout_case
    core.config = MoeRunnerConfig()
    inputs, info, quant_type = tensors()
    api = sys.modules["aiter.moe"]
    api.aiter_moe_shfl_weight = lambda w1, w2, cfg: (
        w1.clone().reshape(2, 4, 8),
        w2.clone().reshape(2, 2, 8),
    )
    cfg = SimpleNamespace(
        need_shuffle=True, solution_type="moe_c", quant_type=quant_type, config={}
    )
    info.w13_weight, info.w2_weight = core._get_unified_weights(info, cfg)
    monkeypatch.setattr(
        runner,
        "_aiter_moe_c_has_exact_m",
        lambda arch, experts, intermediate, quant, tokens: (
            experts == 2 and intermediate == 4
        ),
    )
    selected = core._get_unified_moe_config(inputs, info, quant_type, 0)

    assert selected.solution_type == "moe_c"
    assert state.config_calls[-1]["N1"] == 8
    assert state.config_calls[-1]["N2"] == 4
    assert state.config_calls[-1]["K"] == 4
    assert state.config_calls[-1]["spec_sol_type"] == "moe_c"


def test_legacy_validates_both_weights_before_writing_back(layout_case):
    core, state = layout_case
    inputs, info, quant_type = tensors()
    before = info.w13_weight.clone()
    api = sys.modules["aiter.moe"]
    api.aiter_moe_shfl_weight = lambda w1, w2, cfg: (w1 + 1, torch.cat((w2, w2), 2))
    cfg = SimpleNamespace(
        need_shuffle=True, solution_type="moe_c", quant_type=quant_type, config={}
    )
    with pytest.raises(RuntimeError, match="storage size/layout"):
        core._get_unified_weights(info, cfg)
    torch.testing.assert_close(info.w13_weight, before)
    assert core._legacy_weight_layout is None


def test_legacy_cannot_convert_after_raw_graph_capture(layout_case):
    core, state = layout_case
    inputs, info, quant_type = tensors()
    core._legacy_raw_graph_captured = True
    cfg = SimpleNamespace(
        need_shuffle=True, solution_type="moe_c", quant_type=quant_type, config={}
    )
    with pytest.raises(RuntimeError, match="before CUDA Graph capture"):
        core._get_unified_weights(info, cfg)
    assert state.shuffle_calls == 0


def test_model_marker_separates_shared_config_cache(layout_case):
    legacy, state = layout_case
    minimax = runner.AiterRunnerCore(
        MoeRunnerConfig(
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
            gate_up_interleaved=False,
            is_minimax_m3=True,
        )
    )
    inputs, info, quant_type = tensors(torch.float8_e4m3fn)
    first = minimax._get_unified_moe_config(inputs, info, quant_type, 0)
    second = legacy._get_unified_moe_config(inputs, info, quant_type, 0)
    assert first is not second
    assert minimax._get_unified_moe_config(inputs, info, quant_type, 0) is first
    assert len(state.config_calls) == 2


def test_legacy_retains_original_moe_c_config_selection(layout_case, monkeypatch):
    legacy, state = layout_case
    monkeypatch.setattr(runner, "_aiter_moe_c_has_exact_m", lambda *args: False)
    inputs, info, quant_type = tensors(torch.float8_e4m3fn)
    selected = legacy._get_unified_moe_config(inputs, info, quant_type, 0)
    minimax = runner.AiterRunnerCore(MoeRunnerConfig(is_minimax_m3=True))
    guarded = minimax._get_unified_moe_config(inputs, info, quant_type, 0)
    assert selected.solution_type == "moe_c"
    assert guarded.solution_type == "triton"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU Graph support")
def test_legacy_graph_reads_live_storage_after_local_eplb_update(
    layout_case, monkeypatch
):
    from sglang.srt.eplb.expert_location_updater import (
        create_temp_buffers,
        update_expert_weights_single_layer,
    )

    free_memory, _ = torch.cuda.mem_get_info()
    if free_memory < 1024**3:
        pytest.skip("insufficient free memory for an isolated Graph microtest")
    core, state = layout_case
    core.config = MoeRunnerConfig()
    inputs, info, quant_type = tensors()
    for name in ("w13_weight", "w2_weight"):
        setattr(info, name, getattr(info, name).float().cuda())
    info.w13_scale = torch.ones((2, 8, 1), device="cuda")
    info.w2_scale = torch.ones((2, 4, 1), device="cuda")
    info.w13_weight[0].fill_(1)
    info.w13_weight[1].fill_(2)
    info.w2_weight[0].fill_(3)
    info.w2_weight[1].fill_(4)
    info.w13_scale[0].fill_(5)
    info.w13_scale[1].fill_(6)
    cfg = SimpleNamespace(
        need_shuffle=True, solution_type="moe_c", quant_type=quant_type, config={}
    )
    weights = core._get_unified_weights(info, cfg)
    with monkeypatch.context() as patch:
        patch.setattr(
            torch.cuda,
            "is_current_stream_capturing",
            lambda: pytest.fail("converted weights should not query capture state"),
        )
        again = core._get_unified_weights(info, cfg)
    assert again[0].data_ptr() == weights[0].data_ptr()
    pointers = info.w13_weight.data_ptr(), info.w2_weight.data_ptr()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        output = weights[0][0, 0] * info.w13_scale[0, 0] + weights[1][0, 0]

    model_weights = [info.w13_weight, info.w2_weight, info.w13_scale, info.w2_scale]
    buffers = create_temp_buffers(model_weights)
    update_expert_weights_single_layer(
        routed_experts_weights=model_weights,
        temp_buffers=buffers,
        old_physical_to_logical_map=[0, 1],
        new_physical_to_logical_map=[1, 0],
        num_local_physical_experts=2,
        num_gpu_per_node=1,
        rank=0,
        world_size=1,
    )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 16))
    assert (info.w13_weight.data_ptr(), info.w2_weight.data_ptr()) == pointers
    assert not core._unified_weight_cache


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
