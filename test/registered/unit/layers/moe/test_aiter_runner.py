import sglang.srt.layers.moe.fused_moe_triton.layer as fused_moe_layer
import sglang.srt.layers.moe.token_dispatcher.aiter_utils as dispatcher_aiter_utils
import sglang.srt.layers.moe.token_dispatcher.moriep as mori_dispatcher
import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe as channel_fp8_scheme
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
    get_aiter_moe_activation_kwargs,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
    CompressedTensorsW8A8Fp8MoE,
)
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.aiter as aiter_runner
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.aiter import _remap_aiter_ep_topk
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="stage-b-test-cpu-intel")


def _runner_input():
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    return AiterRunnerInput(
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        topk_ids=topk_ids,
        topk_weights=torch.ones(topk_ids.shape, dtype=torch.float32),
        quant_type=AiterQuantType.PER_1X32,
    )


def _quant_info(**overrides):
    kwargs = {
        "w13_weight": torch.empty((2, 8, 2)),
        "w2_weight": torch.empty((2, 4, 2)),
        "quant_type": AiterQuantType.PER_1X32,
    }
    kwargs.update(overrides)
    return AiterMoeQuantInfo(**kwargs)


@pytest.fixture(autouse=True)
def _upstream_aiter_platform(monkeypatch):
    monkeypatch.setattr(aiter_runner, "_is_hcu", False)


def _install_fake_aiter(monkeypatch, fused_moe):
    # Exercise the upstream AITER contract independently of the host platform.
    monkeypatch.setattr(aiter_runner, "_is_hcu", False)
    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    fake_aiter.ActivationType = SimpleNamespace(Silu="Silu")
    fake_aiter.QuantType = SimpleNamespace(per_1x32="per_1x32")

    fake_fused_moe = ModuleType("aiter.fused_moe")
    fake_fused_moe.fused_moe = fused_moe

    fake_ops = ModuleType("aiter.ops")
    fake_ops.__path__ = []
    fake_flydsl = ModuleType("aiter.ops.flydsl")
    fake_flydsl.__path__ = []
    fake_moe_common = ModuleType("aiter.ops.flydsl.moe_common")
    fake_moe_common.GateMode = SimpleNamespace(
        INTERLEAVE=SimpleNamespace(value="INTERLEAVE")
    )

    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fake_fused_moe)
    monkeypatch.setitem(sys.modules, "aiter.ops", fake_ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", fake_flydsl)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.moe_common", fake_moe_common)


def test_aiter_runner_forwards_no_combine_and_extra_fused_moe_kwargs(monkeypatch):
    captured = {}

    def fused_moe(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"]

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )

    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu", no_combine=True))

    runner.run(
        _runner_input(),
        _quant_info(fused_moe_kwargs={"custom_fused_moe_kwarg": "enabled"}),
        running_state={},
    )

    assert captured["activation"] == "Silu"
    assert captured["quant_type"] == "per_1x32"
    assert captured["no_combine"] is True
    assert captured["custom_fused_moe_kwarg"] == "enabled"


def test_aiter_runner_rejects_no_combine_when_fused_moe_does_not_support_it(
    monkeypatch,
):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: False
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))

    with pytest.raises(NotImplementedError, match="no_combine=True"):
        runner.run(_runner_input(), _quant_info(), running_state={})


def test_aiter_runner_preserves_no_combine_rank_for_empty_input(monkeypatch):
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_no_combine", lambda: True
    )
    runner = AiterRunnerCore(MoeRunnerConfig(no_combine=True))
    runner_input = _runner_input()
    runner_input.hidden_states = torch.zeros((0, 4), dtype=torch.bfloat16)
    runner_input.topk_ids = torch.zeros((0, 2), dtype=torch.int32)
    runner_input.topk_weights = torch.zeros((0, 2), dtype=torch.float32)

    output = runner.run(runner_input, _quant_info(), running_state={})

    assert output.hidden_states.shape == (0, 2, 4)


def test_remap_aiter_ep_topk_masks_nonlocal_and_invalid_global_ids():
    expert_map = torch.full((8,), -1, dtype=torch.int32)
    expert_map[2:4] = torch.arange(2, dtype=torch.int32)
    topk_ids = torch.tensor([[2, 3, 1, -1, 8]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.1]])

    local_ids, local_weights = _remap_aiter_ep_topk(topk_ids, topk_weights, expert_map)

    torch.testing.assert_close(
        local_ids, torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.int32)
    )
    torch.testing.assert_close(local_weights, torch.tensor([[0.5, 0.4, 0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("supports_output", [False, True])
def test_aiter_runner_uses_epv2_output_when_kernel_supports_it(
    monkeypatch, supports_output
):
    def fused_moe(hidden_states, output=None, **kwargs):
        result = hidden_states + 2
        if output is not None:
            output.copy_(result)
            return output
        return result

    _install_fake_aiter(monkeypatch, fused_moe)
    monkeypatch.setattr(
        aiter_runner, "_aiter_fused_moe_supports_output", lambda: supports_output
    )
    runner_input = _runner_input()
    runner_input.hidden_states.fill_(1)
    runner_input.output = torch.zeros_like(runner_input.hidden_states)
    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu"))

    result = runner.run(runner_input, _quant_info(), running_state={}).hidden_states

    torch.testing.assert_close(result, torch.full_like(result, 3))
    assert (result.data_ptr() == runner_input.output.data_ptr()) == supports_output
    torch.testing.assert_close(
        runner_input.output,
        torch.full_like(result, 3 if supports_output else 0),
    )


@pytest.mark.parametrize(
    "low_latency,recv_cap,expected_rows",
    [(False, 0, 16), (False, 32, 16), (True, 5, 5)],
)
def test_mori_pre_permute_consumes_dispatcher_cap(
    monkeypatch, low_latency, recv_cap, expected_rows
):
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
    )

    fake_utils = ModuleType("sglang.kernels.ops.moe.rocm_moe_utils")
    fake_utils.upscale = None
    fake_utils.upscale_mxfp4 = None
    monkeypatch.setitem(sys.modules, fake_utils.__name__, fake_utils)
    output_type = MoriEPLLDispatchOutput if low_latency else MoriEPNormalDispatchOutput
    hidden = torch.zeros((16, 4), dtype=torch.bfloat16)
    scales = torch.ones((16, 1))
    ids = torch.zeros((16, 2), dtype=torch.int32)
    weights = torch.ones((16, 2))
    kwargs = {} if low_latency else {"expert_output": torch.empty_like(hidden)}
    dispatched = output_type(
        hidden_states=hidden,
        hidden_states_scale=scales,
        topk_ids=ids,
        topk_weights=weights,
        num_recv_tokens_per_expert=torch.tensor([5]),
        origin_topk_ids=ids[:2],
        origin_topk_weights=weights[:2],
        out_dtype=torch.bfloat16,
        recv_cap=recv_cap,
        **kwargs,
    )
    quant_info = _quant_info(quant_type=AiterQuantType.NONE)
    state = {}
    # The dispatcher owns the override. The runner must not read it again.
    with envs.SGLANG_MORI_MOE_MAX_INPUT_TOKENS.override(3):
        result = aiter_runner._pre_permute_deepep_to_aiter(
            dispatched, quant_info, MoeRunnerConfig(), state
        )
    assert result.hidden_states.shape[0] == expected_rows
    assert result.topk_ids.shape[0] == expected_rows
    assert result.topk_weights.shape[0] == expected_rows
    assert result.a1_scale.shape[0] == expected_rows
    assert result.hidden_states.data_ptr() == hidden.data_ptr()
    assert state["aiter_combine_topk_ids"] is dispatched.origin_topk_ids
    if not low_latency:
        assert result.output.shape[0] == expected_rows
        assert result.output.data_ptr() == dispatched.expert_output.data_ptr()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_aiter_compact_asm_requires_exact_shape(tmp_path):
    csv_path = tmp_path / "tuned_fmoe_asm_w8a8_channel.csv"
    csv_path.write_text(
        "arch,quant_type,indtype,token,inter_dim,model_dim,expert,topk,"
        "q_size_n,q_size_k,sol_type,sol_id,time_us\n"
        "gfx938,f8_w8a8_channel,torch.float8_e4m3fn,64,3072,6144,16,1,"
        "0,0,asm,10011+20000,700.0\n"
    )

    assert aiter_runner._aiter_asm_has_exact_shape(
        "gfx938", 16, 3072, 6144, 1, 64, "torch.float8_e4m3fn", str(csv_path)
    )
    assert not aiter_runner._aiter_asm_has_exact_shape(
        "gfx938", 16, 3072, 6144, 1, 96, "torch.float8_e4m3fn", str(csv_path)
    )


def test_aiter_moe_c_requires_exact_m_in_both_stage_tables(tmp_path):
    directory = tmp_path / "gfx938" / "fp8_w8a8"
    directory.mkdir(parents=True)
    base = "E=16,N=3072,dtype=fp8_w8a8"
    (directory / f"{base}.json").write_text('{"512": {}}')
    (directory / f"{base},is_bottom=True.json").write_text('{"512": {}}')

    assert aiter_runner._aiter_moe_c_has_exact_m(
        "gfx938", 16, 3072, "fp8_w8a8", 512, str(tmp_path)
    )
    assert not aiter_runner._aiter_moe_c_has_exact_m(
        "gfx938", 16, 3072, "fp8_w8a8", 1024, str(tmp_path)
    )


def test_unified_aiter_moe_c_uses_oai_activation_and_interleaved_scale(
    monkeypatch,
):
    captured = {}

    class MoeSolutionType:
        ASM = "asm"
        MOE_C = "moe_c"

    def aiter_moe(*, gemm1_alpha=None, gemm1_limit=None, **kwargs):
        captured.update(kwargs)
        captured["gemm1_alpha"] = gemm1_alpha
        captured["gemm1_limit"] = gemm1_limit
        return kwargs["hidden_states"]

    fake_moe = ModuleType("aiter.moe")
    fake_moe.MoeSolutionType = MoeSolutionType
    fake_moe.aiter_moe = aiter_moe
    monkeypatch.setitem(sys.modules, "aiter.moe", fake_moe)

    runner = AiterRunnerCore(
        MoeRunnerConfig(
            activation="silu",
            is_gated=True,
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
            gate_up_interleaved=False,
        )
    )
    config = SimpleNamespace(
        need_shuffle=False,
        solution_type="moe_c",
        quant_type="fp8_w8a8",
    )
    runner._get_unified_moe_config = lambda *args: config
    runner._unified_quant_params = lambda quant_type: ("fp8_w8a8", 0, None)
    scale = torch.arange(16, dtype=torch.float32).reshape(2, 8, 1)

    output = runner._run_unified_moe(
        _runner_input(),
        _quant_info(w13_scale=scale),
    )

    assert output.hidden_states.shape == (1, 4)
    assert captured["activation"] == "swigluoai"
    assert captured["gemm1_alpha"] == pytest.approx(1.702)
    assert captured["gemm1_limit"] == pytest.approx(7.0)
    assert captured["w1_scale"][0, :, 0].tolist() == [
        0.0,
        4.0,
        1.0,
        5.0,
        2.0,
        6.0,
        3.0,
        7.0,
    ]


def test_unified_aiter_moe_c_interleaves_minimax_gate_up_before_shuffle(
    monkeypatch,
):
    fake_moe = ModuleType("aiter.moe")
    fake_moe.MoeQuantType = SimpleNamespace(W8A8="int8_w8a8", FP8_W8A8="fp8_w8a8")
    fake_moe.MoeSolutionType = SimpleNamespace(MOE_C="moe_c")
    captured = {}

    def shuffle(w1, w2, _config):
        captured["w1"] = w1
        captured["w2"] = w2
        return w1, w2

    fake_moe.aiter_moe_shfl_weight = shuffle
    monkeypatch.setitem(sys.modules, "aiter.moe", fake_moe)

    runner = AiterRunnerCore(
        MoeRunnerConfig(
            activation="silu",
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
            gate_up_interleaved=False,
        )
    )
    w1 = torch.tensor([[[1.0], [2.0], [10.0], [20.0]]])
    w2 = torch.tensor([[[3.0]]])
    w1_scale = torch.tensor([[[0.1], [0.2], [1.0], [2.0]]])
    quant_info = _quant_info(
        w13_weight=w1,
        w2_weight=w2,
        w13_scale=w1_scale,
    )
    config = SimpleNamespace(
        need_shuffle=True,
        solution_type="moe_c",
        quant_type="fp8_w8a8",
    )

    result = runner._get_unified_weights(quant_info, config)
    interleaved_scale = runner._get_unified_w1_scale(quant_info, config)

    assert result[0].flatten().tolist() == [1.0, 10.0, 2.0, 20.0]
    assert result[1] is w2
    assert captured["w1"] is result[0]
    assert captured["w2"] is w2
    assert interleaved_scale.flatten().tolist() == pytest.approx([0.1, 1.0, 0.2, 2.0])


@pytest.mark.parametrize(
    ("runner_quant_type", "expected_aiter_quant", "expected_block_shape"),
    [
        (AiterQuantType.NONE, "w16a16", None),
        (AiterQuantType.PER_TOKEN, "fp8_w8a8", None),
        (AiterQuantType.PER_128X128, "fp8_w8a8", [128, 128]),
    ],
)
def test_minimax_uses_unified_aiter_moe_for_bf16_and_fp8(
    monkeypatch,
    runner_quant_type,
    expected_aiter_quant,
    expected_block_shape,
):
    captured = {}
    _install_fake_unified_aiter(monkeypatch, captured)

    runner = AiterRunnerCore(
        MoeRunnerConfig(
            activation="silu",
            is_gated=True,
            inplace=True,
            gemm1_alpha=1.702,
            gemm1_beta=1.0,
            gemm1_clamp_limit=7.0,
            gate_up_interleaved=False,
        )
    )
    runner_input = _runner_input()
    runner_input.quant_type = runner_quant_type
    expert_mask = torch.tensor([True, False, True])
    quant_info = _quant_info(
        quant_type=runner_quant_type,
        expert_mask=expert_mask,
        w13_scale=torch.ones((2, 8, 1)),
        w2_scale=torch.ones((2, 4, 1)),
    )

    output = runner.run(runner_input, quant_info, running_state={})

    config_kwargs = captured["config_kwargs"]
    call_kwargs = captured["call_kwargs"]
    assert output.hidden_states is runner_input.hidden_states
    assert config_kwargs["quant_type"] == expected_aiter_quant
    assert call_kwargs["moe_config"].solution_type == "triton"
    assert call_kwargs["gemm1_alpha"] == pytest.approx(1.702)
    assert call_kwargs["gemm1_limit"] == pytest.approx(7.0)
    assert call_kwargs["activation"] == "silu"
    assert call_kwargs["block_shape"] == expected_block_shape
    assert call_kwargs["global_num_experts"] == 3
    assert call_kwargs["expert_map"].tolist() == [0, -1, 1]


def _install_fake_unified_aiter(monkeypatch, captured):
    monkeypatch.setattr(aiter_runner, "_is_hcu", True)
    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []

    fake_moe = ModuleType("aiter.moe")

    class AiterMoeConfig:
        def __init__(
            self,
            quant_type=None,
            solution_type=None,
            config=None,
            need_shuffle=False,
        ):
            self.quant_type = quant_type
            self.solution_type = solution_type
            self.config = config
            self.need_shuffle = need_shuffle

    fake_moe.AiterMoeConfig = AiterMoeConfig
    fake_moe.MoeQuantType = SimpleNamespace(
        W16A16="w16a16",
        W8A8="int8_w8a8",
        FP8_W8A8="fp8_w8a8",
    )
    fake_moe.MoeSolutionType = SimpleNamespace(
        MOE_C="moe_c",
        TRITON="triton",
    )

    def get_aiter_moe_config(**kwargs):
        captured["config_kwargs"] = kwargs
        # Exercise the no-tuned-config fallback used by MiniMax-M3/BW1100.
        return False, AiterMoeConfig(quant_type=kwargs["quant_type"])

    def aiter_moe(
        *,
        gemm1_alpha=None,
        gemm1_limit=None,
        **kwargs,
    ):
        captured["call_kwargs"] = {
            **kwargs,
            "gemm1_alpha": gemm1_alpha,
            "gemm1_limit": gemm1_limit,
        }
        return kwargs["hidden_states"]

    fake_moe.get_aiter_moe_config = get_aiter_moe_config
    fake_moe.aiter_moe = aiter_moe
    fake_moe.aiter_moe_shfl_weight = lambda w1, w2, config: (w1, w2)

    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.moe", fake_moe)


def test_aiter_activation_kwargs_reject_old_aiter_api():
    def old_aiter_moe(**kwargs):
        return kwargs

    config = MoeRunnerConfig(
        activation="silu",
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
        gate_up_interleaved=False,
    )

    with pytest.raises(RuntimeError, match="upgrade AITER"):
        get_aiter_moe_activation_kwargs(config, old_aiter_moe)


def test_aiter_activation_kwargs_reject_mismatched_minimax_layout():
    config = MoeRunnerConfig(
        activation="silu",
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
        gate_up_interleaved=True,
    )

    with pytest.raises(ValueError, match="expects split"):
        get_aiter_moe_activation_kwargs(
            config, _unified_aiter_moe_with_activation_params
        )


def test_aiter_activation_kwargs_support_minimax_m3_split_layout():
    config = MoeRunnerConfig(
        activation="silu",
        is_gated=True,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
        gate_up_interleaved=False,
    )

    assert get_aiter_moe_activation_kwargs(
        config, _unified_aiter_moe_with_activation_params
    ) == {"gemm1_alpha": 1.702, "gemm1_limit": 7.0}


def test_aiter_activation_kwargs_do_not_change_ordinary_models():
    config = MoeRunnerConfig(activation="silu")

    assert (
        get_aiter_moe_activation_kwargs(
            config, _unified_aiter_moe_with_activation_params
        )
        == {}
    )


def _unified_aiter_moe_with_activation_params(
    *, gemm1_alpha=None, gemm1_limit=None, **kwargs
):
    return gemm1_alpha, gemm1_limit


def test_unified_aiter_converts_expert_map_to_membership_mask_for_asm(
    monkeypatch,
):
    captured = {}

    class MoeSolutionType:
        ASM = "asm"

    def aiter_moe(*, gemm1_alpha=None, gemm1_limit=None, **kwargs):
        captured.update(kwargs)
        captured["gemm1_alpha"] = gemm1_alpha
        captured["gemm1_limit"] = gemm1_limit
        return kwargs["hidden_states"]

    fake_aiter_moe = ModuleType("aiter.moe")
    fake_aiter_moe.MoeSolutionType = MoeSolutionType
    fake_aiter_moe.aiter_moe = aiter_moe
    monkeypatch.setitem(sys.modules, "aiter.moe", fake_aiter_moe)

    runner = AiterRunnerCore(
        MoeRunnerConfig(
            activation="silu",
            gemm1_alpha=1.7,
            gemm1_clamp_limit=7.0,
            gate_up_interleaved=False,
        )
    )
    runner._get_unified_moe_config = lambda *args: SimpleNamespace(
        solution_type=MoeSolutionType.ASM,
        need_shuffle=False,
        need_shuffle_scale=False,
    )
    runner._get_unified_weights = lambda quant_info, config: (
        quant_info.w13_weight,
        quant_info.w2_weight,
    )
    runner._unified_quant_params = lambda quant_type: ("fp8", 0, None)

    expert_map = torch.tensor([-1, 0, 1, -1], dtype=torch.int32)
    runner._run_unified_moe(
        _runner_input(),
        _quant_info(expert_mask=expert_map),
    )

    assert captured["global_num_experts"] == 4
    assert captured["expert_map"].tolist() == [0, 1, 1, 0]
    assert captured["expert_map"].dtype == torch.int32


def test_unified_aiter_caches_normalized_bool_expert_map():
    runner = AiterRunnerCore(MoeRunnerConfig())
    sink_mask = torch.tensor([True, True, False])

    first = runner._build_expert_map(sink_mask, num_local_experts=2)
    second = runner._build_expert_map(sink_mask, num_local_experts=2)

    assert first[0] == 3
    assert first[1].tolist() == [0, 1, -1]
    assert second[1] is first[1]


def test_unified_aiter_rejects_membership_mask_as_expert_map():
    runner = AiterRunnerCore(MoeRunnerConfig())
    membership_mask = torch.tensor([0, 1, 0, 1], dtype=torch.int32)

    with pytest.raises(ValueError, match="membership mask"):
        runner._build_expert_map(membership_mask, num_local_experts=2)


def test_mori_aiter_metadata_maps_global_experts_to_local_ids():
    legacy_mask, expert_map = mori_dispatcher._build_mori_aiter_expert_metadata(
        num_experts=8,
        num_local_experts=2,
        ep_rank=2,
        device=torch.device("cpu"),
    )

    assert legacy_mask.tolist() == [0, 0, 0, 0, 1, 1, 0, 0]
    assert expert_map.tolist() == [-1, -1, -1, -1, 0, 1, -1, -1]


def test_deepep_aiter_sink_metadata_keeps_legacy_and_unified_semantics():
    from sglang.srt.layers.moe.token_dispatcher.aiter_utils import (
        build_aiter_sink_expert_metadata,
    )

    legacy_mask, unified_mask = build_aiter_sink_expert_metadata(3, torch.device("cpu"))

    assert legacy_mask.dtype == torch.int32
    assert legacy_mask.tolist() == [1, 1, 1, 0]
    assert unified_mask.dtype == torch.bool
    assert unified_mask.tolist() == [True, True, True, False]


def test_channel_fp8_aiter_prefers_standard_dispatcher_local_expert_mapping(
    monkeypatch,
):
    monkeypatch.setattr(channel_fp8_scheme, "_is_hcu", True)
    captured = {}

    class FakeRunner:
        runner_backend = MoeRunnerBackend.AITER

        def run(self, dispatch_output, quant_info):
            captured["dispatch_output"] = dispatch_output
            captured["quant_info"] = quant_info
            return dispatch_output.hidden_states

    scheme = object.__new__(CompressedTensorsW8A8Fp8MoE)
    scheme.runner = FakeRunner()
    scheme.moe_runner_config = MoeRunnerConfig(gemm1_alpha=1.702, gemm1_clamp_limit=7.0)

    expert_map = torch.tensor([-1, 0, -1, 1], dtype=torch.int32)
    membership_mask = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    layer = SimpleNamespace(
        w13_weight=torch.empty((2, 8, 4)),
        w2_weight=torch.empty((2, 4, 4)),
        w13_weight_scale=torch.ones((2, 8, 1)),
        w2_weight_scale=torch.ones((2, 4, 1)),
        w13_input_scale=None,
        w2_input_scale=None,
        dispatcher=SimpleNamespace(
            local_expert_mapping=expert_map,
            expert_mask_gpu=membership_mask,
        ),
    )
    topk_ids = torch.tensor([[1, 3]], dtype=torch.int32)
    dispatch_output = StandardDispatchOutput(
        hidden_states=torch.zeros((1, 4), dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.ones((1, 2), dtype=torch.float32),
            topk_ids=topk_ids,
            router_logits=torch.zeros((1, 4), dtype=torch.float32),
        ),
    )

    result = scheme.apply_weights(layer, dispatch_output)

    assert result is dispatch_output.hidden_states
    assert captured["dispatch_output"] is dispatch_output
    assert captured["quant_info"].expert_mask is expert_map


def test_explicit_aiter_runner_enables_padded_weight_loading(monkeypatch):
    monkeypatch.setattr(fused_moe_layer, "_use_aiter", False)
    monkeypatch.setattr(
        fused_moe_layer,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.AITER,
    )
    layer = SimpleNamespace(
        w2_weight=SimpleNamespace(weight_padded=True),
        use_flashinfer_trtllm_moe=False,
    )

    assert fused_moe_layer.FusedMoE.use_padded_loading.func(layer)


def test_explicit_aiter_runner_controls_channel_fp8_weight_layout(monkeypatch):
    monkeypatch.setattr(channel_fp8_scheme, "_use_aiter", False)
    monkeypatch.setattr(
        channel_fp8_scheme,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.AITER,
    )

    assert channel_fp8_scheme._should_use_aiter_runner()


def test_explicit_non_aiter_runner_overrides_legacy_switch(monkeypatch):
    monkeypatch.setattr(dispatcher_aiter_utils, "get_bool_env_var", lambda _: True)
    monkeypatch.setattr(dispatcher_aiter_utils, "is_hip", lambda: True)
    monkeypatch.setattr(
        dispatcher_aiter_utils,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.DEEP_GEMM,
    )

    assert not dispatcher_aiter_utils.should_use_aiter_runner()


def test_dispatchers_honor_explicit_aiter_runner(monkeypatch):
    monkeypatch.setattr(dispatcher_aiter_utils, "get_bool_env_var", lambda _: False)
    monkeypatch.setattr(dispatcher_aiter_utils, "is_hip", lambda: True)
    monkeypatch.setattr(
        dispatcher_aiter_utils,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.AITER,
    )

    assert dispatcher_aiter_utils.should_use_aiter_runner()


def test_empty_input_respects_requested_output_dtype():
    runner = AiterRunnerCore(MoeRunnerConfig(activation="silu"))
    runner_input = AiterRunnerInput(
        hidden_states=torch.empty((0, 16), dtype=torch.float8_e4m3fn),
        topk_ids=torch.empty((0, 4), dtype=torch.int32),
        topk_weights=torch.empty((0, 4), dtype=torch.float32),
        quant_type=AiterQuantType.PER_TOKEN,
        output_dtype=torch.bfloat16,
    )

    output = runner.run(runner_input, None, {})

    assert output.hidden_states.shape == (0, 16)
    assert output.hidden_states.dtype == torch.bfloat16


def test_hcu_unquantized_activation_reaches_unified_runner_before_dispatch_unpack(
    monkeypatch,
):
    from sglang.srt.layers.quantization import unquant

    captured = {}

    class FakeRunner:
        def __init__(self, backend, config):
            self.runner_backend = backend

        def run(self, dispatch_output, quant_info):
            captured.update(dispatch_output=dispatch_output, quant_info=quant_info)
            return dispatch_output.hidden_states

    monkeypatch.setattr(unquant, "MoeRunner", FakeRunner)
    monkeypatch.setattr(unquant, "_is_hcu", True)
    monkeypatch.setattr(unquant, "_is_hip", True)
    monkeypatch.setattr(unquant, "_is_npu", False)
    monkeypatch.setattr(
        unquant, "get_moe_runner_backend", lambda: MoeRunnerBackend.AITER
    )
    monkeypatch.setattr(
        unquant,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(supports_aiter=lambda: True),
    )
    method = object.__new__(unquant.UnquantizedFusedMoEMethod)
    method.use_flashinfer_trtllm_moe = False
    method.use_flashinfer_cutlass = False
    method.use_deep_gemm = False
    method.use_triton_kernels = False
    config = MoeRunnerConfig(gemm1_alpha=1.702, gemm1_clamp_limit=7.0)
    expert_map = torch.tensor([-1, 0, 1], dtype=torch.int32)
    layer = SimpleNamespace(
        intermediate_size_per_partition=128,
        moe_runner_config=config,
        w13_weight=torch.empty((2, 256, 4)),
        w2_weight=torch.empty((2, 4, 128)),
        dispatcher=SimpleNamespace(local_expert_mapping=expert_map),
    )
    method.create_moe_runner(layer, config)
    assert method._use_hcu_unified_aiter
    # DeepEP-style outputs do not expose StandardDispatchOutput.topk_output.
    dispatch_output = SimpleNamespace(hidden_states=torch.zeros((1, 4)))
    assert method.forward_cuda(layer, dispatch_output) is dispatch_output.hidden_states
    assert captured["quant_info"].expert_mask is expert_map
