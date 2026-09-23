import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _target_checkpoint_bundles_dspark_draft,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_utils import (
    get_dflash_attention_sliding_window_size,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_BUNDLED_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
_PLAIN_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash"


def _bundled_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=256,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_noise_token_id=128799,
    )


def _plain_hf_config() -> SimpleNamespace:
    return SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])


def _make_dspark_server_args(
    *, model_path: str, hf_config: SimpleNamespace
) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = model_path
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    server_args.speculative_draft_model_path = None
    server_args.speculative_dspark_block_size = 5
    server_args.model_config = SimpleNamespace(hf_config=hf_config)
    return server_args


def _enable_pd_prefill_cp(server_args: ServerArgs) -> None:
    server_args.disaggregation_mode = "prefill"
    server_args.disaggregation_transfer_backend = "mooncake"
    server_args.tp_size = 4
    server_args.dp_size = 1
    server_args.pp_size = 1
    server_args.attn_cp_size = 4
    server_args.enable_prefill_cp = True
    server_args.cp_strategy = "interleave"
    server_args.enable_dp_attention = True
    server_args.enable_dp_lm_head = False


class TestTargetCheckpointBundlesDsparkDraft(CustomTestCase):
    def test_bundled_dsv4_config_is_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        self.assertTrue(_target_checkpoint_bundles_dspark_draft(server_args))

    def test_plain_target_config_is_not_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        self.assertFalse(_target_checkpoint_bundles_dspark_draft(server_args))


class TestDsparkDraftPathDefaulting(CustomTestCase):
    def test_bundled_checkpoint_defaults_draft_path_to_model_path(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        _handle_dspark(server_args)
        self.assertEqual(server_args.speculative_draft_model_path, _BUNDLED_MODEL_PATH)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_plain_target_without_draft_path_raises(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        with self.assertRaises(ValueError):
            _handle_dspark(server_args)

    def test_explicit_draft_path_is_not_overwritten(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.speculative_draft_model_path = "deepseek-ai/some-other-dspark-draft"
        _handle_dspark(server_args)
        self.assertEqual(
            server_args.speculative_draft_model_path,
            "deepseek-ai/some-other-dspark-draft",
        )

    def test_pd_prefill_cp_does_not_require_dp_lm_head(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        _enable_pd_prefill_cp(server_args)

        _handle_dspark(server_args)

        self.assertEqual(server_args.speculative_draft_model_path, _BUNDLED_MODEL_PATH)

    def test_context_parallel_remains_rejected_outside_pd_prefill(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        _enable_pd_prefill_cp(server_args)
        server_args.disaggregation_mode = "decode"

        with self.assertRaisesRegex(
            ValueError, "only supported for DeepSeek-V4 PD prefill"
        ):
            _handle_dspark(server_args)

    def test_decode_target_deepep_allows_draft_none(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.enable_dp_attention = True
        server_args.enable_dp_lm_head = True
        server_args.moe_a2a_backend = "deepep"
        server_args.moe_runner_backend = "deep_gemm"
        server_args.speculative_moe_a2a_backend = "none"

        _handle_dspark(server_args)


class TestDsparkDpAttentionMoeA2aGate(CustomTestCase):
    """Gate contract for DSpark + dp attention + MoE a2a backends."""

    def _dp_server_args(self, *, moe_a2a_backend: str) -> ServerArgs:
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.enable_dp_attention = True
        server_args.enable_dp_lm_head = True
        server_args.dp_size = 2
        server_args.tp_size = 2
        server_args.moe_a2a_backend = moe_a2a_backend
        return server_args

    def test_ep_a2a_backends_are_admitted(self):
        """Static verification admits the supported token A2A backends."""
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            for backend in ("deepep", "megamoe"):
                _handle_dspark(self._dp_server_args(moe_a2a_backend=backend))
            for backend in ("pplx",):
                with self.assertRaisesRegex(ValueError, backend):
                    _handle_dspark(self._dp_server_args(moe_a2a_backend=backend))

    def test_a2a_backend_with_compact_verify_mode_raises(self):
        for backend in ("deepep", "megamoe"):
            server_args = self._dp_server_args(moe_a2a_backend=backend)
            with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
                with self.assertRaisesRegex(ValueError, "static"):
                    _handle_dspark(server_args)


class TestDsparkSlidingWindowCompatibility(CustomTestCase):
    def test_official_dspark_swa_field_is_used_when_qwen3_drops_top_level(self):
        config = SimpleNamespace(
            layer_types=["sliding_attention"] * 6,
            sliding_window=None,
            dflash_config={"use_swa": True, "swa_window_size": 1024},
        )
        self.assertEqual(get_dflash_attention_sliding_window_size(config), 1023)


if __name__ == "__main__":
    unittest.main()
