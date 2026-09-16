"""Regression tests for Hunyuan-V4 NEXTN weight loading configuration."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from sglang.srt.models.hunyuan_v4_nextn import _mtp_quant_config


class TestHunyuanV4NextNQuantConfig(CustomTestCase):
    def test_mtp_ignore_paths_follow_draft_decoder_without_broad_exclusion(self):
        fp8_config = object()
        quant_config = SimpleNamespace(
            ignored_layers=[
                "model.mtp.layers.0.self_attn.q_proj",
                "model.mtp_layers.0.self_attn.q_proj",
            ],
            ignore=["model.mtp.layers.0.self_attn.linear_gate"],
            linear_fp8_config=fp8_config,
        )

        mtp_config = _mtp_quant_config(quant_config)

        self.assertIsNot(mtp_config, quant_config)
        self.assertEqual(
            quant_config.ignored_layers,
            [
                "model.mtp.layers.0.self_attn.q_proj",
                "model.mtp_layers.0.self_attn.q_proj",
            ],
        )
        self.assertEqual(
            mtp_config.ignored_layers,
            ["model.decoder.self_attn.q_proj"],
        )
        self.assertEqual(
            mtp_config.ignore,
            ["model.decoder.self_attn.linear_gate"],
        )
        self.assertIs(quant_config.linear_fp8_config, fp8_config)
        self.assertIsNone(mtp_config.linear_fp8_config)
        self.assertNotIn("model.decoder", mtp_config.ignored_layers)
        self.assertNotIn("model.decoder", mtp_config.ignore)
        self.assertFalse(
            any(target.startswith("re:") for target in mtp_config.ignored_layers)
        )

        self.assertTrue(
            should_ignore_layer(
                "model.decoder.self_attn.q_proj", mtp_config.ignored_layers
            )
        )
        self.assertTrue(
            should_ignore_layer(
                "model.decoder.self_attn.linear_gate", mtp_config.ignore
            )
        )
        self.assertFalse(
            should_ignore_layer(
                "model.decoder.self_attn.q_a_proj", mtp_config.ignored_layers
            )
        )

    def test_all_supported_mtp_prefixes_are_normalized(self):
        quant_config = SimpleNamespace(
            ignored_layers=[
                "model.mtp.layers.0.eh_proj",
                "model.mtp_layers.0.enorm",
                "mtp.layers.0.hnorm",
                "mtp_layers.0.shared_head.norm",
            ],
            ignore=[],
            linear_fp8_config=None,
        )

        mtp_config = _mtp_quant_config(quant_config)

        self.assertEqual(
            mtp_config.ignored_layers,
            [
                "model.decoder.eh_proj",
                "model.decoder.enorm",
                "model.decoder.hnorm",
                "model.decoder.shared_head.norm",
            ],
        )

    def test_none_quant_config_stays_none(self):
        self.assertIsNone(_mtp_quant_config(None))


if __name__ == "__main__":
    unittest.main()
