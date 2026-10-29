"""Checkpoint metadata guards run without Torch or an accelerator."""

import copy
import importlib.util
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/layers/quantization"
spec = importlib.util.spec_from_file_location(
    "checkpoint_probe", SOURCE / "dsv41_channelwise_fp8_checkpoint.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
WEIGHTS = ["layers.0.attn.wo_a.weight", "head.weight", "mtp.2.markov_head.head.weight"]

if importlib.util.find_spec("torch") is not None:
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CheckpointFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        probe.checkpoint_supports_channelwise_fp8.cache_clear()

    def checkpoint(self, dtype="F8_E4M3", suffix=".scale", indexed=False):
        header = {}
        for name in WEIGHTS:
            header[name] = {"dtype": dtype, "shape": [4, 8]}
            header[name.removesuffix(".weight") + suffix] = {
                "dtype": "F32",
                "shape": [4, 1],
            }
        self.write(header, indexed=indexed)
        return header

    def write(self, header, indexed=False):
        offset = 0
        for meta in header.values():
            size = 1
            for dim in meta["shape"]:
                size *= dim
            size *= {"F8_E4M3": 1, "BF16": 2, "F32": 4}[meta["dtype"]]
            meta["data_offsets"] = [offset, offset + size]
            offset += size
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        (self.root / "model.safetensors").write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + bytes(offset)
        )
        if indexed:
            (self.root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {k: "model.safetensors" for k in header}})
            )
        probe.checkpoint_supports_channelwise_fp8.cache_clear()


class TestCheckpointMetadata(_CheckpointFixture, unittest.TestCase):
    def test_three_channelwise_projections_and_scale_aliases(self):
        for suffix in (".scale", ".weight_scale_inv", ".weight_scale"):
            with self.subTest(suffix=suffix):
                self.checkpoint(suffix=suffix)
                self.assertTrue(
                    probe.checkpoint_supports_channelwise_fp8(str(self.root))
                )

    def test_indexed_checkpoint(self):
        self.checkpoint(indexed=True)
        self.assertTrue(probe.checkpoint_supports_channelwise_fp8(str(self.root)))

    def test_plain_fp8_models_bf16_special_projections_stay_disabled(self):
        self.checkpoint(dtype="BF16", indexed=True)
        self.assertFalse(probe.checkpoint_supports_channelwise_fp8(str(self.root)))

    def test_mixed_special_projection_dtypes_do_not_enable_all_three(self):
        header = self.checkpoint()
        header[WEIGHTS[-1]]["dtype"] = "BF16"
        self.write(header)
        self.assertFalse(probe.checkpoint_supports_channelwise_fp8(str(self.root)))

    def test_every_wo_a_must_match(self):
        header = self.checkpoint()
        header["layers.1.attn.wo_a.weight"] = {"dtype": "BF16", "shape": [4, 8]}
        self.write(header)
        self.assertFalse(probe.checkpoint_supports_channelwise_fp8(str(self.root)))

    def test_missing_scales_are_not_eligible(self):
        header = self.checkpoint()
        del header["head.scale"]
        self.write(header)
        self.assertFalse(probe.checkpoint_supports_channelwise_fp8(str(self.root)))

    def test_block_scales_are_not_channelwise(self):
        header = self.checkpoint()
        header["head.scale"]["shape"] = [1, 1]
        self.write(header)
        self.assertFalse(probe.checkpoint_supports_channelwise_fp8(str(self.root)))

    def test_missing_markov_head_is_not_eligible(self):
        header = self.checkpoint()
        del header[WEIGHTS[-1]]
        self.write(header)
        self.assertFalse(probe.checkpoint_supports_channelwise_fp8(str(self.root)))

    def test_stale_scale_names_in_index_use_actual_shard_headers(self):
        self.checkpoint(suffix=".weight_scale_inv", indexed=True)
        index = self.root / "model.safetensors.index.json"
        data = json.loads(index.read_text())
        data["weight_map"] = {
            k.replace(".weight_scale_inv", ".scale"): v
            for k, v in data["weight_map"].items()
        }
        index.write_text(json.dumps(data))
        self.assertTrue(probe.checkpoint_supports_channelwise_fp8(str(self.root)))

    def test_nonlocal_and_truncated_checkpoints_are_not_guessed(self):
        self.assertFalse(
            probe.checkpoint_supports_channelwise_fp8("hub-org/nonlocal-model")
        )
        (self.root / "model.safetensors").write_bytes(b"bad")
        self.assertFalse(probe.checkpoint_supports_channelwise_fp8(str(self.root)))


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Torch CPU runtime required for dispatch tests"
)
class TestAutomaticDispatch(_CheckpointFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        from sglang.srt.layers.quantization import dsv41_channelwise_fp8 as runtime

        self.runtime = runtime
        self.config = SimpleNamespace(
            get_name=lambda: "fp8",
            is_checkpoint_fp8_serialized=True,
            weight_block_size=None,
            use_mxfp8=False,
            activation_scheme="dynamic",
        )
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("SGLANG_DSV41_CHANNELWISE_FP8_RUNTIME", None)
        hardware = patch("sglang.srt.utils.is_hcu", return_value=True)
        hardware.start()
        self.addCleanup(hardware.stop)

    def test_default_without_export_enables_real_fp8_headers(self):
        self.checkpoint(indexed=True)
        self.assertTrue(self.runtime.enabled(self.config, model_path=str(self.root)))

    def test_default_does_not_enable_bf16_checkpoint(self):
        self.checkpoint(dtype="BF16")
        self.assertFalse(self.runtime.enabled(self.config, model_path=str(self.root)))

    def test_explicit_zero_overrides_eligible_checkpoint(self):
        self.checkpoint()
        os.environ["SGLANG_DSV41_CHANNELWISE_FP8_RUNTIME"] = "0"
        self.assertFalse(self.runtime.enabled(self.config, model_path=str(self.root)))

    def test_explicit_one_preserves_custom_loader_override(self):
        os.environ["SGLANG_DSV41_CHANNELWISE_FP8_RUNTIME"] = "1"
        self.assertTrue(self.runtime.enabled(self.config, model_path="custom-loader"))

    def test_auto_with_bound_model_path(self):
        self.checkpoint()
        self.runtime.configure_checkpoint(
            self.config, SimpleNamespace(_name_or_path=str(self.root))
        )
        self.assertTrue(self.runtime.enabled(self.config))

    def test_target_and_separate_draft_do_not_share_auto_decision(self):
        self.checkpoint()
        target_path = str(self.root)
        draft_temp = tempfile.TemporaryDirectory()
        self.addCleanup(draft_temp.cleanup)
        self.root = Path(draft_temp.name)
        self.checkpoint(dtype="BF16")
        target = copy.deepcopy(self.config)
        draft = copy.deepcopy(self.config)
        self.runtime.configure_checkpoint(
            target, SimpleNamespace(_name_or_path=target_path)
        )
        self.runtime.configure_checkpoint(
            draft, SimpleNamespace(_name_or_path=str(self.root))
        )
        with patch(
            "sglang.srt.server_args.get_global_server_args",
            return_value=SimpleNamespace(model_path=target_path),
        ):
            self.assertTrue(self.runtime.enabled(target))
            self.assertFalse(self.runtime.enabled(draft))
        self.runtime.configure_checkpoint(
            target, SimpleNamespace(_name_or_path=str(self.root))
        )
        self.runtime.configure_checkpoint(
            draft, SimpleNamespace(_name_or_path=target_path)
        )
        self.assertFalse(self.runtime.enabled(target))
        self.assertTrue(self.runtime.enabled(draft))

    def test_unknown_model_path_does_not_guess_from_target(self):
        self.runtime.configure_checkpoint(self.config, SimpleNamespace())
        self.assertFalse(self.runtime.enabled(self.config))

    def test_auto_does_not_change_other_formats(self):
        self.assertFalse(self.runtime.enabled(None))
        self.config.weight_block_size = [128, 128]
        self.assertFalse(self.runtime.enabled(self.config, model_path=str(self.root)))
        os.environ["SGLANG_DSV41_CHANNELWISE_FP8_RUNTIME"] = "1"
        with self.assertRaises(ValueError):
            self.runtime.enabled(self.config)


if __name__ == "__main__":
    unittest.main()
