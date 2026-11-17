"""
CPU-only unit tests for the weight cache's non-state_dict state transfer.

A weight-cache engine maps the daemon's parameters over a freshly built model.
What a checkpoint load leaves in plain attributes (fix-up flags, derived
tensors, repacked weights, post-load module links) has to follow, or the engine
serves post-processed weights with bookkeeping that still says "fresh". These
tests cover that path without a GPU or a daemon process:

  - module naming that survives post-load module links
  - export -> apply round trip for scalars, tensors and containers
  - immutable (NamedTuple) state left to the model's post-load hook, and the
    refusal to serve when the hook does not rebuild it
  - the opt-outs that keep per-process state from crossing processes
  - host-memory modules handing their state out themselves
  - the HCU channel-FP8 allowlist entry and the expert-placement stamp
"""

import enum
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import NamedTuple, Optional
from unittest import mock

import torch
from torch import nn

from sglang.srt.weight_cache import extra_state
from sglang.srt.weight_cache.extra_state import (
    apply_extra_state,
    attach_host_state,
    canonical_modules,
    collect_extra_state,
    export_extra_state,
    export_host_state,
    verify_deferred_state,
)
from sglang.srt.weight_cache.protocol import (
    compute_weight_layout_stamp,
    ipc_adopts_postprocessed_layout,
    is_ipc_quant_supported,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Mode(enum.IntEnum):
    A = 1


class _Parts(NamedTuple):
    weight: torch.Tensor
    scale: float


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.converted = False
        self.packed = None
        self.parts: Optional[_Parts] = None
        self.mode = _Mode.A
        self._next_layer = None

    def load(self):
        """What a checkpoint load leaves outside state_dict."""
        self.converted = True
        self.packed = (self.linear.weight.data * 2, torch.arange(3))
        self.parts = _Parts(self.linear.weight.data + 1, 0.5)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])

    def load(self):
        for layer in self.layers:
            layer.load()
        self.bind()

    def bind(self):
        # A post-load link: registers layer 1 a second time, under layer 0.
        self.layers[0]._next_layer = self.layers[1]

    def post_ipc_weight_load(self, rebuild_parts: bool = True):
        self.bind()
        if rebuild_parts:
            for layer in self.layers:
                layer.parts = _Parts(layer.linear.weight.data + 1, 0.5)


def _daemon_and_engine():
    torch.manual_seed(0)
    daemon = _Model()
    daemon.load()
    engine = _Model()
    engine.load_state_dict(
        {k: v for k, v in daemon.state_dict().items() if "_next_layer" not in k}
    )
    return daemon, engine


def _round_trip(daemon, engine, rebuild_parts: bool = True):
    payload = export_extra_state(daemon, serialize=lambda t: t)
    result = apply_extra_state(engine, payload, deserialize=lambda h: h)
    engine.post_ipc_weight_load(rebuild_parts)
    return payload, result


class TestCanonicalModules(CustomTestCase):
    def test_post_load_link_does_not_rename_modules(self):
        model = _Model()
        model.load()
        # named_modules() reaches layer 1 through layer 0's link first.
        self.assertIn("layers.0._next_layer", dict(model.named_modules()))
        names = [name for name, _ in canonical_modules(model)]
        self.assertIn("layers.1", names)
        self.assertNotIn("layers.0._next_layer", names)
        self.assertEqual(len(names), len(set(names)))

    def test_same_names_before_and_after_loading(self):
        fresh, loaded = _Model(), _Model()
        loaded.load()
        self.assertEqual(
            [n for n, _ in canonical_modules(fresh)],
            [n for n, _ in canonical_modules(loaded)],
        )


class TestExtraStateRoundTrip(CustomTestCase):
    def test_flags_and_derived_tensors_reach_the_engine(self):
        daemon, engine = _daemon_and_engine()
        self.assertFalse(engine.layers[0].converted)
        _, result = _round_trip(daemon, engine)
        verify_deferred_state(engine, result)
        for d_layer, e_layer in zip(daemon.layers, engine.layers):
            self.assertTrue(e_layer.converted)
            self.assertIsInstance(e_layer.packed, tuple)
            self.assertEqual(len(e_layer.packed), 2)
            for d, e in zip(d_layer.packed, e_layer.packed):
                self.assertTrue(torch.equal(d, e))

    def test_small_tensors_are_copied_not_shared(self):
        daemon, engine = _daemon_and_engine()
        _round_trip(daemon, engine)
        # An engine must never write into, or read another engine's writes
        # from, a small tensor it did not allocate.
        engine.layers[0].packed[0].zero_()
        self.assertTrue(bool(daemon.layers[0].packed[0].abs().sum() > 0))

    def test_namedtuple_state_is_left_to_the_hook_and_verified(self):
        daemon, engine = _daemon_and_engine()
        _, result = _round_trip(daemon, engine)
        self.assertTrue(result.deferred_tensors)
        verify_deferred_state(engine, result)  # the hook rebuilt it
        self.assertIsInstance(engine.layers[0].parts, _Parts)

    def test_refuses_to_serve_when_the_hook_does_not_rebuild(self):
        daemon, engine = _daemon_and_engine()
        _, result = _round_trip(daemon, engine, rebuild_parts=False)
        with self.assertRaises(RuntimeError) as ctx:
            verify_deferred_state(engine, result)
        self.assertIn("layers.0.parts.weight", str(ctx.exception))

    def test_enums_are_not_exported_as_scalars(self):
        # An enum would pickle as its class, which the wire format rejects.
        daemon, _ = _daemon_and_engine()
        state = collect_extra_state(daemon)
        paths = {extra_state.format_path(p) for p in state["scalars"]}
        self.assertIn("layers.0.converted", paths)
        self.assertNotIn("layers.0.mode", paths)

    def test_empty_payload_is_a_noop(self):
        _, engine = _daemon_and_engine()
        result = apply_extra_state(engine, None, deserialize=lambda h: h)
        verify_deferred_state(engine, result)
        self.assertFalse(engine.layers[0].converted)


class TestPerProcessStateOptOuts(CustomTestCase):
    def test_skip_attrs_and_skip_classes_are_not_walked(self):
        class Handle:
            __module__ = "sglang.srt.fake"
            _weight_cache_skip = True

            def __init__(self):
                self.address = 0xDEAD

        class Mod(nn.Module):
            _weight_cache_skip_attrs = ("device_ptr",)

            def __init__(self):
                super().__init__()
                self.device_ptr = 0xBEEF
                self.handle = Handle()
                self.kept = 7

        state = collect_extra_state(Mod())
        paths = {extra_state.format_path(p) for p in state["scalars"]}
        self.assertEqual(paths, {"<root>.kept"})

    def test_runtime_infrastructure_modules_are_not_walked(self):
        class Group:
            __module__ = "sglang.srt.distributed.fake"

            def __init__(self):
                self.port = 12345

        class Mod(nn.Module):
            def __init__(self):
                super().__init__()
                self.tp_group = Group()

        state = collect_extra_state(Mod())
        self.assertEqual(state["scalars"], {})
        self.assertIn("Group", set(state["opaque"].values()))


class _HostModule(nn.Module):
    def __init__(self, loaded: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.loaded = loaded
        self.attached = None

    def export_ipc_host_state(self):
        return {"handle": b"h"} if self.loaded else None

    def attach_ipc_host_state(self, desc):
        self.attached = desc


class TestHostState(CustomTestCase):
    def test_covered_tensors_are_named_and_handed_to_the_module(self):
        daemon = nn.ModuleDict({"table": _HostModule(True), "fc": nn.Linear(2, 2)})
        host_state, covered = export_host_state(daemon)
        self.assertEqual(host_state, {"table": {"handle": b"h"}})
        self.assertEqual(covered, {"table.weight"})

        engine = nn.ModuleDict({"table": _HostModule(False), "fc": nn.Linear(2, 2)})
        self.assertEqual(attach_host_state(engine, host_state), 1)
        self.assertEqual(engine["table"].attached, {"handle": b"h"})

    def test_host_state_for_an_unknown_module_is_an_error(self):
        engine = nn.ModuleDict({"fc": nn.Linear(2, 2)})
        with self.assertRaises(RuntimeError):
            attach_host_state(engine, {"table": {"handle": b"h"}})


_CHANNEL_FP8 = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "config_groups": {
        "group_0": {
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "channel",
                "dynamic": False,
            }
        }
    },
}


class TestHcuChannelFp8Allowlist(CustomTestCase):
    def test_supported_on_hcu_only(self):
        with mock.patch("sglang.srt.utils.is_hcu", return_value=True):
            self.assertTrue(is_ipc_quant_supported("compressed-tensors", _CHANNEL_FP8))
        with mock.patch("sglang.srt.utils.is_hcu", return_value=False):
            self.assertFalse(is_ipc_quant_supported("compressed-tensors", _CHANNEL_FP8))

    def test_other_compressed_tensors_schemes_stay_rejected(self):
        w4 = {
            **_CHANNEL_FP8,
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "type": "int", "strategy": "group"}
                }
            },
        }
        packed = {**_CHANNEL_FP8, "format": "pack-quantized"}
        with mock.patch("sglang.srt.utils.is_hcu", return_value=True):
            self.assertFalse(is_ipc_quant_supported("compressed-tensors", w4))
            self.assertFalse(is_ipc_quant_supported("compressed-tensors", packed))
            self.assertFalse(is_ipc_quant_supported("compressed-tensors", None))

    def test_only_this_method_adopts_the_daemon_layout(self):
        self.assertTrue(ipc_adopts_postprocessed_layout("compressed-tensors"))
        self.assertFalse(ipc_adopts_postprocessed_layout("fp8"))
        self.assertFalse(ipc_adopts_postprocessed_layout(""))

    def test_repacked_deleted_params_are_dropped_on_client(self):
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

        class _MoELayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.w13_weight = nn.Parameter(torch.ones(2, 4), requires_grad=False)
                self.w13_weight_scale = nn.Parameter(
                    torch.ones(2, 1), requires_grad=False
                )

            def repack(self):
                self.register_buffer(
                    "w13_weight_deepgemm", self.w13_weight * 2, persistent=False
                )
                self._dsv4_channel_fp8_deepgemm_repacked = True
                del self.w13_weight

        daemon_model = nn.ModuleDict({"moe": _MoELayer()})
        daemon_model["moe"].repack()
        extra = export_extra_state(daemon_model, serialize=lambda t: t)
        entries = {
            "moe.w13_weight_scale": {
                "tensor": daemon_model["moe"].w13_weight_scale.data,
                "is_param": True,
            },
            "moe.w13_weight_deepgemm": {
                "tensor": daemon_model["moe"].w13_weight_deepgemm,
                "is_param": False,
            },
        }
        loader = IpcModelLoader(LoadConfig(load_format=LoadFormat.IPC_CACHE))
        model_cfg = SimpleNamespace(
            dtype=torch.float32,
            quantization="compressed-tensors",
            hf_config=SimpleNamespace(quantization_config=_CHANNEL_FP8),
        )
        with (
            mock.patch(
                "sglang.srt.weight_cache.ipc_loader._initialize_model",
                side_effect=lambda *a, **kw: nn.ModuleDict({"moe": _MoELayer()}),
            ),
            mock.patch.object(
                loader._transport_backend,
                "import_tensor",
                side_effect=lambda entry: entry["tensor"],
            ),
        ):
            client = loader._load_zero_copy_mode(
                model_cfg,
                device_config=None,
                entries=entries,
                quant_config=None,
                extra_state=extra,
            )
        self.assertFalse(hasattr(client["moe"], "w13_weight"))
        self.assertTrue(
            torch.equal(
                client["moe"].w13_weight_deepgemm,
                daemon_model["moe"].w13_weight_deepgemm,
            )
        )


class TestWeightLayoutStamp(CustomTestCase):
    def _stamp(self, location, redundant):
        moe = SimpleNamespace(
            init_expert_location=location, ep_num_redundant_experts=redundant
        )
        with mock.patch(
            "sglang.srt.runtime_context.get_exec",
            return_value=SimpleNamespace(moe=moe),
        ):
            return compute_weight_layout_stamp()

    def test_placement_file_is_hashed_by_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "placement.pt")
            with open(path, "wb") as f:
                f.write(b"placement-a")
            first = self._stamp(path, 32)
            self.assertEqual(first, self._stamp(path, 32))
            with open(path, "wb") as f:
                f.write(b"placement-b")
            # Regenerated under the same name: a different cache identity.
            self.assertNotEqual(first, self._stamp(path, 32))

    def test_redundant_experts_and_literal_location_change_the_stamp(self):
        self.assertNotEqual(self._stamp("trivial", 0), self._stamp("trivial", 32))
        self.assertNotEqual(self._stamp("trivial", 0), self._stamp("other", 0))


class TestDeepseekV4RopeMetaCache(CustomTestCase):
    def test_meta_init_does_not_poison_deepseek_v4_rope_cache(self):
        from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis

        precompute_freqs_cis.cache_clear()
        args = (64, 128, 64, 10000.0, 16.0, 32, 1)
        with torch.device("meta"):
            meta_freqs = precompute_freqs_cis(*args)
        self.assertEqual(meta_freqs.device.type, "meta")
        cpu_freqs = precompute_freqs_cis(*args)
        self.assertEqual(cpu_freqs.device.type, "cpu")
        self.assertEqual(cpu_freqs.shape, meta_freqs.shape)


if __name__ == "__main__":
    unittest.main()
