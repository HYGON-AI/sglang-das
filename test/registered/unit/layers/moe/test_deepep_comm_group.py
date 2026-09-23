"""CPU checks for DeepEP communicator isolation and async configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed import parallel_state
from sglang.srt.layers.moe.fused_moe_triton import layer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepEPCommGroup(CustomTestCase):
    def test_full_ep_aliases_tp_by_default(self):
        with patch.object(
            parallel_state.envs.SGLANG_DEEPEP_USE_MOE_EP_GROUP,
            "get",
            return_value=False,
        ):
            self.assertTrue(parallel_state.should_reuse_tp_group_for_full_moe_ep(8, 8))

    def test_full_ep_does_not_alias_tp_when_isolated(self):
        with patch.object(
            parallel_state.envs.SGLANG_DEEPEP_USE_MOE_EP_GROUP,
            "get",
            return_value=True,
        ):
            self.assertFalse(parallel_state.should_reuse_tp_group_for_full_moe_ep(8, 8))

    def test_default_uses_tp_group(self):
        tp_group = object()
        with (
            patch.object(
                layer.envs.SGLANG_DEEPEP_USE_MOE_EP_GROUP,
                "get",
                return_value=False,
            ),
            patch.object(
                layer,
                "get_tp_group",
                return_value=SimpleNamespace(device_group=tp_group),
            ),
        ):
            result = layer._get_deepep_comm_group(
                SimpleNamespace(is_mori=lambda: False)
            )
        self.assertIs(result, tp_group)

    def test_isolation_uses_moe_ep_group(self):
        tp_group = object()
        ep_group = object()
        with (
            patch.object(
                layer.envs.SGLANG_DEEPEP_USE_MOE_EP_GROUP,
                "get",
                return_value=True,
            ),
            patch.object(
                layer,
                "get_tp_group",
                return_value=SimpleNamespace(device_group=tp_group),
            ),
            patch.object(
                layer,
                "get_moe_ep_group",
                return_value=SimpleNamespace(device_group=ep_group),
            ),
            patch.object(layer, "print_info_once"),
        ):
            result = layer._get_deepep_comm_group(
                SimpleNamespace(is_mori=lambda: False)
            )
        self.assertIs(result, ep_group)


if __name__ == "__main__":
    unittest.main()
