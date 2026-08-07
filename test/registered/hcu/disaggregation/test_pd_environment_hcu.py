# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

from sglang.srt.disaggregation.utils import (
    KVClassType,
    TransferBackend,
    get_kv_class,
)
from sglang.test.ci.ci_register import register_hcu_ci

try:
    from .pd_hcu_utils import (
        active_rdma_devices,
        require_hcu_devices,
        require_transfer_backend,
        transfer_backend_available,
    )
except ImportError:
    from pd_hcu_utils import (
        active_rdma_devices,
        require_hcu_devices,
        require_transfer_backend,
        transfer_backend_available,
    )


register_hcu_ci(est_time=60, suite="stage-b-test-1-hcu-small")


class TestHcuPDEnvironment(unittest.TestCase):
    def test_hcu_devices_are_visible(self):
        device_names = require_hcu_devices(2)
        print(f"HCU PD visible devices ({len(device_names)}): {device_names}")
        self.assertGreaterEqual(len(device_names), 2)

    def test_rdma_devices_are_active(self):
        devices = active_rdma_devices()
        print(f"HCU PD active RDMA devices ({len(devices)}): {devices}")
        self.assertTrue(devices, "No active RDMA device was found in sysfs.")

    def test_router_is_importable(self):
        router = importlib.import_module("sglang_router")
        self.assertIsNotNone(router)

    def test_mooncake_sglang_adapter_is_importable(self):
        require_transfer_backend("mooncake")
        manager_cls = get_kv_class(TransferBackend.MOONCAKE, KVClassType.MANAGER)
        self.assertIsNotNone(manager_cls)

    def test_optional_backend_inventory(self):
        inventory = {
            backend: transfer_backend_available(backend)
            for backend in ("mooncake", "nixl", "mori")
        }
        print(f"HCU PD transport inventory: {inventory}")
        self.assertTrue(inventory["mooncake"])


if __name__ == "__main__":
    unittest.main()
