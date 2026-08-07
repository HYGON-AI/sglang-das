# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_hcu_ci

try:
    from .pd_hcu_utils import (
        backend_module_name,
        parse_backend_names,
        resolve_model_path,
        resolve_rdma_args,
    )
except ImportError:
    from pd_hcu_utils import (
        backend_module_name,
        parse_backend_names,
        resolve_model_path,
        resolve_rdma_args,
    )


register_hcu_ci(est_time=30, suite="stage-b-test-1-hcu-small")


class TestHcuPDUtils(unittest.TestCase):
    def test_parse_backend_names_normalizes_and_deduplicates(self):
        self.assertEqual(
            parse_backend_names(" Mooncake,nixl,mooncake "),
            ["mooncake", "nixl"],
        )

    def test_parse_backend_names_rejects_unknown_backend(self):
        with self.assertRaisesRegex(ValueError, "Unsupported HCU PD backend"):
            parse_backend_names("mooncake,unknown")

    def test_backend_module_mapping(self):
        self.assertEqual(backend_module_name("mooncake"), "mooncake")
        self.assertEqual(backend_module_name("nixl"), "nixl")
        self.assertEqual(backend_module_name("mori"), "mori")

    def test_resolve_model_path_accepts_existing_override(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with patch.dict(
                os.environ,
                {"SGLANG_HCU_PD_MODEL": model_dir},
                clear=False,
            ):
                self.assertEqual(resolve_model_path(), model_dir)

    def test_resolve_model_path_rejects_missing_override(self):
        with patch.dict(
            os.environ,
            {"SGLANG_HCU_PD_MODEL": "/missing/hcu-pd-model"},
            clear=False,
        ):
            with self.assertRaisesRegex(AssertionError, "missing directory"):
                resolve_model_path()

    def test_resolve_rdma_args_prefers_explicit_environment(self):
        with patch.dict(
            os.environ,
            {"SGLANG_TEST_RDMA_DEVICE": "mlx5_1,mlx5_5"},
            clear=False,
        ):
            self.assertEqual(
                resolve_rdma_args(),
                ["--disaggregation-ib-device", "mlx5_1,mlx5_5"],
            )


if __name__ == "__main__":
    unittest.main()
