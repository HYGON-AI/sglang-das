# Copyright (c) 2026 gencheng liu
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sglang.kernels.ops.moe import w4a8_deepep_aiter


class TestW4A8DeepEPAiterLoader(unittest.TestCase):
    def tearDown(self):
        w4a8_deepep_aiter._load_extension.cache_clear()

    def test_aiter_source_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for header in w4a8_deepep_aiter._REQUIRED_AITER_HEADERS:
                (root / header).touch()
            with patch.dict(os.environ, {"AITER_MOE_SRC": directory}):
                self.assertEqual(
                    w4a8_deepep_aiter._find_aiter_moe_sources(), root.resolve()
                )

    def test_target_arch_uses_first_configured_target(self):
        with patch.dict(os.environ, {"PYTORCH_ROCM_ARCH": "gfx936:sramecc-;gfx938"}):
            self.assertEqual(w4a8_deepep_aiter._target_arch(), "gfx936")

    def test_wrappers_delegate_to_loaded_extension(self):
        extension = Mock()
        with patch.object(w4a8_deepep_aiter, "_load_extension", return_value=extension):
            w4a8_deepep_aiter.w4a8_mmac_contiguous_out("normal")
            w4a8_deepep_aiter.w4a8_mmac_masked_out("masked")

        extension.w4a8_mmac_contiguous_out.assert_called_once_with("normal")
        extension.w4a8_mmac_masked_out.assert_called_once_with("masked")

    def test_loader_restores_compiler_environment(self):
        w4a8_deepep_aiter._load_extension.cache_clear()
        with tempfile.TemporaryDirectory() as directory:
            compiler = Path(directory) / "aicc"
            compiler.touch()
            environment = {
                "HCU_EXTENSION_COMPILER": str(compiler),
                "PYTORCH_ROCM_ARCH": "gfx936",
                "PYTORCH_NVCC": "previous-compiler",
            }
            extension = object()
            with (
                patch.dict(os.environ, environment),
                patch.object(
                    w4a8_deepep_aiter,
                    "_find_aiter_moe_sources",
                    return_value=Path(directory),
                ),
                patch("torch.utils.cpp_extension.load", return_value=extension) as load,
            ):
                self.assertIs(w4a8_deepep_aiter._load_extension(), extension)
                self.assertEqual(os.environ["PYTORCH_NVCC"], "previous-compiler")
                self.assertEqual(os.environ["PYTORCH_ROCM_ARCH"], "gfx936")

            self.assertEqual(
                load.call_args.kwargs["name"], "sglang_w4a8_deepep_aiter_gfx936"
            )


if __name__ == "__main__":
    unittest.main()
