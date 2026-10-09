"""Regression tests for padded Engram IDLE batches."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import engram
from sglang.srt.layers.engram import EngramHasher
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestEngramHCUDefaults(CustomTestCase):
    def test_fused_gate_defaults_on_only_for_hcu_and_can_be_disabled(self):
        tree = ast.parse(Path(engram.__file__).read_text())
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_use_hcu_fused_engram_gate"
                for target in node.targets
            )
        )
        code = compile(
            ast.Module(body=[assignment], type_ignores=[]), engram.__file__, "exec"
        )
        for hcu in (False, True):
            for flag in (None, "0", "1"):
                with self.subTest(hcu=hcu, flag=flag), patch.dict(os.environ):
                    if flag is None:
                        os.environ.pop("SGLANG_HCU_OPT_ENGRAM_GATE", None)
                    else:
                        os.environ["SGLANG_HCU_OPT_ENGRAM_GATE"] = flag
                    namespace = {
                        "is_hcu": lambda: hcu,
                        "get_bool_env_var": engram.get_bool_env_var,
                    }
                    exec(code, namespace)
                    self.assertEqual(
                        namespace["_use_hcu_fused_engram_gate"], hcu and flag != "0"
                    )


class TestEngramIdle(CustomTestCase):
    def test_padded_idle_returns_dummy_hashes_without_committing_history(self):
        hasher = EngramHasher.__new__(EngramHasher)
        torch.nn.Module.__init__(hasher)
        hasher.max_ngram_size = 4
        hasher.history = torch.full((3, 3), 17, dtype=torch.int32)
        hasher.primes = torch.empty((2, 3, 4), dtype=torch.int64)
        hasher.offsets = torch.empty((2, 12), dtype=torch.int64)

        history_before = hasher.history.clone()
        input_ids = torch.tensor([11, 22, 33], dtype=torch.int64)
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            req_pool_indices=torch.empty(0, dtype=torch.int64),
        )

        output = hasher(input_ids, forward_batch)

        self.assertEqual(output.shape, (3, 2, 12))
        self.assertEqual(output.dtype, torch.int64)
        self.assertEqual(output.device, input_ids.device)
        self.assertTrue(torch.count_nonzero(output).item() == 0)
        self.assertTrue(torch.equal(hasher.history, history_before))


if __name__ == "__main__":
    import unittest

    unittest.main()
