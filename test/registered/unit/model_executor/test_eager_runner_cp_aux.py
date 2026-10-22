"""CPU regression tests for CP outputs with captured auxiliary states."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
from sglang.srt.layers.cp import utils as cp_utils
from sglang.srt.model_executor.runner import eager_runner

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestEagerRunnerCpAux(CustomTestCase):
    def _run(self, aux, *, capture=True, last_rank=True, plain=False):
        hidden = torch.arange(8, dtype=torch.float32).view(2, 4)
        pre_hc = torch.arange(32, dtype=torch.float32).view(2, 16)
        body = hidden if plain else (hidden, pre_hc)
        result = (body, aux) if capture else body
        logits = mock.Mock(return_value=object())
        model = SimpleNamespace(
            model=mock.Mock(return_value=result),
            capture_aux_hidden_states=capture,
            pp_group=SimpleNamespace(is_last_rank=last_rank),
            logits_processor=logits,
            lm_head=object(),
        )
        runner = object.__new__(eager_runner.EagerRunner)
        runner.model_runner = SimpleNamespace(model=model)
        batch = SimpleNamespace(input_ids=torch.arange(2), positions=torch.arange(2))
        strategy = SimpleNamespace(
            gather_hidden_states=mock.Mock(
                side_effect=lambda tensor, *_: torch.cat((tensor, tensor), dim=0)
            )
        )
        with (
            mock.patch.object(
                eager_runner,
                "cp_shard_model_inputs",
                return_value=contextlib.nullcontext(
                    (hidden, batch.positions, batch.input_ids)
                ),
            ),
            mock.patch.object(cp_utils, "is_cp_active", return_value=True),
            mock.patch.object(cp_utils, "get_cp_strategy", return_value=strategy),
            mock.patch.object(eager_runner.torch.cuda, "current_stream"),
        ):
            actual = runner._execute_extend_cp(batch, {"input_embeds": hidden})
        return actual, logits, strategy, hidden, pre_hc, body

    def test_aux_list_drops_unused_pre_hc_before_gather(self):
        aux = [torch.full((2, 4), float(i)) for i in range(3)]
        actual, logits, strategy, hidden, pre_hc, _ = self._run(aux)
        self.assertIs(actual, logits.return_value)
        args, kwargs = logits.call_args
        torch.testing.assert_close(args[1], torch.cat((hidden, hidden)))
        self.assertEqual(len(args[4]), len(aux))
        for actual_aux, expected_aux in zip(args[4], aux):
            torch.testing.assert_close(
                actual_aux, torch.cat((expected_aux, expected_aux))
            )
        self.assertNotIn("hidden_states_before_norm", kwargs)
        self.assertEqual(strategy.gather_hidden_states.call_count, 1 + len(aux))
        self.assertFalse(
            any(
                call.args[0] is pre_hc
                for call in strategy.gather_hidden_states.call_args_list
            )
        )

    def test_aux_tensor_preserves_logits_and_aux(self):
        aux = torch.ones(2, 4)
        _, logits, strategy, hidden, _, _ = self._run(aux)
        args, kwargs = logits.call_args
        torch.testing.assert_close(args[1], torch.cat((hidden, hidden)))
        torch.testing.assert_close(args[4], torch.cat((aux, aux)))
        self.assertNotIn("hidden_states_before_norm", kwargs)
        self.assertEqual(strategy.gather_hidden_states.call_count, 2)

    def test_packed_aux_preserves_capture_order_and_single_gather(self):
        captures = [
            torch.arange(8).reshape(2, 4).float() + i * 10 for i in range(3)
        ]
        packer = AuxHiddenStatePacker(len(captures))
        for capture in captures:
            packer.append(capture)
        _, logits, strategy, _, _, _ = self._run(packer.finalize())
        gathered = logits.call_args.args[4]
        expected = torch.cat(
            [torch.cat((capture, capture)) for capture in captures], dim=-1
        )
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
        self.assertEqual(strategy.gather_hidden_states.call_count, 2)

    def test_without_aux_preserves_pre_hc_gather(self):
        for capture in (False, True):
            with self.subTest(capture=capture):
                _, logits, strategy, hidden, pre_hc, _ = self._run(
                    None, capture=capture
                )
                args, kwargs = logits.call_args
                torch.testing.assert_close(args[1], torch.cat((hidden, hidden)))
                torch.testing.assert_close(
                    kwargs["hidden_states_before_norm"], torch.cat((pre_hc, pre_hc))
                )
                self.assertIsNone(args[4])
                self.assertEqual(strategy.gather_hidden_states.call_count, 2)

    def test_plain_tensor_body_with_aux_is_unchanged(self):
        aux = torch.ones(2, 4)
        _, logits, strategy, hidden, _, _ = self._run(aux, plain=True)
        args, kwargs = logits.call_args
        torch.testing.assert_close(args[1], torch.cat((hidden, hidden)))
        torch.testing.assert_close(args[4], torch.cat((aux, aux)))
        self.assertEqual(kwargs, {})
        self.assertEqual(strategy.gather_hidden_states.call_count, 2)

    def test_non_last_pp_rank_preserves_local_outputs(self):
        aux = [torch.ones(2, 4)]
        actual, logits, strategy, _, _, body = self._run(aux, last_rank=False)
        self.assertIs(actual[0], body)
        self.assertIs(actual[1], aux)
        logits.assert_not_called()
        strategy.gather_hidden_states.assert_not_called()


if __name__ == "__main__":
    unittest.main()
