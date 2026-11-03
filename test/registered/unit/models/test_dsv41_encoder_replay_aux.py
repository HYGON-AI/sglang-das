import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

register_cpu_ci(est_time=10, stage="base-a", runner_config="cpu")


class TestEncoderReplayAux(CustomTestCase):
    def _check(self, replay):
        hidden = torch.zeros(128, 4)
        pre_hc = torch.ones(128, 16)
        aux = torch.ones(128, 12)
        output = SimpleNamespace()
        processor = Mock(return_value=output)
        model = SimpleNamespace(
            capture_aux_hidden_states=True,
            model=SimpleNamespace(dspark_layers_to_capture=[2], late_layer_start=20),
            logits_processor=processor,
            lm_head=object(),
        )
        batch = SimpleNamespace(
            encoder_swa_replay=replay,
            pd_hidden_capture_layer_ids=[] if replay else [2],
            forward_mode=ForwardMode.EXTEND,
        )
        body = (hidden, pre_hc) if replay else ((hidden, pre_hc), [aux])
        with patch("sglang.srt.models.deepseek_v4.get_attn_backend",
                   return_value=SimpleNamespace(tail_forward_metadata=None)):
            result = DeepseekV4ForCausalLM.logits_from_body_output(
                model, torch.arange(128), body, batch
            )
        self.assertIs(result, output)
        self.assertIs(processor.call_args.args[1], hidden)
        if replay:
            self.assertIsNone(processor.call_args.args[4])
            self.assertIs(processor.call_args.kwargs["hidden_states_before_norm"], pre_hc)
            self.assertFalse(hasattr(output, "hidden_states"))
        else:
            self.assertIs(processor.call_args.args[4], aux)
            self.assertIs(output.hidden_states, aux)
        self.assertTrue(model.capture_aux_hidden_states)

    def test_replay_disables_aux_unpack_without_mutating_model(self):
        self._check(True)

    def test_normal_pd_still_returns_aux(self):
        self._check(False)


if __name__ == "__main__":
    unittest.main()
