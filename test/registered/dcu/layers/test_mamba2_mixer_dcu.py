import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_dcu_ci

register_dcu_ci(est_time=60, suite="stage-b-test-1-gpu-small-dcu")


class TestMamba2MixerDCU(unittest.TestCase):
    def test_mixer2_gated_norm_single_gpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA device not available")

        device = torch.device("cuda:0")
        dtype = torch.float16
        torch.manual_seed(0)

        batch_size = 8
        seq_len = 128
        hidden_size = 64
        n_groups = 1

        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size, dtype=dtype, device=device
        )
        gate_states = torch.randn(
            batch_size, seq_len, hidden_size, dtype=dtype, device=device
        )
        weight = torch.rand(hidden_size, dtype=dtype, device=device)

        import sglang.srt.layers.attention.mamba.mixer2_rms_norm_gated as m2

        with (
            patch.object(m2, "get_tensor_model_parallel_world_size", return_value=1),
            patch.object(m2, "get_tensor_model_parallel_rank", return_value=0),
        ):
            mixer = m2.Mixer2RMSNormGated(
                full_hidden_size=hidden_size,
                full_n_groups=n_groups,
            ).to(device=device, dtype=dtype)
        mixer.weight.data.copy_(weight)

        output = mixer(hidden_states, gate_states)

        gated = hidden_states * torch.nn.functional.silu(gate_states.to(torch.float32))
        variance = gated.pow(2).mean(dim=-1, keepdim=True)
        reference = weight * (gated * torch.rsqrt(variance + mixer.variance_epsilon)).to(
            dtype
        )

        torch.testing.assert_close(output, reference, atol=5e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
