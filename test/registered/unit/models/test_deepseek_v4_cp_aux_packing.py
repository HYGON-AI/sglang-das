"""CPU regression tests for rank-local DSV4 DSPARK auxiliary packing."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.layernorm import mhc
from sglang.srt.models import deepseek_v4

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class _Layer(nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.engram = None
        self.increment = increment

    def forward(self, *, hidden_states, **kwargs):
        return hidden_states + self.increment, None, None, None

    def forward_hc_pre_from_prev(self, *, hidden_states, **kwargs):
        return hidden_states + self.increment, None


class TestDeepseekV4CpAuxPacking(CustomTestCase):
    def _forward(
        self,
        *,
        cp=True,
        hcu=True,
        pre_from_prev=True,
        captures=True,
        pp_size=1,
        rows=5,
        dtype=torch.bfloat16,
    ):
        model = object.__new__(deepseek_v4.DeepseekV4Model)
        nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(
            is_first_rank=True, is_last_rank=True, world_size=pp_size
        )
        model.hc_mult = 2
        model.hidden_size = 4
        model.start_layer = 0
        model.end_layer = 3
        model.layers = nn.ModuleList([_Layer(i + 1) for i in range(3)])
        model.dspark_layers_to_capture = [0, 1, 2] if captures else None
        model.hc_pre_from_prev_sublayer = pre_from_prev
        model.use_fused_mhc_post_pre = False
        model.engram_hasher = None
        model.late_layer_start = None
        model.config = SimpleNamespace(model_type="deepseek_v41")
        model._can_run_tbo = mock.Mock(return_value=False)
        model.norm = nn.Identity()
        model.hc_head = lambda x, *_: x.mean(dim=1)
        model.hc_head_fn = model.hc_head_scale = model.hc_head_base = None
        input_embeds = torch.arange(rows * 4, dtype=dtype).reshape(rows, 4)
        batch = SimpleNamespace(
            attn_cp_metadata=None,
            forward_mode=SimpleNamespace(
                is_extend=lambda: True,
                is_extend_without_speculative=lambda: True,
            ),
        )

        def combine(x, _, hc_mult, dtype):
            return x.reshape(rows, hc_mult, 4).mean(dim=1).to(dtype)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(deepseek_v4, "_is_hcu", hcu))
            stack.enter_context(mock.patch.object(deepseek_v4, "_is_npu", False))
            stack.enter_context(
                mock.patch.object(deepseek_v4, "is_cp_active", return_value=cp)
            )
            stack.enter_context(
                mock.patch.object(
                    deepseek_v4,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_dp_size=1),
                )
            )
            stack.enter_context(mock.patch.object(deepseek_v4, "get_attn_backend"))
            stack.enter_context(
                mock.patch.object(
                    deepseek_v4, "check_cuda_graph_backend", return_value=True
                )
            )
            stack.enter_context(
                mock.patch.object(mhc, "hc_combine", side_effect=combine)
            )
            result = model.forward(
                torch.arange(rows), torch.arange(rows), batch, input_embeds
            )
        return result, input_embeds

    def test_packed_captures_equal_original_for_both_hc_paths(self):
        for pre_from_prev in (True, False):
            for rows in (0, 5):
                for dtype in (torch.bfloat16, torch.float32):
                    with self.subTest(
                        pre_from_prev=pre_from_prev, rows=rows, dtype=dtype
                    ):
                        actual, input_embeds = self._forward(
                            pre_from_prev=pre_from_prev, rows=rows, dtype=dtype
                        )
                        original, _ = self._forward(
                            cp=False,
                            pre_from_prev=pre_from_prev,
                            rows=rows,
                            dtype=dtype,
                        )
                        body, aux = actual
                        self.assertIsInstance(aux, torch.Tensor)
                        self.assertEqual(aux.shape, (rows, 12))
                        self.assertTrue(aux.is_contiguous())
                        torch.testing.assert_close(
                            aux, torch.cat(original[1], dim=-1), rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            body[0], original[0][0], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            body[1], original[0][1], rtol=0, atol=0
                        )
                        offsets = (0, 1, 3) if pre_from_prev else (1, 3, 6)
                        for i, offset in enumerate(offsets):
                            torch.testing.assert_close(
                                aux[:, i * 4 : (i + 1) * 4],
                                input_embeds + offset,
                                rtol=0,
                                atol=0,
                            )

    def test_non_cp_non_hcu_and_multi_pp_keep_list(self):
        for kwargs in (
            dict(cp=False),
            dict(hcu=False),
            dict(pp_size=2, pre_from_prev=False),
        ):
            with self.subTest(**kwargs):
                (_, aux), _ = self._forward(**kwargs)
                self.assertIsInstance(aux, list)
                self.assertEqual(len(aux), 3)

    def test_without_capture_preserves_body_output(self):
        (hidden, pre_hc), input_embeds = self._forward(captures=False)
        torch.testing.assert_close(hidden, input_embeds + 6, rtol=0, atol=0)
        self.assertEqual(pre_hc.shape, (5, 8))


if __name__ == "__main__":
    unittest.main()
