import unittest

import torch
from sgl_kernel import concat_mla_absorb_q

from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.test_utils import CustomTestCase

register_hcu_ci(est_time=30, suite="stage-b-test-1-hcu-small")


class TestConcatMlaAbsorbQHcu(CustomTestCase):
    def test_matches_torch_cat(self):
        for dim_0, dim_1 in ((1, 1), (1, 5), (4, 16), (9, 7)):
            with self.subTest(dim_0=dim_0, dim_1=dim_1):
                q_nope = torch.randn(
                    dim_0, dim_1, 512, device="cuda", dtype=torch.bfloat16
                )
                q_rope = torch.randn(
                    dim_0, dim_1, 64, device="cuda", dtype=torch.bfloat16
                )
                expected = torch.cat((q_nope, q_rope), dim=-1)
                actual = concat_mla_absorb_q(q_nope, q_rope)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_supports_aligned_non_contiguous_rows(self):
        q_nope_storage = torch.randn(3, 11, 520, device="cuda", dtype=torch.bfloat16)
        q_rope_storage = torch.randn(3, 11, 72, device="cuda", dtype=torch.bfloat16)
        q_nope = q_nope_storage[..., :512]
        q_rope = q_rope_storage[..., :64]
        self.assertFalse(q_nope.is_contiguous())
        self.assertFalse(q_rope.is_contiguous())

        actual = concat_mla_absorb_q(q_nope, q_rope)
        expected = torch.cat((q_nope, q_rope), dim=-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
