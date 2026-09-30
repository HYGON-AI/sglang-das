import unittest

import torch
import triton

from sglang.kernels.ops.kvcache.kv_indices import (
    create_draft_extend_kv_metadata_triton,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA/HIP is required")
class TestDraftExtendKVMetadata(CustomTestCase):
    def _run_case(self, seq_lens_values, extend_lens_values, width):
        device = torch.device("cuda")
        bs = len(seq_lens_values)
        max_context = max(seq_lens_values) + 16
        req_to_token = torch.arange(
            (bs + 3) * max_context, dtype=torch.int64, device=device
        ).view(bs + 3, max_context)
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device=device) + 2
        seq_lens = torch.tensor(seq_lens_values, dtype=torch.int64, device=device)
        extend_lens = torch.tensor(extend_lens_values, dtype=torch.int32, device=device)
        kv_indptr = torch.full((bs + 1,), -1, dtype=torch.int32, device=device)
        qo_indptr = torch.full((bs + 1,), -1, dtype=torch.int32, device=device)
        kv_indices = torch.full(
            (bs * max_context,), -1, dtype=torch.int64, device=device
        )

        def launch():
            create_draft_extend_kv_metadata_triton[(bs,)](
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_lens,
                kv_indptr,
                qo_indptr,
                kv_indices,
                req_to_token.stride(0),
                NUM_TOKENS_PER_REQ=width,
                BS_BLOCK=triton.next_power_of_2(bs),
            )

        launch()
        torch.cuda.synchronize()

        kv_lens = (seq_lens - extend_lens).clamp_min(0).to(torch.int32)
        expected_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                kv_lens.cumsum(0, dtype=torch.int32),
            ]
        )
        expected_qo = torch.arange(
            0, (bs + 1) * width, width, dtype=torch.int32, device=device
        )
        expected_indices = torch.cat(
            [req_to_token[req_pool_indices[i], : int(kv_lens[i])] for i in range(bs)]
        )
        torch.testing.assert_close(kv_indptr, expected_indptr, rtol=0, atol=0)
        torch.testing.assert_close(qo_indptr, expected_qo, rtol=0, atol=0)
        torch.testing.assert_close(
            kv_indices[: expected_indices.numel()], expected_indices, rtol=0, atol=0
        )

    def test_ragged_prefixes(self):
        self._run_case([129], [3], 3)
        self._run_case([129, 80, 201, 17], [3, 2, 4, 20], 4)

    def test_hip_graph_replay_reads_updated_lengths(self):
        device = torch.device("cuda")
        bs, width, max_context = 2, 3, 256
        req_to_token = torch.arange(
            bs * max_context, dtype=torch.int64, device=device
        ).view(bs, max_context)
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device=device)
        seq_lens = torch.tensor([129, 80], dtype=torch.int64, device=device)
        extend_lens = torch.full((bs,), width, dtype=torch.int32, device=device)
        kv_indptr = torch.empty(bs + 1, dtype=torch.int32, device=device)
        qo_indptr = torch.empty(bs + 1, dtype=torch.int32, device=device)
        kv_indices = torch.empty(bs * max_context, dtype=torch.int64, device=device)

        def launch():
            create_draft_extend_kv_metadata_triton[(bs,)](
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_lens,
                kv_indptr,
                qo_indptr,
                kv_indices,
                req_to_token.stride(0),
                NUM_TOKENS_PER_REQ=width,
                BS_BLOCK=2,
            )

        for _ in range(2):
            launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()

        seq_lens.copy_(torch.tensor([130, 81], dtype=torch.int64, device=device))
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.tensor([0, 127, 205], dtype=torch.int32, device=device)
        torch.testing.assert_close(kv_indptr, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
