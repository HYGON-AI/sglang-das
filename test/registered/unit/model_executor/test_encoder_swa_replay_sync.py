"""Encoder replay must use its own MLP-sync counts, not the parent prefill's."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor import forward_batch_info as fbi
from sglang.srt.model_executor.encoder_swa_replay import run_encoder_swa_replay
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEncoderSwaReplaySync(CustomTestCase):
    def _run(self, parent_tokens, *, prefixes=(512,), cp_size=8, sync=True):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            reqs=[
                SimpleNamespace(
                    rid=str(i), full_untruncated_fill_ids=list(range(n + 512))
                )
                for i, n in enumerate(prefixes)
            ],
            encoder_swa_reset=[True] * len(prefixes),
            prefix_lens=list(prefixes),
            input_ids=torch.arange(parent_tokens),
            req_pool_indices=torch.arange(len(prefixes)),
            req_pool_indices_cpu=torch.arange(len(prefixes)),
            global_num_tokens=[parent_tokens] if sync else None,
            global_num_tokens_for_logprob=[len(prefixes)] if sync else None,
            dp_spec_prefill_coordination_applied=False,
            can_run_decode_cuda_graph=False,
            can_run_dp_draft_cuda_graph=False,
        )
        window = SimpleNamespace(reset=MagicMock())
        runner = SimpleNamespace(
            device="cpu",
            token_to_kv_pool=SimpleNamespace(request_window=window),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(4096).repeat(len(prefixes), 1)
            ),
            model=SimpleNamespace(model=SimpleNamespace(engram_hasher=None)),
            attn_tp_sequence_sharded=lambda _: False,
            attn_backend=SimpleNamespace(get_cpu_graph_seq_len_fill_value=lambda: 1),
            is_draft_worker=False,
        )
        built = []

        def init_new(replay, model_runner, **kwargs):
            # Keep model-specific position/vision setup out of this CPU test;
            # exercise the real sync-metadata conversion and padding below.
            fb = ForwardBatch(
                forward_mode=replay.forward_mode,
                batch_size=len(replay.reqs),
                input_ids=replay.input_ids,
                req_pool_indices=replay.req_pool_indices,
                seq_lens=replay.seq_lens,
                seq_lens_sum=replay.seq_lens_sum,
                out_cache_loc=replay.out_cache_loc,
                positions=torch.arange(
                    replay.prefix_lens[0], replay.seq_lens_sum
                ),
                seq_lens_cpu=replay.seq_lens_cpu,
                is_extend_in_batch=True,
            )
            fb.init_mlp_sync_metadata(replay, torch.device("cpu"))
            built.append((replay, fb))
            return fb

        def forward(fb):
            if fb.global_num_tokens_cpu is not None:
                fb.prepare_mlp_sync_batch(runner)

        runner.forward = forward
        parallel = SimpleNamespace(
            attn_cp_size=cp_size,
            attn_cp_rank=0,
            attn_tp_size=1,
            attn_cp_group=SimpleNamespace(cpu_group=None),
        )

        def gather(plans, plan, **kwargs):
            plans[:] = [plan] * cp_size

        with ExitStack() as stack:
            for target, value in [
                ("sglang.srt.runtime_context.get_parallel", parallel),
                (
                    "sglang.srt.model_executor.forward_batch_info.get_parallel",
                    parallel,
                ),
                (
                    "sglang.srt.model_executor.forward_batch_info.get_exec",
                    SimpleNamespace(
                        graph=SimpleNamespace(
                            cuda_graph_config=SimpleNamespace(
                                prefill=SimpleNamespace(bs=[])
                            )
                        )
                    ),
                ),
            ]:
                stack.enter_context(patch(target, return_value=value))
            stack.enter_context(
                patch.object(ForwardBatch, "init_new", side_effect=init_new)
            )
            stack.enter_context(
                patch("torch.distributed.all_gather_object", side_effect=gather)
            )
            stack.enter_context(patch.object(fbi, "_is_cpu", True))
            stack.enter_context(patch.object(fbi, "dp_slot_in", return_value=0))
            stack.enter_context(patch.object(fbi, "set_dp_buffer_len_from_batch"))
            stack.enter_context(patch.object(fbi, "set_is_extend_in_batch"))
            stack.enter_context(
                patch.object(
                    fbi, "_elastic_should_preserve_local_token_counts", return_value=False
                )
            )
            stack.enter_context(
                patch.object(
                    DpPaddingMode,
                    "get_dp_padding_mode",
                    return_value=DpPaddingMode.SUM_LEN,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
                    return_value=cp_size,
                )
            )
            stack.enter_context(
                patch("sglang.srt.layers.cp.utils.enable_cp_v2", return_value=False)
            )
            stack.enter_context(
                patch(
                    "sglang.srt.batch_overlap.two_batch_overlap."
                    "TboForwardBatchPreparer.prepare"
                )
            )
            run_encoder_swa_replay(SimpleNamespace(model_runner=runner), batch)

        self.assertEqual(len(built), sum(n > 0 for n in prefixes))
        for replay, fb in built:
            count = min(128, replay.seq_lens_sum)
            self.assertEqual(fb.input_ids.numel(), count)
            torch.testing.assert_close(
                fb.input_ids,
                torch.arange(replay.seq_lens_sum - count, replay.seq_lens_sum),
            )
            if sync:
                self.assertEqual(fb.global_num_tokens_cpu, [count])
                self.assertEqual(fb.global_num_tokens_for_logprob_cpu, [1])
                self.assertEqual(fb.global_dp_buffer_len, count)
        self.assertEqual(batch.global_num_tokens, [parent_tokens] if sync else None)
        self.assertEqual(
            batch.global_num_tokens_for_logprob, [len(prefixes)] if sync else None
        )
        self.assertEqual(batch.input_ids.numel(), parent_tokens)

    def test_short_parent_reproduces_gsm8k_negative_40_padding(self):
        self._run(88)

    def test_replay_does_not_pad_to_a_larger_parent(self):
        self._run(4096)

    def test_multiple_requests_each_have_one_replay_logits_row(self):
        self._run(329, prefixes=(512, 256, 64))

    def test_equal_length_parent(self):
        self._run(128)

    def test_non_mlp_sync_path(self):
        self._run(88, sync=False)

    def test_single_rank_and_uncached_prefix(self):
        self._run(88, prefixes=(0, 64, 512), cp_size=1)


if __name__ == "__main__":
    unittest.main()
