"""CPU regression coverage for the local #43936/#47521 adaptation."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.common import dsv4_needs_rebootstrap, retraction_backup
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV4Retraction(unittest.TestCase):
    def _req(self, **overrides):
        req = Req.__new__(Req)
        req.multimodal_inputs = None
        req.input_embeds = None
        req.positional_embed_overrides = None
        req.send_token_offset = 0
        req.offload_kv_cache = MagicMock()
        for key, value in overrides.items():
            setattr(req, key, value)
        return req

    def _allocator(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.device = "cpu"
        return SimpleNamespace(get_kvcache=lambda: pool)

    def test_dsv4_recompute_does_not_copy_to_cpu(self):
        req, allocator = self._req(), self._allocator()
        self.assertTrue(dsv4_needs_rebootstrap(allocator))
        with patch("torch.get_device_module") as device_module:
            self.assertTrue(retraction_backup(req, None, None, allocator, "cpu_tensor"))
        device_module.return_value.synchronize.assert_called_once_with("cpu")
        req.offload_kv_cache.assert_not_called()

    def test_unrebuildable_requests_abort_before_copy(self):
        for field in ("multimodal_inputs", "input_embeds", "positional_embed_overrides"):
            with self.subTest(field=field):
                req = self._req(**{field: object()})
                with patch("torch.get_device_module") as device_module:
                    self.assertFalse(
                        retraction_backup(req, None, None, self._allocator(), "cpu_tensor")
                    )
                device_module.assert_not_called()
                req.offload_kv_cache.assert_not_called()

    def test_other_models_keep_cpu_backup(self):
        req = self._req()
        allocator = SimpleNamespace(get_kvcache=lambda: object())
        self.assertFalse(dsv4_needs_rebootstrap(allocator))
        self.assertTrue(retraction_backup(req, None, None, allocator, "cpu_tensor"))
        req.offload_kv_cache.assert_called_once_with(None, allocator)

    def test_boundary_replay_preserves_original_sampling_mask(self):
        scalar = lambda value: SimpleNamespace(item=lambda: value)
        metadata = [
            [scalar(99)],
            [scalar(value) for value in [2, 2, 0, 0, 0, 0, 0]],
            None, None, None, None, [scalar(-1)], [], None,
            None, None, None, None, [scalar(17)],
        ]
        req = self._req()
        req.rid = "replayed"
        req.bootstrap_host = "127.0.0.1"
        req.bootstrap_room = 17
        req.output_ids = [10, 11]
        req.pd_rebootstrap_forced_output_id = 12
        req.return_logprob = True
        req.return_sampling_mask = True
        req.logprob = SimpleNamespace(output_token_logprobs_val=[0.1, 0.2, 0.3])
        req.output_token_sampling_mask = [[10], [11], [12]]
        req.output_token_sampling_logprobs = [0.1, 0.2, 0.3]
        req.time_stats = MagicMock()
        receiver = MagicMock()
        transfer = SimpleNamespace(
            req=req, metadata_buffer_index=0, is_rebootstrap=True,
            kv_receiver=receiver,
        )
        queue = SimpleNamespace(
            metadata_buffers=SimpleNamespace(get_buf=lambda idx: metadata),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            _commit_hicache_local_restore_to_req=MagicMock(),
        )
        with patch("sglang.srt.disaggregation.decode._is_fake_transfer", return_value=False):
            DecodeTransferQueue._commit_transfer_to_req(queue, transfer)
        self.assertEqual(req.output_ids, [10, 11, 12])
        self.assertEqual(req.logprob.output_token_logprobs_val, [0.1, 0.2, 0.3])
        self.assertEqual(req.output_token_sampling_mask, [[10], [11], [12]])
        self.assertEqual(req.output_token_sampling_logprobs, [0.1, 0.2, 0.3])
        self.assertEqual(req.cached_tokens, 2)
        self.assertEqual(req.already_computed, 2)
        self.assertIsNone(transfer.kv_receiver)
        receiver.clear.assert_called_once()

    def test_input_embeds_reprefill_is_safe_only_before_output(self):
        for offset in (0, 1, 10):
            with self.subTest(offset=offset):
                req = self._req(input_embeds=object(), send_token_offset=offset)
                self.assertEqual(req.resumable_by_reprefill, offset == 0)
                self.assertFalse(req.resumable_by_rebootstrap)
        self.assertTrue(self._req(multimodal_inputs=object()).resumable_by_reprefill)
        self.assertTrue(self._req(positional_embed_overrides=object()).resumable_by_reprefill)


if __name__ == "__main__":
    unittest.main()
