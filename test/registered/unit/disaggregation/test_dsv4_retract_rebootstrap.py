"""CPU regression coverage for the local #43936/#47521 adaptation."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
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

    def test_new_room_requires_the_supported_decode_topology(self):
        sizes = dict(
            tp_size=16, dp_size=16, attn_cp_size=1, pp_size=1,
            attn_dp_size=16, attn_tp_size=1,
        )
        supports = DecodePreallocQueue._supports_new_rebootstrap_room
        self.assertTrue(supports(SimpleNamespace(**sizes), self._allocator()))
        for field, value in (
            ("tp_size", 8), ("dp_size", 8), ("attn_cp_size", 2), ("pp_size", 2),
            ("attn_dp_size", 1), ("attn_tp_size", 2),
        ):
            with self.subTest(field=field):
                self.assertFalse(
                    supports(SimpleNamespace(**{**sizes, field: value}), self._allocator())
                )
        self.assertFalse(
            supports(SimpleNamespace(**sizes), SimpleNamespace(get_kvcache=object))
        )

    def test_room_is_allocated_before_receiver_only_for_opted_in_rebootstrap(self):
        for enabled, rebootstrap in ((False, True), (True, False), (True, True)):
            with self.subTest(enabled=enabled, rebootstrap=rebootstrap):
                queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
                queue.use_new_rebootstrap_room = enabled
                queue._check_if_req_exceed_kv_capacity = MagicMock(return_value=False)
                queue._resolve_prefill_dp_rank = MagicMock(return_value=3)
                receiver = MagicMock()

                def create(req, is_rebootstrap):
                    receiver.bootstrap_room = req.bootstrap_room
                    return SimpleNamespace(req=req, kv_receiver=receiver)

                queue._create_receiver_and_enqueue = create
                req = self._req(
                    rid="r", bootstrap_host="127.0.0.1", bootstrap_room=17,
                    pd_bound_prefill_dp_rank=3, disagg_prefill_dp_rank=None,
                    retraction_count=1,
                )
                with patch(
                    "sglang.srt.disaggregation.decode.secrets.randbits",
                    side_effect=[0, 17, 32],
                ) as randbits:
                    queue.add(req, is_rebootstrap=rebootstrap)
                self.assertEqual(receiver.bootstrap_room, 32 if enabled and rebootstrap else 17)
                self.assertEqual(req.disagg_prefill_dp_rank, 3 if enabled and rebootstrap else None)
                self.assertEqual(randbits.call_count, 3 if enabled and rebootstrap else 0)
                receiver.init.assert_called_once_with(3)

    def test_recompute_payload_uses_receiver_room_and_bound_prefill_rank(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.use_new_rebootstrap_room = True
        queue.kv_manager = SimpleNamespace(submit_prefill_recompute=MagicMock())
        req = self._req(
            bootstrap_host="127.0.0.1",
            pd_bound_prefill_dp_rank=3,
            build_rebootstrap_payload=lambda: {
                "bootstrap_room": 17,
                "disagg_prefill_dp_rank": 3,
            },
        )
        receiver = SimpleNamespace(bootstrap_room=32)
        queue._submit_prefill_recompute(SimpleNamespace(req=req, kv_receiver=receiver))
        queue.kv_manager.submit_prefill_recompute.assert_called_once_with(
            receiver,
            {"bootstrap_room": 32, "routed_dp_rank": 3, "disagg_prefill_dp_rank": 3},
        )
        queue.use_new_rebootstrap_room = False
        queue.kv_manager.submit_prefill_recompute.reset_mock()
        queue._submit_prefill_recompute(SimpleNamespace(req=req, kv_receiver=receiver))
        queue.kv_manager.submit_prefill_recompute.assert_called_once_with(
            receiver, {"bootstrap_room": 17, "disagg_prefill_dp_rank": 3}
        )


if __name__ == "__main__":
    unittest.main()
