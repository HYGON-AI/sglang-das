import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation import utils as disagg_utils  # noqa: E402
from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.mooncake.conn import (  # noqa: E402
    MooncakeKVReceiver,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecodePollAbortReduce(unittest.TestCase):
    @staticmethod
    def receiver(ready):
        return SimpleNamespace(
            is_drained_abort_ready=Mock(return_value=ready),
            require_staging=False,
        )

    @staticmethod
    def decode_req(receiver):
        return SimpleNamespace(kv_receiver=receiver)

    def reduce_with_peer(self, local_poll, receiver, peer_poll):
        with patch.object(
            disagg_utils,
            "_all_reduce_polls",
            side_effect=lambda polls, group: [min(polls[0], peer_poll)],
        ) as reduce:
            result = disagg_utils._all_reduce_decode_polls(
                [int(local_poll)], [self.decode_req(receiver)], group=None
            )
        return result[0], reduce.call_args.args[0]

    def test_only_drained_abort_ready_is_deferred_across_decode_ranks(self):
        result, encoded = self.reduce_with_peer(
            KVPoll.Failed, self.receiver(True), int(KVPoll.WaitingForInput)
        )
        self.assertEqual(encoded, [int(KVPoll.Success)])  # 4: abort-ready
        self.assertEqual(result, KVPoll.WaitingForInput)

        result, encoded = self.reduce_with_peer(
            KVPoll.Failed, self.receiver(True), int(KVPoll.Success) + 1
        )
        self.assertEqual(encoded, [int(KVPoll.Success)])
        self.assertEqual(result, KVPoll.Failed)

        result, encoded = self.reduce_with_peer(
            KVPoll.Failed, self.receiver(True), int(KVPoll.Success)
        )
        self.assertEqual(result, KVPoll.Failed)

    def test_success_and_legacy_or_partial_failure_keep_poll_order(self):
        result, encoded = self.reduce_with_peer(
            KVPoll.Success, self.receiver(False), int(KVPoll.Success) + 1
        )
        self.assertEqual(encoded, [int(KVPoll.Success) + 1])  # 5: success
        self.assertEqual(result, KVPoll.Success)

        result, encoded = self.reduce_with_peer(
            KVPoll.Success, self.receiver(False), int(KVPoll.WaitingForInput)
        )
        self.assertEqual(result, KVPoll.WaitingForInput)

        # Legacy failures remain KVPoll.Failed, so MIN still propagates them
        # immediately.
        result, encoded = self.reduce_with_peer(
            KVPoll.Failed, self.receiver(False), int(KVPoll.WaitingForInput)
        )
        self.assertEqual(encoded, [int(KVPoll.Failed)])
        self.assertEqual(result, KVPoll.Failed)

    def test_receiver_requires_the_complete_drained_abort_rank_set(self):
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.bootstrap_room = 17
        receiver.kv_mgr = SimpleNamespace(
            prefill_drained_abort_rooms={17},
            required_prefill_response_num_table={17: 2},
            prefill_response_tracker=defaultdict(set, {17: {0}}),
        )
        self.assertFalse(receiver.is_drained_abort_ready())

        receiver.kv_mgr.prefill_response_tracker[17].add(1)
        self.assertTrue(receiver.is_drained_abort_ready())

        receiver.kv_mgr.prefill_drained_abort_rooms.clear()
        self.assertFalse(receiver.is_drained_abort_ready())

        receiver.kv_mgr.prefill_drained_abort_rooms.add(17)
        receiver.kv_mgr.prefill_response_tracker[17].discard(1)
        self.assertFalse(receiver.is_drained_abort_ready())
        result, encoded = self.reduce_with_peer(
            KVPoll.Failed, receiver, int(KVPoll.WaitingForInput)
        )
        self.assertEqual(encoded, [int(KVPoll.Failed)])
        self.assertEqual(result, KVPoll.Failed)

    def test_decode_poll_entry_points_use_encoded_minimum(self):
        receiver = self.receiver(True)
        decode_req = self.decode_req(receiver)
        with (
            patch.object(
                disagg_utils,
                "_poll_with_failure_injection",
                return_value=[KVPoll.Failed],
            ),
            patch.object(
                disagg_utils,
                "_all_reduce_polls",
                side_effect=lambda polls, group: [
                    min(polls[0], KVPoll.WaitingForInput)
                ],
            ) as reduce,
        ):
            result = disagg_utils.poll_and_all_reduce(
                [object()], gloo_group=None, decode_reqs=[decode_req]
            )
        self.assertEqual(reduce.call_args.args[0], [int(KVPoll.Success)])
        self.assertEqual(result, [KVPoll.WaitingForInput])

        with (
            patch.object(
                disagg_utils,
                "_poll_with_failure_injection",
                return_value=[KVPoll.Success],
            ),
            patch.object(
                disagg_utils,
                "_all_reduce_polls",
                side_effect=lambda polls, group: list(polls),
            ) as reduce,
        ):
            result = disagg_utils.poll_and_all_reduce([object()], gloo_group=None)
        self.assertEqual(reduce.call_args.args[0], [KVPoll.Success])
        self.assertEqual(result, [KVPoll.Success])

        with (
            patch.object(
                disagg_utils,
                "_poll_with_failure_injection",
                return_value=[KVPoll.Failed],
            ),
            patch.object(
                disagg_utils,
                "_all_reduce_polls",
                side_effect=lambda polls, group: [
                    min(polls[0], KVPoll.WaitingForInput)
                ],
            ) as reduce,
        ):
            result = disagg_utils.poll_and_all_reduce_with_staging(
                [decode_req], staging_handler=object(), gloo_group=None
            )
        self.assertEqual(reduce.call_args.args[0], [int(KVPoll.Success)])
        self.assertEqual(result, [KVPoll.WaitingForInput])


if __name__ == "__main__":
    unittest.main()
