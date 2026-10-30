"""Unit tests for the encode-disaggregation receiver."""

import asyncio
import gc
import threading
import time
import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.encoder.receiver import (
    EmbeddingPool,
    MMReceiverBase,
    WaitingMMRequestStatus,
    WaitingRDMARequest,
    WaitingZmqRequest,
    WaitingZmqRequestGrpc,
    _ReceiveRegistrationRunner,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import EncoderDispatchErrorReq
from sglang.srt.managers.mm_utils import release_consumed_pool_slots
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_registration_request(request_cls):
    request = request_cls.__new__(request_cls)
    request.rid = "registration-test"
    request.registration_runner = _ReceiveRegistrationRunner(
        "test-encoder-receive-registration"
    )
    request.registration_future = None
    request.registration_error = None
    request.registration_lock = threading.Lock()
    request.status = WaitingMMRequestStatus.PENDING
    request.error_msg = None
    request.error_code = None
    request.embedding_pool = None
    request.embeddings_buffer = None
    request.recv_embedding_data = None
    request._pool_slot_id = None
    request._mm_finalizer = None
    request.recv_socket = None
    request.recv_req = SimpleNamespace(rid=request.rid)
    request.num_items_assigned = {Modality.IMAGE: [1]}
    request.encoder_urls = ["http://encoder"]
    request.host_name = "127.0.0.1"
    request.receive_count = 1
    request.embedding_port = 12345
    return request


def _cancel_registration(request):
    future = request.registration_future
    request.release_resources()
    deadline = time.monotonic() + 1
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert future.cancelled()


class BlockingResponse:
    def __init__(self, started):
        self.started = started

    async def __aenter__(self):
        self.started.set()
        await asyncio.Event().wait()

    async def __aexit__(self, *args):
        return False


class BlockingSession:
    started = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return BlockingResponse(self.started)


class FailingResponse:
    async def __aenter__(self):
        raise ConnectionError("encoder unavailable")

    async def __aexit__(self, *args):
        return False


class FailingSession(BlockingSession):
    def post(self, *args, **kwargs):
        return FailingResponse()


class TestReceiveRegistration(CustomTestCase):
    def test_http_registration_does_not_block_scheduler(self):
        started = threading.Event()
        BlockingSession.started = started
        request = _make_registration_request(WaitingZmqRequest)
        with patch(
            "sglang.srt.disaggregation.encoder.receiver.aiohttp.ClientSession",
            BlockingSession,
        ):
            scheduler_call = threading.Thread(
                target=request.send_encode_request, daemon=True
            )
            scheduler_call.start()
            self.assertTrue(started.wait(timeout=1))
            scheduler_call.join(timeout=0.1)

        self.assertFalse(scheduler_call.is_alive())
        self.assertEqual(request.status, WaitingMMRequestStatus.PENDING)
        _cancel_registration(request)

    def test_grpc_registration_does_not_block_scheduler(self):
        started = threading.Event()

        async def blocking_registration(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        request = _make_registration_request(WaitingZmqRequestGrpc)
        with patch(
            "sglang.srt.disaggregation.encoder.receiver._grpc_scheduler_receive_url",
            blocking_registration,
        ):
            scheduler_call = threading.Thread(
                target=request.send_encode_request, daemon=True
            )
            scheduler_call.start()
            self.assertTrue(started.wait(timeout=1))
            scheduler_call.join(timeout=0.1)

        self.assertFalse(scheduler_call.is_alive())
        self.assertEqual(request.status, WaitingMMRequestStatus.PENDING)
        _cancel_registration(request)

    def test_failure_is_request_local(self):
        request = _make_registration_request(WaitingZmqRequest)
        with patch(
            "sglang.srt.disaggregation.encoder.receiver.aiohttp.ClientSession",
            FailingSession,
        ):
            request.send_encode_request()
            deadline = time.monotonic() + 1
            while request.status == WaitingMMRequestStatus.PENDING:
                self.assertLess(time.monotonic(), deadline)
                request._try_recv_mm_data()
                time.sleep(0.01)

        self.assertEqual(request.status, WaitingMMRequestStatus.FAIL)
        self.assertEqual(request.error_code, HTTPStatus.BAD_GATEWAY)
        self.assertIn("encoder unavailable", request.error_msg)


class TestEncodeReceiverRequestConstruction(CustomTestCase):
    def test_early_dispatch_error_waits_for_scheduler_request(self):
        encode_finished = threading.Event()
        scheduler_dispatch_ready = threading.Event()
        reported = []
        failure = EncoderDispatchErrorReq(
            rid="request-1",
            error_msg="encoder unavailable",
            error_code=HTTPStatus.BAD_GATEWAY,
        )

        async def fail_encode(**kwargs):
            encode_finished.set()
            return failure

        receiver = SimpleNamespace(encode=fail_encode)
        worker = threading.Thread(
            target=MMReceiverBase._run_encode_in_thread,
            args=(
                receiver,
                failure.rid,
                [],
                "encode",
                {},
                [],
                None,
                scheduler_dispatch_ready,
                reported.append,
            ),
        )
        worker.start()

        self.assertTrue(encode_finished.wait(timeout=1))
        worker.join(timeout=0.05)
        self.assertTrue(worker.is_alive())
        self.assertEqual(reported, [])

        scheduler_dispatch_ready.set()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(reported, [failure])

    def test_dispatch_error_fails_only_owning_wait(self):
        class WaitingRequest:
            def __init__(self, rid):
                self.rid = rid
                self.recv_req = SimpleNamespace(rid=rid)
                self.status = WaitingMMRequestStatus.PENDING
                self.error_msg = None
                self.error_code = None
                self.start_time = 0

            def _try_recv_mm_data(self):
                pass

            def _fail_and_release(self, error_msg, error_code=None):
                self.error_msg = error_msg
                self.error_code = error_code
                self.status = WaitingMMRequestStatus.FAIL

            def release_resources(self):
                pass

            def close_recv_socket(self):
                pass

        owner = WaitingRequest("request-1")
        other = WaitingRequest("request-2")
        receiver = SimpleNamespace(
            waiting_list=[owner, other],
            waiting_by_rid={owner.rid: owner, other.rid: other},
            scheduler_recv_socket=None,
            wait_timeout=float("inf"),
            tp_group=SimpleNamespace(cpu_group=object()),
            _drain_scheduler_embeddings=lambda: None,
            _sync_fail_info_across_tp=lambda request: None,
            create_req=lambda request: request,
        )
        dispatch_error = EncoderDispatchErrorReq(
            rid=owner.rid,
            error_msg="bad media",
            error_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

        with patch("torch.distributed.all_reduce"):
            _, abort_reqs = MMReceiverBase._process_waiting_requests(
                receiver, [dispatch_error], waiting_cls=None
            )

        self.assertEqual(owner.status, WaitingMMRequestStatus.FAIL)
        self.assertEqual(owner.error_msg, dispatch_error.error_msg)
        self.assertEqual(owner.error_code, dispatch_error.error_code)
        self.assertEqual(other.status, WaitingMMRequestStatus.PENDING)
        self.assertEqual([req.rid for req, _, _ in abort_reqs], [owner.rid])

    def test_extra_key_and_cache_salt_are_forwarded(self):
        scheduler = SimpleNamespace(
            model_config=SimpleNamespace(hf_eos_token_id={2}, vocab_size=128),
            disaggregation_mode=DisaggregationMode.NULL,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            metrics_collector=None,
            dllm_config=None,
            tokenizer=object(),
        )
        receiver = SimpleNamespace(scheduler=scheduler)
        recv_req = SimpleNamespace(
            rid="request-1",
            input_text="hello",
            input_ids=array("q", [1, 2]),
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            lora_id=None,
            input_embeds=None,
            custom_logit_processor=None,
            require_reasoning=False,
            return_hidden_states=False,
            return_routed_experts=False,
            routed_experts_start_len=0,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
            routed_dp_rank=None,
            disagg_prefill_dp_rank=None,
            priority=None,
            extra_key="classification",
            cache_salt="tenant-a",
            http_worker_ipc=None,
        )

        req = MMReceiverBase.create_req(receiver, recv_req)

        self.assertEqual(req.extra_key, "classification")
        self.assertEqual(req.cache_salt, "tenant-a")

    def test_rdma_worker_error_is_released_on_scheduler_thread(self):
        scheduler_thread = threading.get_ident()

        class ThreadCheckedSocket:
            closed_by = None

            def close(self):
                self.closed_by = threading.get_ident()

        recv_socket = ThreadCheckedSocket()
        request = WaitingRDMARequest.__new__(WaitingRDMARequest)
        request.rid = "request-1"
        request.status = WaitingMMRequestStatus.PENDING
        request.error_msg = None
        request.error_code = None
        request.recv_socket = recv_socket
        request._receive_error = None
        request._receive_error_lock = threading.Lock()
        request._buffer_lock = threading.Lock()
        request._terminal = False
        request._receive_running = False
        request.registration_future = None
        request.embeddings_buffer = None
        request._pool_slot_id = None
        request.embedding_pool = None
        request._mm_finalizer = None

        worker = threading.Thread(
            target=lambda: asyncio.run(
                request._check_encoder_responses(
                    [ConnectionError("encoder unavailable")], "/send"
                )
            )
        )
        worker.start()
        worker.join(timeout=1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(request.status, WaitingMMRequestStatus.PENDING)
        self.assertIsNone(request.recv_socket.closed_by)

        request._try_recv_mm_data()

        self.assertEqual(request.status, WaitingMMRequestStatus.FAIL)
        self.assertIsNone(request.recv_socket)
        self.assertTrue(request._terminal)
        self.assertEqual(recv_socket.closed_by, scheduler_thread)

    def test_tp_peer_failure_closes_local_receive_socket(self):
        class WaitingRequest:
            rid = "request-1"
            recv_req = SimpleNamespace(rid=rid)
            status = WaitingMMRequestStatus.PENDING
            error_msg = "peer failed"
            error_code = None
            start_time = 0
            released = False
            closed = False

            def _try_recv_mm_data(self):
                pass

            def release_resources(self):
                self.released = True

            def close_recv_socket(self):
                self.closed = True

        waiting_req = WaitingRequest()
        receiver = SimpleNamespace(
            waiting_list=[waiting_req],
            waiting_by_rid={waiting_req.rid: waiting_req},
            scheduler_recv_socket=None,
            wait_timeout=float("inf"),
            tp_group=SimpleNamespace(cpu_group=object()),
            _drain_scheduler_embeddings=lambda: None,
            _sync_fail_info_across_tp=lambda request: None,
            create_req=lambda request: request,
        )

        def force_peer_failure(status, **kwargs):
            status.fill_(WaitingMMRequestStatus.FAIL)

        with patch("torch.distributed.all_reduce", force_peer_failure):
            _, abort_reqs = MMReceiverBase._process_waiting_requests(
                receiver, [], waiting_cls=None
            )

        self.assertTrue(waiting_req.released)
        self.assertTrue(waiting_req.closed)
        self.assertEqual(len(abort_reqs), 1)


_real_empty = torch.empty


def _make_pool(size_bytes):
    # The pool hardcodes a CUDA device; back it with host memory for CPU CI.
    with patch.object(
        torch, "empty", lambda n, dtype, device: _real_empty(n, dtype=dtype)
    ):
        return EmbeddingPool(gpu_id=0, size_bytes=size_bytes)


class _FakeEvent:
    def __init__(self, done_after_queries):
        self._remaining = done_after_queries

    def query(self):
        self._remaining -= 1
        return self._remaining < 0


class _RecordingLease:
    def __init__(self):
        self.released = 0

    def mark_used(self, event):
        pass

    def release(self, event=None):
        self.released += 1


def _image_item(offsets, embedding):
    return MultimodalDataItem(
        modality=Modality.IMAGE, offsets=offsets, precomputed_embeddings=embedding
    )


def _make_bound_request(pool, slot, view, lease_pool_slots):
    request = WaitingRDMARequest.__new__(WaitingRDMARequest)
    request.embedding_pool = pool
    request.lease_pool_slots = lease_pool_slots
    request._pool_slot_id = slot
    request.embeddings_buffer = view
    request._mm_finalizer = None
    return request


class TestPoolSlotLifetime(CustomTestCase):
    def test_slot_outlives_rebuilt_mm_container(self):
        """The scheduler rebuilds the receiver's mm container keeping only its items;
        dropping the original container must not free a slot the items still view,
        or the next request's RDMA overwrites embeddings before prefill reads them."""
        pool = _make_pool(size_bytes=1024)
        view, _, slot = pool.try_alloc(256)
        item = _image_item(offsets=[(0, 127)], embedding=view.view(torch.bfloat16))
        original = SimpleNamespace(mm_items=[item])

        request = _make_bound_request(pool, slot, view, lease_pool_slots=True)
        self.assertTrue(request._bind_pool_slot_to_mm_inputs(original))

        rebuilt = MultimodalInputs(mm_items=[item])
        del original, request
        gc.collect()
        self.assertIn(slot, pool._inflight)

        del rebuilt, item
        gc.collect()
        self.assertNotIn(slot, pool._inflight)

    def test_lease_binding_scoped_to_opted_in_models(self):
        """Models outside the lease allowlist must bind no item leases, so the
        consumed-slot release path never touches their embeddings."""
        pool = _make_pool(size_bytes=1024)
        view, _, slot = pool.try_alloc(256)
        item = _image_item(offsets=[(0, 127)], embedding=view.view(torch.bfloat16))
        original = SimpleNamespace(mm_items=[item])

        request = _make_bound_request(pool, slot, view, lease_pool_slots=False)
        self.assertTrue(request._bind_pool_slot_to_mm_inputs(original))
        self.assertEqual(item.pool_slot_leases, ())

    def test_release_with_pending_event_defers_reuse(self):
        """A slot released while the GPU reads of it are still queued must not be
        handed to the next RDMA write until the reads' event completes."""
        pool = _make_pool(size_bytes=256)
        _, _, slot = pool.try_alloc(256)
        pool.release(slot, ready_event=_FakeEvent(done_after_queries=2))
        self.assertIsNone(pool.try_alloc(256))
        self.assertIsNotNone(pool.try_alloc(256))

    def test_blocked_alloc_wakes_when_deferred_event_completes(self):
        """No release() call fires when a deferred event completes, so a blocked
        alloc() must poll for it instead of sleeping until its timeout."""
        pool = _make_pool(size_bytes=256)
        _, _, slot = pool.try_alloc(256)
        pool.release(slot, ready_event=_FakeEvent(done_after_queries=3))

        result = {}

        def alloc():
            start = time.monotonic()
            result["slot"] = pool.alloc(256, timeout=10.0)
            result["elapsed"] = time.monotonic() - start

        worker = threading.Thread(target=alloc)
        worker.start()
        worker.join(timeout=15.0)
        self.assertIsNotNone(result["slot"])
        self.assertLess(result["elapsed"], 5.0)


class TestReleaseConsumedPoolSlots(CustomTestCase):
    def test_slot_released_only_after_last_chunk(self):
        """An image spanning two prefill chunks must keep its slot after the first;
        offsets are inclusive, so it frees only once the end position is processed."""
        lease = _RecordingLease()
        item = _image_item(offsets=[(2, 9)], embedding=torch.zeros(8, 4))
        item.attach_pool_slot_leases([lease])
        mm_inputs = MultimodalInputs(mm_items=[item])

        release_consumed_pool_slots(
            mm_inputs_list=[mm_inputs], extend_prefix_lens=[0], extend_seq_lens=[6]
        )
        self.assertEqual(lease.released, 0)

        release_consumed_pool_slots(
            mm_inputs_list=[mm_inputs], extend_prefix_lens=[6], extend_seq_lens=[3]
        )
        self.assertEqual(lease.released, 0)

        release_consumed_pool_slots(
            mm_inputs_list=[mm_inputs], extend_prefix_lens=[6], extend_seq_lens=[4]
        )
        self.assertEqual(lease.released, 1)

    def test_shared_lease_waits_for_every_item(self):
        """Per-part leases are shared by all items of a request; one finished image
        must not free a slot that a later image in the same request still reads."""
        lease = _RecordingLease()
        first = _image_item(offsets=[(0, 3)], embedding=torch.zeros(4, 4))
        second = _image_item(offsets=[(10, 13)], embedding=torch.zeros(4, 4))
        for item in (first, second):
            item.attach_pool_slot_leases([lease])
        mm_inputs = MultimodalInputs(mm_items=[first, second])

        release_consumed_pool_slots(
            mm_inputs_list=[mm_inputs], extend_prefix_lens=[0], extend_seq_lens=[8]
        )
        self.assertEqual(lease.released, 0)

        release_consumed_pool_slots(
            mm_inputs_list=[mm_inputs], extend_prefix_lens=[8], extend_seq_lens=[8]
        )
        self.assertEqual(lease.released, 1)


if __name__ == "__main__":
    unittest.main()
