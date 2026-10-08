"""Failed loads must drain all writers before request slots can be reused."""

from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import Mock
from threading import Event

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import LayerWiseLoadCounter


def test_failed_layer_does_not_skip_waiting_for_remaining_writers():
    counter = LayerWiseLoadCounter(2)
    index = counter.update_producer()
    counter.futures[index][0].set_exception(RuntimeError("first layer failed"))
    last_writer = counter.futures[index][1]
    entered = Event()
    exception = last_writer.exception
    last_writer.exception = lambda: (entered.set(), exception())[1]
    with ThreadPoolExecutor(1) as executor:
        finish = executor.submit(counter.finish, index)
        try:
            assert entered.wait(timeout=5)
            # Neither failure nor early return permits dropping the last writer.
            assert not finish.done()
            assert index in counter.futures
        finally:
            last_writer.set_result(None)
        assert finish.result(timeout=5) is False
    assert index not in counter.futures


def test_finish_keeps_worker_completion_registered_until_worker_done():
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    completion = Future()
    entered = Event()
    result = completion.result
    completion.result = lambda: (entered.set(), result())[1]
    linker.batch_load_done = {3: completion}
    linker.layer_done_counter = Mock()
    linker.layer_done_counter.finish.return_value = True
    with ThreadPoolExecutor(1) as executor:
        finish = executor.submit(linker.finish_layer_wise_loading, 3)
        try:
            assert not finish.done()
            assert entered.wait(timeout=5)
            assert linker.batch_load_done[3] is completion
        finally:
            completion.set_result(True)
        assert finish.result(timeout=5) is True
    assert 3 not in linker.batch_load_done
    linker.layer_done_counter.finish.assert_called_once_with(3)
