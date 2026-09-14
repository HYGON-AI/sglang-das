#!/usr/bin/env python3
"""TCP content and failure-cleanup checks for Mooncake ReadPlan.

Requires a running mooncake_master and built store Python module.

Example:
  PYTHONPATH=build/mooncake-integration \\
  MOONCAKE_PROTOCOL=tcp \\
  MOONCAKE_MASTER=127.0.0.1:50051 \\
  MOONCAKE_TE_META_DATA_SERVER=P2PHANDSHAKE \\
  python3 mooncake_read_plan_tcp_smoke.py
"""

from __future__ import annotations

import ctypes
import os
import sys
import time


def _require_store():
    try:
        from mooncake import store
    except Exception as exc:  # pragma: no cover
        print(f"import_fail {exc}", flush=True)
        raise SystemExit(10)
    return store


def _ptr(buf: ctypes.Array) -> int:
    return ctypes.addressof(buf)


def run() -> int:
    store = _require_store()

    master = os.getenv("MOONCAKE_MASTER", "127.0.0.1:50051")
    metadata = os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE")
    protocol = os.getenv("MOONCAKE_PROTOCOL", "tcp")
    device = os.getenv("MOONCAKE_DEVICE", "")
    hostname = os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost:17814")
    segment = int(os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(64 * 1024 * 1024)))
    local_buf = int(os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", str(64 * 1024 * 1024)))

    num_layers = int(os.getenv("E2E_NUM_LAYERS", "4"))
    page_size = int(os.getenv("E2E_PAGE_SIZE", "4096"))
    num_keys = int(os.getenv("E2E_NUM_KEYS", "3"))
    object_size = page_size * num_layers

    print(
        f"e2e_session_ranges protocol={protocol} master={master} "
        f"keys={num_keys} layers={num_layers} page={page_size}",
        flush=True,
    )

    mc = store.MooncakeDistributedStore()
    setup_ret = mc.setup(
        hostname, metadata, segment, local_buf, protocol, device, master
    )
    print(f"setup_ret {setup_ret}", flush=True)
    if setup_ret != 0:
        return setup_ret

    src = (ctypes.c_char * (object_size * num_keys))()
    dst = (ctypes.c_char * (object_size * num_keys))()
    for i in range(len(src)):
        src[i] = ord("a") + (i % 26)
        dst[i] = ord("B")

    assert mc.register_buffer(_ptr(src), len(src)) == 0
    assert mc.register_buffer(_ptr(dst), len(dst)) == 0

    keys = [f"session_e2e_key_{i}_{int(time.time())}" for i in range(num_keys)]
    sizes = [object_size] * num_keys

    put_start = mc.batch_put_session_start(keys, sizes)
    print(f"batch_put_session_start {put_start}", flush=True)
    if any(rc != 0 for rc in put_start):
        return 20

    for layer in range(num_layers):
        all_buffers = []
        all_sizes = []
        all_dst_offsets = []
        for i in range(num_keys):
            offset = i * object_size + layer * page_size
            all_buffers.append([_ptr(src) + offset])
            all_sizes.append([page_size])
            all_dst_offsets.append([layer * page_size])
        put_rcs = mc.batch_put_from_multi_buffer_ranges(
            keys, all_buffers, all_sizes, all_dst_offsets
        )
        print(f"batch_put_ranges layer={layer} rcs={put_rcs}", flush=True)
        if any(rc != page_size for rc in put_rcs):
            mc.batch_put_session_revoke(keys)
            return 21

    put_end = mc.batch_put_session_end(keys)
    print(f"batch_put_session_end {put_end}", flush=True)
    if any(rc != 0 for rc in put_end):
        return 22

    Plan = mc.create_read_plan
    reuse = os.environ.get("E2E_REUSE_RANGES", "0") == "1"
    layout = [
        [
            (
                _ptr(dst) + layer * page_size,
                object_size,
                page_size // 2,
                layer * page_size,
            ),
            (
                _ptr(dst) + layer * page_size + page_size // 2,
                object_size,
                page_size // 2,
                layer * page_size + page_size // 2,
            ),
        ]
        for layer in range(num_layers)
    ]
    plan = Plan(
        [(keys, list(range(num_keys)), True, layout)],
        num_layers,
        reuse_ranges=reuse,
    )
    plan.run()
    for layer in range(num_layers):
        plan.wait(layer)
    assert plan.stats() == [num_layers, num_layers * num_keys, object_size * num_keys]
    print("read_plan_stats", plan.stats(), flush=True)

    if bytes(src) != bytes(dst):
        print("data_mismatch", flush=True)
        return 40

    # Partial session-start failure must notify consumers and clean up.
    bad_plan = Plan(
        [(keys[:1] + ["missing_read_plan_key"], [0, 1], True, layout)],
        num_layers,
        reuse_ranges=reuse,
    )
    try:
        bad_plan.run()
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing key accepted")
    try:
        bad_plan.wait(num_layers - 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed plan wait unexpectedly succeeded")
    assert mc.batch_get_session_start(keys) == [0] * num_keys
    assert mc.batch_get_session_end(keys) == 0
    print("read_plan_failure_cleanup PASSED", flush=True)

    # Revoke path: start put then revoke before end.
    revoke_keys = [f"session_e2e_revoke_{int(time.time())}"]
    revoke_start = mc.batch_put_session_start(revoke_keys, [page_size])
    print(f"batch_put_session_start_revoke {revoke_start}", flush=True)
    if any(rc != 0 for rc in revoke_start):
        return 50
    revoke_rcs = mc.batch_put_session_revoke(revoke_keys)
    print(f"batch_put_session_revoke {revoke_rcs}", flush=True)
    if any(rc != 0 for rc in revoke_rcs):
        return 51

    print("session_ranges_tcp_e2e PASSED", flush=True)
    mc.close()
    return 0


if __name__ == "__main__":
    sys.exit(run())
