import pytest
import torch

from sglang.srt.mem_cache.memory_pool import _scaled_fp8_set_kv_buffer
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_scaled_fp8_kv_store_saturates_uint8_backing():
    """Finite E4M3 overflows must saturate instead of becoming 0x7f/0xff NaN."""
    fp8_dtype = torch.float8_e4m3fn
    source_k = torch.tensor([[[496.0]]], device="cuda", dtype=torch.bfloat16)
    source_v = torch.tensor([[[-496.0]]], device="cuda", dtype=torch.bfloat16)
    k_backing = torch.zeros((4, 1, 1), device="cuda", dtype=torch.uint8)
    v_backing = torch.zeros((4, 1, 1), device="cuda", dtype=torch.uint8)
    locations = torch.tensor([1], device="cuda", dtype=torch.int64)

    _scaled_fp8_set_kv_buffer(
        source_k,
        source_v,
        k_backing.view(fp8_dtype),
        v_backing.view(fp8_dtype),
        locations,
        k_scale=0.107421875,
        v_scale=0.107421875,
        row_dim=1,
        v_row_dim=1,
        size_limit=4,
    )
    torch.cuda.synchronize()

    assert int(k_backing[1, 0, 0]) == 0x7E
    assert int(v_backing[1, 0, 0]) == 0xFE
    stored = torch.stack(
        (
            k_backing.view(fp8_dtype)[1, 0, 0].float(),
            v_backing.view(fp8_dtype)[1, 0, 0].float(),
        )
    )
    torch.testing.assert_close(
        stored, torch.tensor([448.0, -448.0], device="cuda"), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_scaled_fp8_kv_store_device_scale_is_graph_safe():
    fp8_dtype = torch.float8_e4m3fn
    source = torch.tensor([[[-496.0]]], device="cuda", dtype=torch.bfloat16)
    k_backing = torch.zeros((4, 1, 1), device="cuda", dtype=torch.uint8)
    v_backing = torch.zeros_like(k_backing)
    locations = torch.tensor([1], device="cuda", dtype=torch.int64)
    scale = torch.tensor(0.107421875, device="cuda", dtype=torch.float32)

    def write():
        _scaled_fp8_set_kv_buffer(
            source,
            source,
            k_backing.view(fp8_dtype),
            v_backing.view(fp8_dtype),
            locations,
            k_scale=scale,
            v_scale=scale,
            row_dim=1,
            v_row_dim=1,
            size_limit=4,
        )

    # Compile before capture, then verify both capture and replay.
    write()
    torch.cuda.synchronize()
    k_backing.zero_()
    v_backing.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write()
    graph.replay()
    torch.cuda.synchronize()

    assert int(k_backing[1, 0, 0]) == 0xFE
    assert int(v_backing[1, 0, 0]) == 0xFE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
