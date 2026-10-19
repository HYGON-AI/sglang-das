"""V4 cache storage on HCU must retain its HIP JIT path after forward-port."""

from unittest.mock import Mock, patch

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import attn
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout


@pytest.mark.parametrize("layout", list(KVLayout))
def test_hcu_physical_layout_gate(layout):
    module = Mock()
    x = torch.zeros(1, 512, dtype=torch.bfloat16)
    indices = torch.ones(1, dtype=torch.int64)
    with (
        patch.object(attn, "_is_hcu", True),
        patch.object(attn, "is_hip_runtime", return_value=True),
        patch.object(attn, "is_gfx95_supported", return_value=False),
        patch.object(attn, "_jit_fused_store_module", return_value=module),
    ):
        if layout is KVLayout.V4:
            attn.fused_store_cache(x, x, indices, page_size=64, type="flashmla", layout=layout)
            module.run.assert_called_once_with(x, x, indices)
        else:
            with pytest.raises(AssertionError, match="require gfx950"):
                attn.fused_store_cache(x, x, indices, page_size=64, type="flashmla", layout=layout)
            module.run.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("page_size", [64, 256])
def test_hcu_v4_store_bytes(page_size):
    from sglang.kernels.jit.utils.arch import get_default_target_flags
    from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

    torch.manual_seed(17)
    dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
    # Match the JIT's explicit scale bound (type.cuh), independently of the
    # physical payload type. On HCU gfx938 the existing source uses the
    # conservative 224 bound while native HIP packs E4M3FN bytes.
    scale_max = 224.0 if "-DHIP_FP8_TYPE_FNUZ=1" in get_default_target_flags() else 448.0
    x = torch.randn(9, 512, dtype=torch.bfloat16, device="cuda")
    x[0] = 0
    indices = torch.tensor([1, 3, 7, 15, 31, 63, page_size, page_size + 2, page_size * 2 - 1], device="cuda")
    cache = torch.zeros(3, KVLayout.V4.page_bytes(page_size), dtype=torch.uint8, device="cuda")
    expected = torch.zeros_like(cache)
    blocks = x[:, :448].float().reshape(9, 7, 64)
    scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / scale_max)))
    payload = (blocks / scale.unsqueeze(-1)).to(dtype).view(torch.uint8).reshape(9, 448)
    exponents = (torch.log2(scale).to(torch.int32) + 127).to(torch.uint8)
    for i, loc in enumerate(indices.tolist()):
        page, slot = divmod(loc, page_size)
        expected[page, slot * 576:slot * 576 + 448] = payload[i]
        expected[page, slot * 576 + 448:(slot + 1) * 576] = x[i, 448:].view(torch.uint8)
        expected[page, page_size * 576 + slot * 8:page_size * 576 + slot * 8 + 7] = exponents[i]
    attn.fused_store_cache(x, cache, indices, page_size=page_size, type="flashmla")
    torch.cuda.synchronize()
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
