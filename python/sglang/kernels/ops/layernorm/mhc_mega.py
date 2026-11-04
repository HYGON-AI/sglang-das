"""SGLang shifted-pre BF16 integration of Mega mHC.

Credit: DeepSeek's DeepGEMM project provides the fused Mega mHC kernel.
https://github.com/deepseek-ai/DeepGEMM
"""

import logging
from functools import lru_cache

import torch

from sglang.srt.utils import is_cuda

if is_cuda():
    import deep_gemm


@lru_cache(maxsize=1)
def hcu_mega_mhc_backend():
    try:
        import deepgemm

        if callable(getattr(deepgemm, "mega_mhc", None)) and callable(
            getattr(deepgemm, "mega_mhc_prepare_fn", None)
        ):
            return deepgemm
        reason = "missing mega_mhc or mega_mhc_prepare_fn"
    except (ImportError, OSError) as exc:
        reason = str(exc)
    logging.getLogger(__name__).warning("HCU Mega mHC disabled: %s", reason)
    return None


def mhc_mega_boundary(
    x,
    residual,
    pre,
    post,
    comb,
    fn,
    scale,
    base,
    weight,
    rms_eps,
    hc_eps,
    norm_eps,
    sinkhorn_iters,
    *,
    fn_packed=None,
):
    """Update the residual, collapse with carried pre, and compute next stats.

    The collapse uses ``pre`` from the preceding sublayer, not the new pre
    produced from the updated residual. Preserve this shifted ordering.
    HCU supplies a prepacked FFN weight. SM10x uses the CUDA backend
    and warms its per-stream barriers before graph capture.
    """
    rows, hidden = x.shape
    hc = residual.shape[1]
    updated = torch.empty_like(residual)
    normalized = torch.empty_like(x)
    next_pre = torch.empty((rows, hc, 1), device=x.device, dtype=torch.float32)
    next_post = torch.empty_like(next_pre)
    next_comb = torch.empty_like(comb)
    backend = deep_gemm if fn_packed is None else hcu_mega_mhc_backend()
    backend.mega_mhc(
        x=x,
        residual=residual,
        shifted_prev_mix=pre.unsqueeze(-1),
        post_mix=post.unsqueeze(-1),
        comb_res_mix=comb,
        fn=fn,
        mix_scales=scale,
        mix_bases=base,
        hc_mult=hc,
        hc_norm_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_post_scale=2.0,
        sinkhorn_eps=hc_eps,
        num_sinkhorn_iters=sinkhorn_iters,
        rmsnorm_weight=weight,
        rmsnorm_eps=norm_eps,
        rmsnorm_scale=1.0,
        new_residual=updated,
        new_prev_mix=next_pre,
        new_post_mix=next_post,
        new_comb_res_mix=next_comb,
        y_bf16=normalized,
        **({"fn_packed": fn_packed} if fn_packed is not None else {}),
    )
    return updated, normalized, (next_pre.squeeze(-1), next_post.squeeze(-1), next_comb)
