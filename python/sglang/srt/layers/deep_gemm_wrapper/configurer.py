import logging

from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_device_sm,
    is_cuda,
    is_hip,
    is_musa,
    is_sm100_supported,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_musa = is_musa()


def _compute_enable_deep_gemm():
    sm_version = get_device_sm()
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False
    # DeepGEMM requires TMEM/tcgen05 (SM100+datacenter), not available on SM120
    if sm_version == 120:
        return False
    if not (_is_cuda or _is_musa):
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()


def _compute_enable_hcu_deepgemm():
    """Detect the DTK/HCU DeepGEMM runtime (``deepgemm``, no underscore).

    Upstream SGLang's CUDA backend is the unrelated ``deep_gemm`` package.
    The BW1100 image ships ``deepgemm`` with gfx938 code objects and a
    different grouped-GEMM ABI.  Keep the two capability bits separate so an
    accidentally installed CUDA wheel can never be selected on HIP.
    """

    if not _is_hip:
        return False
    try:
        import deepgemm  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        return False
    return True


HCU_DEEPGEMM_AVAILABLE = _compute_enable_hcu_deepgemm()
# Compatibility alias for provider modules that need to import the vendor ABI.
# Dispatch/layout policy must additionally check the selected MoE runner.
ENABLE_HCU_DEEPGEMM = HCU_DEEPGEMM_AVAILABLE
ENABLE_DEEPGEMM = ENABLE_JIT_DEEPGEMM or HCU_DEEPGEMM_AVAILABLE

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_sm100_supported()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (
    DEEPGEMM_SCALE_UE8M0 or _is_musa or ENABLE_HCU_DEEPGEMM
)
