import functools
import importlib.util
import logging
from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once
from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var, is_hcu, is_hip

_is_hip = is_hip()
_is_hcu = is_hcu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

_linear_bf16_fp32_algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()
logger = logging.getLogger(__name__)
# Only gfx936 has a measured dispatch profile; other supported HCU targets
# remain on the correctness-preserving cublas fallback until benchmarked.
_HCU_AUTO_TUNED_ARCHS = frozenset(("gfx936",))
# Measured on gfx936: cublas is faster than AITER for N=512, K=6144 in this M window.
_HCU_AUTO_N512_CUBLAS_PROFILE_K = 6144
_HCU_AUTO_N512_CUBLAS_M_RANGE = (1408, 2816)
# Measured on gfx936: SGL is faster for the single-token router shape.
_HCU_AUTO_N256_SGL_PROFILE_K = 4096
_AUTO_AITER_DISABLED = False
_AUTO_SGL_DISABLED = False
_HIP_BF16_OUT_DTYPE_SUPPORTED: Optional[bool] = None
_HPC_GEMM_WEIGHT_CACHE_ATTR = "_sglang_bf16xfp32_weight_cache"
# The HPC-Ops bf16xfp32 GEMM consumes the fp32 weight decomposed into two
# bf16 halves: w_high = w.bf16 and w_low = ((w - w_high) / scale).bf16 with
# scale = 1/256, so that w ~= w_high + scale * w_low.
_HPC_GEMM_WEIGHT_SCALE = 1.0 / 256.0
# Set at model init, never lazily, so all ranks agree; see
# mark_hpc_bf16xfp32_gemm_enabled.
_hpc_gemm_enabled = False


@functools.cache
def _hpc_gemm_bf16xfp32_available() -> bool:
    """HPC-Ops (https://github.com/Tencent/hpc-ops) ships sm90a kernels."""
    if importlib.util.find_spec("hpc") is None:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


def _can_use_hpc_gemm_bf16xfp32(
    x: torch.Tensor, y: torch.Tensor, *, min_m: int = 8
) -> bool:
    if x.dim() != 2 or y.dim() != 2 or x.shape[1] != y.shape[1]:
        return False
    if x.shape[0] < min_m:
        return False
    if not (x.is_cuda and y.is_cuda):
        return False
    if x.dtype != torch.bfloat16 or y.dtype != torch.float32:
        return False
    if not (x.is_contiguous() and y.is_contiguous()):
        return False
    if y.shape[0] % 64 != 0:
        return False
    return _hpc_gemm_bf16xfp32_available()


def _get_bf16xfp32_weight_split(
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the fp32 weight for the HPC-Ops kernel and cache the result
    (plus the split-K flag workspace, which the kernel leaves zeroed) on the
    weight tensor.

    The cache key is layout-only: in-place loader writes
    (``param.data.copy_()``) are unobservable, and captured CUDA graphs
    replay the split buffers by address, so the split is computed once and
    online weight updates are rejected instead (see
    hpc_bf16xfp32_gemm_enabled).
    """
    import hpc

    if not hpc_bf16xfp32_gemm_enabled():
        raise RuntimeError(
            "Call mark_hpc_bf16xfp32_gemm_enabled() at model init before "
            "routing GEMMs to the HPC-Ops bf16xfp32 kernel."
        )

    cache_key = (
        y.data_ptr(),
        tuple(y.shape),
        tuple(y.stride()),
        y.device.index,
        y.dtype,
    )
    cache = getattr(y, _HPC_GEMM_WEIGHT_CACHE_ATTR, None)
    if cache is not None and cache[0] == cache_key:
        return cache[1], cache[2], cache[3]

    with torch.no_grad():
        w_high = y.to(torch.bfloat16)
        w_low = ((y - w_high.float()) / _HPC_GEMM_WEIGHT_SCALE).to(torch.bfloat16)
    split_flag = hpc.get_gemm_bf16xfp32_workspace(y.shape[0])
    setattr(y, _HPC_GEMM_WEIGHT_CACHE_ATTR, (cache_key, w_high, w_low, split_flag))
    return w_high, w_low, split_flag


def mark_hpc_bf16xfp32_gemm_enabled() -> None:
    """Declare at model init that GEMMs may route to the HPC-Ops bf16xfp32
    kernel (no-op when the kernel is unavailable). Must not be called lazily
    from a forward pass: the state must depend only on startup facts so it
    is identical on every rank."""
    global _hpc_gemm_enabled
    if _hpc_gemm_bf16xfp32_available():
        _hpc_gemm_enabled = True


def hpc_bf16xfp32_gemm_enabled() -> bool:
    """Whether this process may cache bf16xfp32 weight splits. The online
    weight-update APIs reject updates while True (the cache cannot survive
    in-place weight writes). Startup-determined, so all ranks agree."""
    if _hpc_gemm_enabled:
        return True
    return _linear_bf16_fp32_algo == "hpc" and _hpc_gemm_bf16xfp32_available()


def _hip_bf16_out_dtype_error(exc: BaseException) -> bool:
    if isinstance(exc, TypeError):
        return True
    message = str(exc).lower()
    return "out_dtype" in message or "not implemented" in message


def _linear_bf16_fp32_cublas(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    global _HIP_BF16_OUT_DTYPE_SUPPORTED

    if x.is_cuda and x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16:
        if not _is_hip:
            return torch.mm(x, y.t(), out_dtype=torch.float32)
        if _HIP_BF16_OUT_DTYPE_SUPPORTED is not False:
            try:
                output = torch.mm(x, y.t(), out_dtype=torch.float32)
            except (TypeError, RuntimeError) as exc:
                if not _hip_bf16_out_dtype_error(exc):
                    raise
                _HIP_BF16_OUT_DTYPE_SUPPORTED = False
                logger.warning(
                    "HIP BF16 out_dtype GEMM is unavailable; using FP32 operands"
                )
            else:
                _HIP_BF16_OUT_DTYPE_SUPPORTED = True
                return output
        return torch.mm(x.float(), y.float().t())
    return torch.mm(x.float(), y.float().t())


def _linear_bf16_fp32_hpc(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    min_m: int = 8,
) -> Optional[torch.Tensor]:
    if not _can_use_hpc_gemm_bf16xfp32(x, y, min_m=min_m):
        return None

    import hpc

    w_high, w_low, split_flag = _get_bf16xfp32_weight_split(y)
    return hpc.gemm_bf16xfp32(
        x,
        w_high,
        w_low,
        _HPC_GEMM_WEIGHT_SCALE,
        use_fp32_output=True,
        use_splitk=True,
        split_flag=split_flag,
    )


@cache_once
def _hcu_arch_name() -> str:
    if not _is_hcu:
        return ""
    try:
        from sglang.kernels.jit.utils.compile.toolchain import gpu_arch_name

        return gpu_arch_name()
    except Exception:
        return "unknown"


@cache_once
def _hcu_auto_profile_enabled() -> bool:
    return _is_hcu and any(arch in _hcu_arch_name() for arch in _HCU_AUTO_TUNED_ARCHS)


@cache_once
def _get_aiter_tgemm():
    if not _is_hip:
        return None
    try:
        from aiter.tuned_gemm import tgemm
    except (ImportError, OSError, RuntimeError, TypeError):
        return None
    return tgemm


def _is_auto_supported_shape(*, m: int, n: int, k: int) -> bool:
    # The reference AITER path supports arbitrary positive K for the known N set.
    return m > 0 and n in (256, 512, 1024, 2048) and k > 0


def _run_aiter(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    tgemm = _get_aiter_tgemm()
    if tgemm is None:
        raise RuntimeError(
            "linear_bf16_fp32 algo=aiter requires aiter.tuned_gemm"
        )
    return tgemm.mm(x, y, otype=x.dtype).float()


def _select_auto_backend(
    *,
    m: int,
    n: int,
    k: int,
    x_dtype: torch.dtype,
    y_dtype: torch.dtype,
    x_contiguous: bool,
    y_contiguous: bool,
    is_hcu_device: bool,
    hcu_profile_enabled: bool,
    sgl_supported: bool,
    aiter_available: bool,
    aiter_shape_supported: bool = True,
) -> str:
    if (
        not is_hcu_device
        or not hcu_profile_enabled
        or x_dtype != torch.bfloat16
        or y_dtype != torch.bfloat16
        or not x_contiguous
        or not y_contiguous
        or m <= 0
        or n <= 0
        or k <= 0
    ):
        return "cublas"

    if n not in (256, 512, 1024, 2048):
        return "cublas"

    if not _is_auto_supported_shape(m=m, n=n, k=k):
        return "torch" if m <= 768 else "cublas"

    if (
        not _AUTO_SGL_DISABLED
        and sgl_supported
        and n == 256
        and k == _HCU_AUTO_N256_SGL_PROFILE_K
        and m == 1
    ):
        return "sgl"
    if not _AUTO_AITER_DISABLED and aiter_available and aiter_shape_supported:
        n512_min_m, n512_max_m = _HCU_AUTO_N512_CUBLAS_M_RANGE
        if (
            n == 512
            and k == _HCU_AUTO_N512_CUBLAS_PROFILE_K
            and n512_min_m <= m <= n512_max_m
        ):
            return "cublas"
        return "aiter"

    if not _AUTO_SGL_DISABLED and sgl_supported and k % 128 == 0 and m <= 64:
        return "sgl"

    if n == 256:
        return "torch" if m <= 768 else "cublas"
    if n == 512:
        return "torch" if m <= 1024 else "cublas"
    if n in (1024, 2048):
        return "torch" if m <= 768 else "cublas"
    return "cublas"


def _auto_dispatch_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    global _AUTO_AITER_DISABLED, _AUTO_SGL_DISABLED

    if x.dim() != 2 or y.dim() != 2 or x.shape[1] != y.shape[1]:
        return _linear_bf16_fp32_cublas(x, y)

    m, k = (int(x.shape[0]), int(x.shape[1]))
    n = int(y.shape[0])
    runtime_hcu = _is_hcu and x.is_cuda and y.is_cuda and x.device == y.device
    profile_enabled = runtime_hcu and _hcu_auto_profile_enabled()
    common_eligible = (
        profile_enabled
        and x.dtype == torch.bfloat16
        and y.dtype == torch.bfloat16
        and x.is_contiguous()
        and y.is_contiguous()
        and n in (256, 512, 1024, 2048)
        and _is_auto_supported_shape(m=m, n=n, k=k)
    )
    aiter = (
        _get_aiter_tgemm()
        if common_eligible and not _AUTO_AITER_DISABLED
        else None
    )
    sgl_supported = False
    if common_eligible and (
        aiter is None
        or (
            m == 1
            and n == 256
            and k == _HCU_AUTO_N256_SGL_PROFILE_K
        )
    ):
        from .hcu_linear_bf16_fp32 import hcu_linear_bf16_fp32_supported

        sgl_supported = hcu_linear_bf16_fp32_supported(x, y)

    def select(aiter_available: bool, aiter_shape_supported: bool = True) -> str:
        return _select_auto_backend(
            m=m,
            n=n,
            k=k,
            x_dtype=x.dtype,
            y_dtype=y.dtype,
            x_contiguous=x.is_contiguous(),
            y_contiguous=y.is_contiguous(),
            is_hcu_device=runtime_hcu,
            hcu_profile_enabled=profile_enabled,
            sgl_supported=sgl_supported,
            aiter_available=aiter_available,
            aiter_shape_supported=aiter_shape_supported,
        )

    backend = select(aiter is not None)
    if backend == "sgl":
        try:
            from .hcu_linear_bf16_fp32 import hcu_linear_bf16_fp32

            return hcu_linear_bf16_fp32(x, y)
        except (OSError, RuntimeError):
            _AUTO_SGL_DISABLED = True
            logger.warning("Disabling SGL BF16 FP32 GEMM for auto dispatch")
            backend = select(aiter is not None)

    if backend == "torch":
        return torch.nn.functional.linear(x.float(), y.float())
    if backend == "aiter":
        try:
            return _run_aiter(x, y)
        except (OSError, RuntimeError, TypeError, ValueError):
            _AUTO_AITER_DISABLED = True
            logger.warning("Disabling AITER BF16 FP32 GEMM for auto dispatch")
            backend = select(False, aiter_shape_supported=False)
            if backend == "sgl":
                try:
                    from .hcu_linear_bf16_fp32 import hcu_linear_bf16_fp32

                    return hcu_linear_bf16_fp32(x, y)
                except (OSError, RuntimeError):
                    _AUTO_SGL_DISABLED = True
                    logger.warning("Disabling SGL BF16 FP32 GEMM for auto dispatch")
                    backend = select(False, aiter_shape_supported=False)
            if backend == "torch":
                return torch.nn.functional.linear(x.float(), y.float())
    return _linear_bf16_fp32_cublas(x, y)


def prewarm_auto_bf16_fp32() -> None:
    """Warm optional auto backends before CUDA graph or compile setup."""
    global _AUTO_AITER_DISABLED, _AUTO_SGL_DISABLED
    if _linear_bf16_fp32_algo != "auto" or not _hcu_auto_profile_enabled():
        return
    if _get_aiter_tgemm() is None:
        _AUTO_AITER_DISABLED = True
    try:
        from .hcu_linear_bf16_fp32 import _jit_linear_bf16_fp32_module

        _jit_linear_bf16_fp32_module()
    except (OSError, RuntimeError):
        _AUTO_SGL_DISABLED = True
        logger.warning("Disabling SGL BF16 FP32 GEMM after prewarm failure")


def linear_bf16_fp32(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    hpc_kernel_min_m: Optional[int] = None,
) -> torch.Tensor:
    if hpc_kernel_min_m is not None:
        output = _linear_bf16_fp32_hpc(x, y, min_m=hpc_kernel_min_m)
        if output is not None:
            return output
        return _linear_bf16_fp32_cublas(x, y)
    if _linear_bf16_fp32_algo == "auto":
        return _auto_dispatch_bf16_fp32(x, y)
    if _linear_bf16_fp32_algo == "sgl":
        from .hcu_linear_bf16_fp32 import hcu_linear_bf16_fp32

        return hcu_linear_bf16_fp32(x, y)
    if _linear_bf16_fp32_algo == "aiter" and y.dtype == torch.bfloat16:
        return _run_aiter(x, y)
    if _use_aiter and y.dtype == torch.bfloat16:
        return _run_aiter(x, y)
    elif _linear_bf16_fp32_algo == "hpc":
        output = _linear_bf16_fp32_hpc(x, y)
        if output is not None:
            return output
        return _linear_bf16_fp32_cublas(x, y)
    elif _linear_bf16_fp32_algo == "deep_gemm" and y.dtype == torch.bfloat16:
        from sglang.srt.layers import deep_gemm_wrapper

        z = torch.empty(x.size(0), y.size(0), dtype=torch.float32, device=x.device)
        deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, y, z)
        return z
    else:
        return _linear_bf16_fp32_cublas(x, y)
