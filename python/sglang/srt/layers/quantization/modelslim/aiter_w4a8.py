"""ModelSlim W4A8 MoE → aiter fused MoE (Plan A on HCU).

AscendV1 checkpoint packs int4 along **N**; aiter MOE_C expects int4 along **K**
plus marlin shuffle. This module converts weights at load time and runs
``aiter_moe`` in ``apply``.

Shared-expert W8A8 Linear layers are unchanged (existing ModelSlim linear /
optional ``get_aiter_w8a8_int8_quant_info`` path).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any, Iterator, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.utils import get_bool_env_var

# Empirically: Ascend Triton golden vs aiter MOE_C matches after scale/16.
MODELSLIM_TO_AITER_SCALE_DIV = 16.0

# Known-good MOE_C kernel modes for Kimi-K3 TP16 W4A8 (gfx938 tuned JSON, M<=256).
_DEFAULT_W4A8_MOE_C_GEMM1 = {"BLOCK_SIZE_M": 16, "MODE": 374116}
_DEFAULT_W4A8_MOE_C_GEMM2 = {"BLOCK_SIZE_M": 16, "MODE": 384116}
_DEFAULT_W4A8_MOE_C_CONFIG = _DEFAULT_W4A8_MOE_C_GEMM1

logger = logging.getLogger(__name__)

_aiter_available = False
try:
    from aiter.moe import (
        AiterMoeConfig,
        MoeQuantType,
        MoeSolutionType,
        aiter_moe,
        get_aiter_moe_config,
    )

    _aiter_available = True
except Exception:  # pragma: no cover - optional dependency
    AiterMoeConfig = None  # type: ignore
    MoeQuantType = None  # type: ignore
    MoeSolutionType = None  # type: ignore
    aiter_moe = None  # type: ignore
    get_aiter_moe_config = None  # type: ignore


_situ_patch_installed = False
_moe_c_situ_glu_fn = None


def _resolve_moe_c_situ_glu():
    """Resolve ``moe_c_situ_glu`` across aiter wheel variants.

    Log ``kimi_k3_node_113346`` failed with::

        ImportError: cannot import name 'moe_c_situ_glu' from 'aiter'

    Some HCU images expose the kernel only via ``aiter.ops.moe_c_op`` (not
  re-exported in ``aiter.__init__``).
    """
    global _moe_c_situ_glu_fn
    if _moe_c_situ_glu_fn is not None:
        return _moe_c_situ_glu_fn

    import importlib

    for mod_name in ("aiter.ops.moe_c_op", "aiter.fused_moe_c"):
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, "moe_c_situ_glu", None)
            if fn is not None:
                _moe_c_situ_glu_fn = fn
                return fn
        except Exception:
            continue

    try:
        import aiter

        fn = getattr(aiter, "moe_c_situ_glu", None)
        if fn is not None:
            _moe_c_situ_glu_fn = fn
            return fn
    except Exception:
        pass

    try:
        from boltops.fused_moe.triton.moe_activation import triton_situ_and_mul

        def _triton_situ_glu(
            out: torch.Tensor,
            inp: torch.Tensor,
            beta1: float = 4.0,
            beta2: float = 25.0,
            **_: Any,
        ) -> None:
            triton_situ_and_mul(out, inp, beta=beta1, linear_beta=beta2)

        _moe_c_situ_glu_fn = _triton_situ_glu
        return _triton_situ_glu
    except Exception as exc:
        raise ImportError(
            "moe_c_situ_glu is unavailable in this aiter build "
            "(tried aiter.ops.moe_c_op, aiter.fused_moe_c, aiter, boltops)"
        ) from exc


def _call_moe_c_situ_glu(
    activated_out: torch.Tensor,
    ffn1_out_2d: torch.Tensor,
    *,
    beta1: float,
    beta2: float,
) -> None:
    fn = _resolve_moe_c_situ_glu()
    try:
        fn(activated_out, ffn1_out_2d, beta1=beta1, beta2=beta2)
    except TypeError:
        fn(activated_out, ffn1_out_2d, beta=beta1, linear_beta=beta2)


def _patch_aiter_apply_activation_for_situ() -> None:
    """Route gated ``situ`` to ``moe_c_situ_glu`` inside ``_apply_activation``.

    Log ``kimi_k3_node_111950`` failed with::

        ValueError: Unsupported gated activation: situ
        at moe_activation._apply_activation

    Some aiter wheels pass ``_normalize_activation_and_gate('situ')`` but either
    lack ``elif activation == 'situ'`` in ``fused_experts_impl_marlin`` or take
    the ``else: _apply_activation(...)`` path during CUDA-graph capture. Kimi
    must always hit ``moe_c_situ_glu`` (beta1/2 from gemm1_alpha/limit).
    """
    global _situ_patch_installed
    if _situ_patch_installed:
        return
    try:
        import aiter.ops.triton.moe_activation as act_mod

        original = act_mod._apply_activation
        if getattr(original, "_modelslim_situ_patched", False):
            _situ_patch_installed = True
            return

        def _apply_activation_with_situ(
            activation: str,
            is_gated: bool,
            activated_out: torch.Tensor,
            ffn1_out_2d: torch.Tensor,
            gemm1_alpha: Optional[float],
            gemm1_limit: Optional[float],
        ) -> None:
            aliases = getattr(act_mod, "_ACTIVATION_ALIASES", {})
            act = aliases.get(str(activation).lower(), str(activation).lower())
            if act == "situ" and is_gated:
                beta1 = 4.0 if gemm1_alpha is None else float(gemm1_alpha)
                beta2 = 25.0 if gemm1_limit is None else float(gemm1_limit)
                _call_moe_c_situ_glu(
                    activated_out,
                    ffn1_out_2d,
                    beta1=beta1,
                    beta2=beta2,
                )
                return
            return original(
                activation,
                is_gated,
                activated_out,
                ffn1_out_2d,
                gemm1_alpha,
                gemm1_limit,
            )

        _apply_activation_with_situ._modelslim_situ_patched = True  # type: ignore[attr-defined]
        act_mod._apply_activation = _apply_activation_with_situ

        try:
            import aiter.fused_moe_c as fused_moe_c

            fused_moe_c._apply_activation = _apply_activation_with_situ
        except Exception:
            pass

        # Fail fast if no situ kernel is available on this image.
        _resolve_moe_c_situ_glu()

        _situ_patch_installed = True
    except Exception:
        pass


def _ensure_aiter_situ_supported() -> None:
    """Kimi uses ``activation=situ``; older aiter builds omit it from the table."""
    try:
        import aiter.ops.triton.moe_activation as act_mod

        supported = act_mod._SUPPORTED_ACTIVATIONS
        if isinstance(supported, set):
            supported.add("situ")
        # Some boltops builds list situ as gated-only; aiter MOE_C accepts gated situ.
        gated_only = getattr(act_mod, "_GATED_ONLY_ACTIVATIONS", None)
        if isinstance(gated_only, set):
            gated_only.discard("situ")
    except Exception:
        pass
    _patch_aiter_apply_activation_for_situ()


_moe_configs_marlin_patch_installed = False


def _normalize_moe_c_config_table(configs: dict) -> dict:
    return {int(k): dict(v) for k, v in configs.items()}


def _install_moe_configs_marlin_patch() -> None:
    """Fallback gfx938 JSON load when LRU-cached ``get_moe_configs_marlin`` is None.

    Serving log ``kimi_k3_node0_224052`` showed lookups under bare
    ``moe_c_configs/E=896,...`` (cache miss / wrong arch) while tuned files live
    under ``moe_c_configs/gfx938/int8_w4a8/``. Once ``None`` is cached, later
    calls never retry; patch clears cache and reads JSON directly.
    """
    global _moe_configs_marlin_patch_installed
    if _moe_configs_marlin_patch_installed:
        return
    try:
        import aiter.fused_moe_c as fused_moe_c
    except Exception:
        return

    original = fused_moe_c.get_moe_configs_marlin

    def _patched_get_moe_configs_marlin(
        E: int,
        N: int,
        dtype: Optional[str] = None,
        block_n: Optional[int] = None,
        block_k: Optional[int] = None,
        is_bottom: bool = False,
        use_moe_wna16_cuda: bool = False,
        K: Optional[int] = None,
    ) -> Optional[dict]:
        configs = original(
            E,
            N,
            dtype,
            block_n,
            block_k,
            is_bottom,
            use_moe_wna16_cuda,
            K,
        )
        if configs is not None:
            return configs

        logical_k = K
        if logical_k is None and dtype in ("int8_w4a8", "fp4_w4a8"):
            return None

        _clear_moe_configs_marlin_cache()
        for n_try in _w4a8_marlin_n_candidates(N):
            table = _load_w4a8_moe_c_json_table(
                e=E,
                n=n_try,
                k=int(logical_k or 0),
                is_bottom=is_bottom,
                borrow_e=0,
            )
            if table is not None:
                return _normalize_moe_c_config_table(table)

        return original(
            E,
            N,
            dtype,
            block_n,
            block_k,
            is_bottom,
            use_moe_wna16_cuda,
            K,
        )

    _patched_get_moe_configs_marlin.__wrapped__ = original  # type: ignore[attr-defined]
    fused_moe_c.get_moe_configs_marlin = _patched_get_moe_configs_marlin
    _moe_configs_marlin_patch_installed = True


if _aiter_available:
    _ensure_aiter_situ_supported()
    _install_moe_configs_marlin_patch()


def is_aiter_w4a8_available() -> bool:
    return _aiter_available


def _normalize_moe_activation(activation: Any) -> str:
    """Map SGLang MoE activation (str / int enum) to aiter name."""
    if isinstance(activation, int):
        # Match triton fused_moe convention: 0 silu, 1 gelu, 2 situ.
        return {0: "silu", 1: "gelu", 2: "situ"}.get(activation, "situ")
    act = str(activation or "situ").strip().lower()
    if act in {"2", "situ"}:
        return "situ"
    if act in {"0", "silu", "swiglu"}:
        return "silu"
    if act in {"1", "gelu"}:
        return "gelu"
    return act or "situ"


def _signed_nibble_to_int8(n: torch.Tensor) -> torch.Tensor:
    n = n.to(torch.int32)
    return torch.where(n >= 8, n - 16, n).to(torch.int8)


def _unpack_int4_ascend_along_last_dim(packed: torch.Tensor) -> torch.Tensor:
    """Ascend nibble order: even=low, odd=high."""
    wu = packed.to(torch.int32) & 0xFF
    low = _signed_nibble_to_int8(wu & 0x0F)
    high = _signed_nibble_to_int8((wu >> 4) & 0x0F)
    out = torch.empty(
        *packed.shape[:-1],
        packed.shape[-1] * 2,
        dtype=torch.int8,
        device=packed.device,
    )
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def _pack_int4_aiter_along_last_dim(logical: torch.Tensor) -> torch.Tensor:
    """aiter nibble order: even=high, odd=low."""
    if logical.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {logical.shape}")
    even = logical[..., 0::2].to(torch.int32) & 0x0F
    odd = logical[..., 1::2].to(torch.int32) & 0x0F
    return ((even << 4) | odd).to(torch.int8).contiguous()


def ascend_w4a8_unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Checkpoint (..., N/2, K) → logical (..., N, K)."""
    w_t = packed.transpose(-1, -2).contiguous()
    logical_kn = _unpack_int4_ascend_along_last_dim(w_t)
    return logical_kn.transpose(-1, -2).contiguous()


def _repack_w4a8_contiguous_to_blocked(packed_nk_half: torch.Tensor) -> torch.Tensor:
    """Contiguous K-pack [N, K/2] → blocked nibble layout expected by MOE_C shuffle."""
    n, k_half = packed_nk_half.shape
    w_u8 = packed_nk_half.to(torch.uint8)
    w_unpacked = torch.stack([(w_u8 >> 4) & 0x0F, w_u8 & 0x0F], dim=-1).view(n, -1)
    blocks = w_unpacked.view(n, -1, 8)
    return ((blocks[..., :4] << 4) | blocks[..., 4:]).view(n, k_half)


def _safe_w4a8_moe_layout_shuffle(w4a8_w: torch.Tensor) -> torch.Tensor:
    """MOE_C layout shuffle for ``[N, K/2]``.

    Mirrors ``aiter.ops.shuffle.w4a8_moe_layout_shuffle``, but **never** forces
    ``n_tile=256`` when ``N % 256 != 0`` (TP-sharded MoE, e.g. inter=192 → N=384).
    Older aiter builds that always used ``n_tile=256`` raise::

        RuntimeError: shape '[56, 32, 1, 256]' is invalid for input of size 688128
    """
    full = w4a8_w.T
    k_tile = 32
    size_k, size_n = full.shape
    n_tile = 256 if size_n % 256 == 0 else size_n
    if size_k % k_tile != 0 or size_n % n_tile != 0 or n_tile % 32 != 0:
        return w4a8_w.contiguous()
    full = full.reshape(size_k // k_tile, k_tile, size_n // n_tile, n_tile)
    full = full.permute((0, 2, 3, 1)).contiguous()
    full = full.reshape(
        size_k // k_tile,
        size_n // n_tile,
        n_tile // 32,
        32,
        k_tile // 8,
        8,
    )
    return full.permute(0, 1, 2, 4, 3, 5).contiguous()


def _repack_and_shuffle_w4a8(weight_data: torch.Tensor, e: int) -> torch.Tensor:
    """Per-expert blocked repack + safe MOE_C shuffle (in-place, no full clone)."""
    for i in range(e):
        expert = weight_data[i]
        n, k_half = expert.shape
        blocked = _repack_w4a8_contiguous_to_blocked(expert)
        shuffled = _safe_w4a8_moe_layout_shuffle(blocked)
        weight_data[i].copy_(shuffled.reshape(n, k_half))
        del blocked, shuffled
    return weight_data


def convert_ascend_w4a8_weight_to_aiter(
    weight: torch.Tensor, *, apply_shuffle: bool = True
) -> torch.Tensor:
    """Ascend N-pack (..., N/2, K) → aiter K-pack (..., N, K/2) + optional shuffle.

    Avoids holding unpack + pack + shuffle clones of the full expert stack at once.
    """
    logical = ascend_w4a8_unpack_int4(weight)
    # Drop Ascend-packed storage ASAP if we own the only ref (caller replaces param).
    packed = _pack_int4_aiter_along_last_dim(logical)
    del logical
    if apply_shuffle and packed.dim() == 3:
        return _repack_and_shuffle_w4a8(packed, packed.shape[0])
    return packed


def _drop_unused_modelslim_moe_params(layer: torch.nn.Module) -> None:
    """Free Ascend-only MoE extras that aiter_moe never reads (offsets / scale_bias)."""
    for name in (
        "w13_weight_offset",
        "w2_weight_offset",
        "w13_scale_bias",
        "w2_scale_bias",
        "w13_weight_scale_second",
        "w2_weight_scale_second",
        "w13_weight_offset_second",
        "w2_weight_offset_second",
    ):
        if not hasattr(layer, name):
            continue
        param = getattr(layer, name)
        # Keep a tiny CUDA placeholder so accidental getattr still works.
        placeholder = torch.nn.Parameter(
            torch.empty(0, device=param.device, dtype=param.dtype),
            requires_grad=False,
        )
        setattr(layer, name, placeholder)
        del param


def convert_modelslim_moe_weights_to_aiter(
    layer: torch.nn.Module,
    *,
    scale_div: float = MODELSLIM_TO_AITER_SCALE_DIV,
    apply_shuffle: bool = True,
    skip_scale_bias: bool = True,
) -> None:
    """In-place AscendV1 MoE params → aiter MOE_C layout on ``layer``.

    - ``w13_weight``: (E, inter, K) Ascend → (E, 2*inter, K/2) aiter
    - ``w2_weight``:  (E, K/2, inter) Ascend → (E, K, inter/2) aiter
    - scales stay fp32; divided by ``scale_div`` for aiter convention
    - Ascend-only ``offset`` / ``scale_bias`` are dropped (aiter path ignores them)
    """
    if not _aiter_available:
        raise ImportError(
            "aiter is required for SGLANG_MODELSLIM_MOE_USE_AITER / "
            "--moe-runner-backend aiter with ModelSlim W4A8"
        )

    old_w13 = layer.w13_weight.data
    old_w2 = layer.w2_weight.data
    w13 = convert_ascend_w4a8_weight_to_aiter(old_w13, apply_shuffle=apply_shuffle)
    w2 = convert_ascend_w4a8_weight_to_aiter(old_w2, apply_shuffle=apply_shuffle)
    layer.w13_weight = Parameter(w13.contiguous(), requires_grad=False)
    layer.w2_weight = Parameter(w2.contiguous(), requires_grad=False)
    del old_w13, old_w2, w13, w2

    w13_s = layer.w13_weight_scale.data.to(torch.float32)
    w2_s = layer.w2_weight_scale.data.to(torch.float32)
    if w13_s.ndim == 2:
        w13_s = w13_s.unsqueeze(-1)
    if w2_s.ndim == 2:
        w2_s = w2_s.unsqueeze(-1)
    if scale_div != 1.0:
        w13_s = w13_s / scale_div
        w2_s = w2_s / scale_div
    layer.w13_weight_scale = Parameter(w13_s.contiguous(), requires_grad=False)
    layer.w2_weight_scale = Parameter(w2_s.contiguous(), requires_grad=False)
    del w13_s, w2_s

    if skip_scale_bias:
        _drop_unused_modelslim_moe_params(layer)

    layer._modelslim_aiter_converted = True  # type: ignore[attr-defined]

    if torch.cuda.is_available():
        e = layer.w13_weight.size(0)
        n1 = layer.w13_weight.size(1)
        k_logical = layer.w13_weight.size(2) * 2
        _warm_w4a8_moe_c_config_cache(
            e=e, n_marlin=n1 // 2, k=k_logical, borrow_e=e
        )
        torch.cuda.empty_cache()


@contextlib.contextmanager
def borrow_moe_c_tuned_configs(borrow_experts: int) -> Iterator[None]:
    """Borrow tuned MOE_C JSON when local E has no config file.

    Mirrors ``aiter_API.aiter_moe.borrow_moe_c_tuned_configs`` for serving.
    """
    if borrow_experts <= 0:
        yield
        return
    import aiter.fused_moe_c as fused_moe_c

    original = fused_moe_c.get_moe_configs_marlin

    def _wrapped(E: int, N: int, *args, **kwargs):
        configs = original(E, N, *args, **kwargs)
        if configs is None and E != borrow_experts:
            configs = original(borrow_experts, N, *args, **kwargs)
        return configs

    fused_moe_c.get_moe_configs_marlin = _wrapped
    try:
        yield
    finally:
        fused_moe_c.get_moe_configs_marlin = original


def _aiter_supports_activation(activation: str) -> bool:
    """Whether fused ``aiter_moe`` can run this activation.

    Prefer ``aiter.ops.triton.moe_activation._SUPPORTED_ACTIVATIONS`` when
    present. Always allow ``situ`` for ModelSlim/Kimi: MOE_C ``aiter_moe``
    accepts it even on builds whose Triton activation table omits ``situ``
    (that mismatch previously aborted CUDA-graph capture with
    ``activation='situ'``).
    """
    act = _normalize_moe_activation(activation)
    if act in {"situ", "silu", "gelu", "relu2"}:
        return True
    try:
        from aiter.ops.triton.moe_activation import _SUPPORTED_ACTIVATIONS

        return act in {str(x).lower() for x in _SUPPORTED_ACTIVATIONS}
    except Exception:
        return False


def _is_gated_activation(activation: str) -> bool:
    return _normalize_moe_activation(activation) in {
        "silu",
        "situ",
        "gelu",
        "swigluoai",
        "swiglustep",
        "gelu_tanh",
    }


def _clear_moe_configs_marlin_cache() -> None:
    try:
        import aiter.fused_moe_c as fused_moe_c

        fn = fused_moe_c.get_moe_configs_marlin
        while hasattr(fn, "__wrapped__"):
            if hasattr(fn, "cache_clear"):
                fn.cache_clear()
            fn = fn.__wrapped__
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
    except Exception:
        pass


def _arch_candidates_for_moe_c_json() -> list[str]:
    """Arch subdirs to try when loading MOE_C JSON (BW200B may not report gfx938)."""
    candidates: list[str] = []
    try:
        import aiter.fused_moe_c as fused_moe_c

        try:
            candidates.append(fused_moe_c._get_gfx_version())
        except Exception:
            pass
    except Exception:
        pass
    for arch in ("gfx938", "gfx92a", "gfx936", "gfx928"):
        if arch not in candidates:
            candidates.append(arch)
    return candidates


def _load_w4a8_moe_c_json_table(
    *,
    e: int,
    n: int,
    k: int,
    is_bottom: bool = False,
    borrow_e: int = 0,
) -> Optional[dict]:
    """Read int8_w4a8 JSON directly (bypasses poisoned LRU cache / wrong arch)."""
    try:
        import aiter.fused_moe_c as fused_moe_c
    except Exception:
        return None

    cat = fused_moe_c._moe_c_config_category("int8_w4a8", None)
    root = fused_moe_c._moe_c_config_root()

    expert_ids = [e]
    if borrow_e > 0 and borrow_e not in expert_ids:
        expert_ids.append(borrow_e)

    names: list[str] = []
    for experts in expert_ids:
        name = fused_moe_c.get_config_file_name_marlin(
            experts, n, "int8_w4a8", None, is_bottom, True, k
        )
        if name not in names:
            names.append(name)

    paths: list[str] = []
    for name in names:
        for arch in _arch_candidates_for_moe_c_json():
            if cat:
                paths.append(os.path.join(root, arch, cat, name))
            paths.append(os.path.join(root, arch, name))
        for arch in _arch_candidates_for_moe_c_json():
            paths.append(
                fused_moe_c._find_moe_c_config_file(name, "int8_w4a8", None, arch)
            )
        # Last resort: scan every arch/category subtree for this filename.
        try:
            for arch in os.listdir(root):
                arch_dir = os.path.join(root, arch)
                if not os.path.isdir(arch_dir):
                    continue
                if cat:
                    paths.append(os.path.join(arch_dir, cat, name))
                for sub in os.listdir(arch_dir):
                    paths.append(os.path.join(arch_dir, sub, name))
        except OSError:
            pass

    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                logger.info("ModelSlim W4A8 MoE loaded config from %s", path)
                return _normalize_moe_c_config_table(json.load(fh))
    return None


def _warm_w4a8_moe_c_config_cache(
    *, e: int, n_marlin: int, k: int, borrow_e: int = 0
) -> None:
    """Prime marlin lookup for GEMM1/GEMM2 after CUDA is up (avoids cached None)."""
    _clear_moe_configs_marlin_cache()
    try:
        import aiter.fused_moe_c as fused_moe_c
    except Exception:
        return
    for is_bottom in (False, True):
        for experts in dict.fromkeys([e, borrow_e] if borrow_e > 0 else [e]):
            try:
                fused_moe_c.get_moe_configs_marlin(
                    E=experts,
                    N=n_marlin,
                    dtype="int8_w4a8",
                    is_bottom=is_bottom,
                    use_moe_wna16_cuda=True,
                    K=k,
                )
            except TypeError:
                fused_moe_c.get_moe_configs_marlin(
                    E=experts,
                    N=n_marlin,
                    dtype="int8_w4a8",
                    is_bottom=is_bottom,
                    use_moe_wna16_cuda=True,
                )
            except Exception:
                pass


def _pick_closest_moe_c_config(configs: dict, m: int) -> dict:
    try:
        from aiter.moe import _pick_closest_config

        picked = dict(_pick_closest_config(configs, m))
    except Exception:
        keys = sorted(int(k) for k in configs.keys())
        best = min(keys, key=lambda bs: abs(bs - m))
        picked = dict(
            configs[str(best)] if str(best) in configs else configs[best]
        )
    picked.setdefault("BLOCK_SIZE_M", 16)
    picked.setdefault("key_selected", m)
    return picked


def _w4a8_marlin_n_candidates(n: int) -> list[int]:
    """Marlin JSON ``N`` may be TP-sharded (192/384/768) — try common Kimi sizes."""
    candidates = [n]
    for alt in (192, 384, 768, 1536, 3072, n // 2, n * 2):
        if alt > 0 and alt not in candidates:
            candidates.append(alt)
    return candidates


def _try_load_w4a8_marlin_configs(
    *,
    e: int,
    n: int,
    k: int,
    borrow_e: int = 0,
    is_bottom: bool = False,
) -> Optional[dict]:
    """Load tuned int8_w4a8 marlin JSON (gfx938, always with K)."""
    for n_try in _w4a8_marlin_n_candidates(n):
        table = _load_w4a8_moe_c_json_table(
            e=e, n=n_try, k=k, is_bottom=is_bottom, borrow_e=borrow_e
        )
        if table is not None:
            return table

    _clear_moe_configs_marlin_cache()
    try:
        import aiter.fused_moe_c as fused_moe_c
    except Exception:
        return None

    expert_candidates = [e]
    if borrow_e > 0 and borrow_e not in expert_candidates:
        expert_candidates.append(borrow_e)

    for experts in expert_candidates:
        try:
            configs = fused_moe_c.get_moe_configs_marlin(
                E=experts,
                N=n,
                dtype="int8_w4a8",
                is_bottom=is_bottom,
                use_moe_wna16_cuda=True,
                K=k,
            )
        except TypeError:
            configs = fused_moe_c.get_moe_configs_marlin(
                E=experts,
                N=n,
                dtype="int8_w4a8",
                is_bottom=is_bottom,
                use_moe_wna16_cuda=True,
            )
        if configs is not None:
            return configs
    return None


def _make_w4a8_moe_c_config(config: dict) -> Any:
    return AiterMoeConfig(
        quant_type=MoeQuantType.W4A8,
        solution_type=MoeSolutionType.MOE_C,
        config=config,
        need_shuffle=True,
        need_shuffle_scale=False,
    )


def _resolve_w4a8_moe_config(
    *,
    m: int,
    e: int,
    n1: int,
    k: int,
    topk: int,
    dtype: torch.dtype,
    activation: str,
    borrow_e: int = 0,
) -> Any:
    """Resolve MOE_C config for ModelSlim W4A8.

    Serving log ``kimi_k3_node0_181229`` failed CUDA-graph capture with::

        aiter did not find a valid W4A8 MoE config for
        M=77, E=896, N1=384, N2=192, K=3584, ... activation=situ

    Warnings showed lookup of ``E=896,N=384`` **without K** (ungated / incomplete
    marlin path). Tuned files live at ``gfx938/.../E=896,N=192,K=3584``. We:
    1. Call ``get_aiter_moe_config`` with explicit ``gated=True`` for SiTU.
    2. Fall back to direct ``get_moe_configs_marlin(..., K=k)`` for N=n1//2 then N=n1.
    3. Last resort: known-good default MODE from the Kimi tuned JSON.
    """
    act = _normalize_moe_activation(activation)
    gated = _is_gated_activation(act)
    n_marlin = (n1 // 2) if gated else n1
    n2 = n_marlin

    # 1) Prefer official helper (handles solution-type priority).
    try:
        kwargs = dict(
            M=m,
            E=e,
            N1=n1,
            N2=n2,
            K=k,
            top_k=topk,
            block_size=0,
            dtype=dtype,
            quant_type=MoeQuantType.W4A8,
            activation=act,
            gated=gated,
        )
        try:
            status, moe_config = get_aiter_moe_config(**kwargs)
        except TypeError:
            kwargs.pop("gated", None)
            try:
                status, moe_config = get_aiter_moe_config(**kwargs)
            except TypeError:
                kwargs.pop("activation", None)
                status, moe_config = get_aiter_moe_config(**kwargs)
        if status and moe_config is not None and moe_config.config is not None:
            return moe_config
    except Exception as exc:
        logger.warning(
            "get_aiter_moe_config failed for W4A8 MoE "
            "(M=%s E=%s N1=%s K=%s act=%s): %s — trying marlin fallback",
            m,
            e,
            n1,
            k,
            act,
            exc,
        )

    # 2) Direct marlin JSON (always pass K; try gated N then full N1).
    for n_try in dict.fromkeys([n_marlin, n1, n1 // 2]):
        if n_try <= 0:
            continue
        configs = _try_load_w4a8_marlin_configs(
            e=e, n=n_try, k=k, borrow_e=borrow_e
        )
        if configs is None:
            continue
        picked = _pick_closest_moe_c_config(configs, m)
        logger.info(
            "ModelSlim W4A8 MoE using marlin fallback config "
            "E=%s N=%s K=%s M=%s -> %s",
            e,
            n_try,
            k,
            m,
            picked,
        )
        return _make_w4a8_moe_c_config(picked)

    # 3) Hard default so CUDA-graph capture can proceed (same MODE as tuned JSON).
    logger.warning(
        "ModelSlim W4A8 MoE: no tuned marlin JSON for "
        "M=%s E=%s N1=%s K=%s; using default %s",
        m,
        e,
        n1,
        k,
        _DEFAULT_W4A8_MOE_C_CONFIG,
    )
    return _make_w4a8_moe_c_config(dict(_DEFAULT_W4A8_MOE_C_GEMM1))


@torch._dynamo.disable()
def apply_modelslim_aiter_moe(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str = "situ",
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    routed_scaling_factor: float = 1.0,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    borrow_tuned_config_experts: int = 0,
) -> torch.Tensor:
    """Fused W4A8 MoE via ``get_aiter_moe_config`` + ``aiter_moe``.

    Aligned with ``aiter_API.aiter_moe.forward_aiter_modelslim_w4a8`` (unit-test path).
    """
    if not _aiter_available or aiter_moe is None or get_aiter_moe_config is None:
        raise ImportError("aiter is required for ModelSlim aiter MoE apply()")

    if not getattr(layer, "_modelslim_aiter_converted", False):
        raise RuntimeError(
            "ModelSlim MoE layer was not converted to aiter layout. "
            "process_weights_after_loading must run the aiter conversion path."
        )

    act = _normalize_moe_activation(activation)
    if not _aiter_supports_activation(act):
        raise ValueError(
            f"aiter does not support activation={activation!r} for ModelSlim W4A8 MoE"
        )

    x = hidden_states
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)

    e = layer.w13_weight.size(0)
    if global_num_experts < 0:
        global_num_experts = e
    k = x.size(-1)
    n1 = layer.w13_weight.size(1)
    n2 = n1 // 2
    topk = topk_ids.size(-1)
    m = x.size(0) if x.dim() == 2 else x.size(0) * x.size(1)
    borrow_e = borrow_tuned_config_experts or e

    cache = getattr(layer, "_modelslim_aiter_moe_config_cache", None)
    if cache is None:
        cache = {}
        layer._modelslim_aiter_moe_config_cache = cache  # type: ignore[attr-defined]
    cache_key = (m, e, n1, k, topk, str(x.dtype), act)
    moe_config = cache.get(cache_key)

    with borrow_moe_c_tuned_configs(borrow_e):
        if moe_config is None:
            status, moe_config = get_aiter_moe_config(
                M=m,
                E=e,
                N1=n1,
                N2=n2,
                K=k,
                top_k=topk,
                block_size=0,
                dtype=x.dtype,
                quant_type=MoeQuantType.W4A8,
                activation=act,
            )
            if not status or moe_config is None or moe_config.config is None:
                raise RuntimeError(
                    "aiter did not find a valid W4A8 MoE config for "
                    f"M={m}, E={e}, N1={n1}, N2={n2}, K={k}, topk={topk}, "
                    f"dtype={x.dtype}, activation={act!r}. "
                    "Tuned JSON may be missing on this device."
                )
            cache[cache_key] = moe_config

        return aiter_moe(
            x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            moe_config=moe_config,
            activation=act,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha,
            gemm1_limit=gemm1_limit,
        )


def modelslim_moe_use_aiter_requested() -> bool:
    """Env opt-in when ``--moe-runner-backend`` is auto."""
    return get_bool_env_var("SGLANG_MODELSLIM_MOE_USE_AITER", default="false")

