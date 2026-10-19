"""Framework-local SlimQuant W4A8 support for the HCU MegaMoE INT8 engine.

Resident weights use compressed PACK5 (two signed nibbles per byte). Before
GEMM they are expanded into a stream-owned reusable workspace. This is NOT a
native packed-INT4 GEMM: it trades an unpack launch/bandwidth for retaining
4-bit model storage and reusing the tuned HCU dispatch/GEMM/combine kernels.
"""

import inspect
import math

import torch
import triton
import triton.language as tl


def validate_megamoe_int8_runtime() -> None:
    """Fail early when the installed base wheel is too old for this adapter."""
    import megamoe

    required = ("int8_mega_moe", "mega_moe_pre_dispatch_int8", "SymmBuffer")
    missing = [name for name in required if not hasattr(megamoe, name)]
    if missing:
        raise RuntimeError(
            "The installed MegaMoE wheel lacks the W8A8-I8 runtime required "
            "by SGLang's W4A8 compatibility adapter: " + ", ".join(missing)
        )
    try:
        parameters = inspect.signature(megamoe.SymmBuffer).parameters
    except (TypeError, ValueError):
        parameters = {}
    if parameters and "quant_mode" not in parameters:
        raise RuntimeError(
            "The installed MegaMoE SymmBuffer does not support quant_mode='int8'"
        )


@triton.jit
def _unpack_signed_int4(P, O, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(P + i // 2, i < SIZE, other=0).to(tl.int32)
    q = (b >> tl.where(i % 2 == 0, 4, 0)) & 15
    tl.store(O + i, ((q ^ 8) - 8).to(tl.int8), i < SIZE)


def unpack_signed_int4_out(packed, out, *, block=2048, num_warps=4):
    """Decode high-nibble-first two's-complement bytes, without FP quantization."""
    if packed.dtype not in (torch.int8, torch.uint8) or out.dtype != torch.int8:
        raise TypeError("packed must be byte INT4 and out must be signed INT8")
    if not packed.is_contiguous() or not out.is_contiguous():
        raise ValueError("packed and output must be contiguous")
    if packed.device != out.device or not packed.is_cuda:
        raise ValueError("packed and output must be on the same GPU")
    if out.numel() != 2 * packed.numel():
        raise ValueError("output must contain exactly two elements per packed byte")
    _unpack_signed_int4[(triton.cdiv(out.numel(), block),)](
        packed, out, out.numel(), block, num_warps=num_warps
    )
    return out


def _pack(weight, scale, scale_multiplier):
    if weight.ndim != 3 or weight.dtype not in (torch.int8, torch.uint8):
        raise ValueError("checkpoint weight must be byte [E,N,K/2]")
    e, n, kh = weight.shape
    if n % 256 or kh % 32:
        raise ValueError("normal PACK5 requires N%256=0 and K%64=0")
    if scale.shape not in ((e, n), (e, n, 1)) or scale.device != weight.device:
        raise ValueError("per-channel scale must be [E,N] or [E,N,1] on weight device")
    # Same permutation as flatten_pack5_weight_asm_normal, with K16 compressed
    # to eight bytes. Adjacent K elements stay adjacent, so unpack commutes with
    # this permutation. No full INT8/FP weight copy is made at model load time.
    packed = weight.reshape(e, n // 256, 16, 16, kh // 32, 4, 8)
    packed = packed.permute(0, 4, 1, 2, 5, 3, 6).contiguous().reshape(e, n * kh)
    scales = scale.reshape(e, n).float().mul(scale_multiplier).contiguous()
    return {"normal_int4": (packed, scales)}


def transform_int4_weights_for_mega_moe_normal(
    l1_weights, l2_weights, *, l1_scale, l2_scale, scale_multiplier=16.0
):
    """Prepare raw SlimQuant checkpoint weights BEFORE any AITER/Marlin shuffle.

    Checkpoint byte = (even_K & 15)<<4 | (odd_K & 15). Nibbles are signed
    two's-complement, not offset-8. This model stores true channel scale / 16.
    Callers with true scales must explicitly use scale_multiplier=1.
    L1 rows are [all gate rows, all up rows], matching the existing INT8 K2.
    """
    if not math.isfinite(scale_multiplier) or scale_multiplier <= 0:
        raise ValueError("scale_multiplier must be finite and positive")
    if (l1_weights.ndim != 3 or l2_weights.ndim != 3
            or l1_weights.shape[0] != l2_weights.shape[0]
            or l1_weights.shape[1] != 4 * l2_weights.shape[2]
            or l2_weights.shape[1] != 2 * l1_weights.shape[2]):
        raise ValueError("expected L1=[E,2I,H/2], L2=[E,H,I/2]")
    return (_pack(l1_weights, l1_scale, scale_multiplier),
            _pack(l2_weights, l2_scale, scale_multiplier))


def int4_w4a8_mega_moe(y, l1_weights, l2_weights, sym_buffer, **kwargs):
    """W4A8 compatibility path; uses an INT8 activation SymmBuffer, eager only.

    Expanded storage is shared across layers on the same stream/buffer. A
    DSV4-Flash EP8 rank retains 384 MiB INT4 per layer and reuses 768 MiB total
    INT8 scratch, instead of expanding all layers persistently.
    """
    import megamoe

    if kwargs.get("graph", False) or torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("W4A8 scratch expansion supports eager normal only")
    if kwargs.get("megamoe_backend", "normal") != "normal":
        raise NotImplementedError("W4A8 supports normal backend only")
    if getattr(sym_buffer, "quant_mode", None) != "int8":
        raise ValueError("W4A8 requires SymmBuffer quant_mode='int8'")
    p1, s1 = l1_weights["normal_int4"]
    p2, s2 = l2_weights["normal_int4"]
    stream = torch.cuda.current_stream(y.device)
    key = (y.device, stream.cuda_stream, tuple(p1.shape), tuple(p2.shape))
    cache = getattr(sym_buffer, "_int4_unpack_workspaces", None)
    if cache is None:
        cache = sym_buffer._int4_unpack_workspaces = {}
    if key not in cache:
        cache[key] = tuple(torch.empty((p.shape[0], p.shape[1] * 2),
                                      dtype=torch.int8, device=p.device)
                           for p in (p1, p2))
    w1, w2 = cache[key]
    unpack_signed_int4_out(p1, w1)
    unpack_signed_int4_out(p2, w2)
    return megamoe.int8_mega_moe(
        y,
        {"normal": (w1, s1)},
        {"normal": (w2, s2)},
        sym_buffer,
        **kwargs,
    )
