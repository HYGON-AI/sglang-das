"""Per-token activation quantization and routing copies into owned MoE buffers."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _quantize_and_stage(
    X,
    IDS,
    WEIGHTS,
    Q,
    SCALE,
    OUT_IDS,
    OUT_WEIGHTS,
    X_ROW: tl.constexpr,
    Q_ROW: tl.constexpr,
    SCALE_ROW: tl.constexpr,
    IDS_ROW: tl.constexpr,
    IDS_COL: tl.constexpr,
    WEIGHTS_ROW: tl.constexpr,
    WEIGHTS_COL: tl.constexpr,
    OUT_IDS_ROW: tl.constexpr,
    OUT_WEIGHTS_ROW: tl.constexpr,
    COLUMNS: tl.constexpr,
    TOPK: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    IS_INT8: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    values = tl.load(X + row * X_ROW + cols, cols < COLUMNS, 0).to(tl.float32)
    maximum = tl.max(tl.abs(values), 0)
    if IS_INT8:
        # Match LightOp's per-token INT8 reference, including RNE and epsilon.
        maximum = tl.maximum(maximum, 1e-10)
        scale = maximum / QUANT_MAX
        quantized = libdevice.nearbyint(values * (QUANT_MAX / maximum))
    else:
        # LightOp FP8 uses a zero scale for an all-zero row.
        scale = maximum / QUANT_MAX
        inverse = tl.where(maximum == 0, 0.0, QUANT_MAX / maximum)
        quantized = values * inverse
    tl.store(Q + row * Q_ROW + cols, quantized, cols < COLUMNS)
    tl.store(SCALE + row * SCALE_ROW, scale)
    choices = tl.arange(0, ROUTE_BLOCK)
    ids = tl.load(IDS + row * IDS_ROW + choices * IDS_COL, choices < TOPK, 0)
    weights = tl.load(
        WEIGHTS + row * WEIGHTS_ROW + choices * WEIGHTS_COL, choices < TOPK, 0
    )
    tl.store(OUT_IDS + row * OUT_IDS_ROW + choices, ids, choices < TOPK)
    tl.store(OUT_WEIGHTS + row * OUT_WEIGHTS_ROW + choices, weights, choices < TOPK)


def _span(tensor, rows):
    elements = (rows - 1) * tensor.stride(0) + 1
    if tensor.ndim == 2:
        elements += (tensor.shape[1] - 1) * tensor.stride(1)
    begin = tensor.data_ptr()
    return begin, begin + elements * tensor.element_size()


def try_quantize_and_stage_moe_inputs(
    x, topk_ids, topk_weights, out_quantized, out_scale, out_ids, out_weights
) -> bool:
    """Stage one token per row; return False without writes for unsupported layouts.

    The output dtype selects INT8 or FP8 per-token quantization. All outputs
    must be independent of the inputs and of each other. The caller retains
    its existing quantization path when this entry returns False.
    """
    if x.ndim != 2:
        return False
    rows, columns = x.shape
    tensors = (
        x,
        topk_ids,
        topk_weights,
        out_quantized,
        out_scale,
        out_ids,
        out_weights,
    )
    if not all(t.is_cuda and t.device == x.device for t in tensors):
        return False
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    quant_dtype = out_quantized.dtype
    if quant_dtype == torch.int8:
        quant_max = torch.iinfo(quant_dtype).max
    elif quant_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        quant_max = torch.finfo(quant_dtype).max
    else:
        return False
    if not (0 < columns <= 16384 and x.stride(1) == 1 and x.stride(0) >= columns):
        return False
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        return False
    topk = topk_ids.shape[1]
    if not (0 < topk <= 1024 and topk_weights.shape[1] == topk):
        return False
    if topk_ids.dtype not in (torch.int32, torch.int64) or out_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return False
    if any(
        t.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        for t in (topk_weights, out_weights)
    ):
        return False
    for tensor, width in (
        (topk_ids, topk),
        (topk_weights, topk),
        (out_quantized, columns),
        (out_ids, topk),
        (out_weights, topk),
    ):
        if (
            tensor.ndim != 2
            or tensor.shape[0] < rows
            or tensor.shape[1] != width
            or min(tensor.stride()) <= 0
        ):
            return False
    for tensor, width in (
        (out_quantized, columns),
        (out_ids, topk),
        (out_weights, topk),
    ):
        if tensor.stride(1) != 1 or tensor.stride(0) < width:
            return False
    if (
        out_scale.ndim != 1
        or out_scale.shape[0] < rows
        or out_scale.dtype != torch.float32
        or out_scale.stride(0) <= 0
    ):
        return False
    if not rows:
        return True
    spans = [_span(t, rows) for t in tensors[:3]]
    for tensor in tensors[3:]:
        begin, end = _span(tensor, rows)
        if any(
            begin < other_end and other_begin < end for other_begin, other_end in spans
        ):
            return False
        spans.append((begin, end))
    block = triton.next_power_of_2(columns)
    with torch.cuda.device(x.device):
        _quantize_and_stage[(rows,)](
            *tensors,
            x.stride(0),
            out_quantized.stride(0),
            out_scale.stride(0),
            topk_ids.stride(0),
            topk_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            out_ids.stride(0),
            out_weights.stride(0),
            columns,
            topk,
            quant_max,
            quant_dtype == torch.int8,
            block,
            triton.next_power_of_2(topk),
            num_warps=min(max(block // 256, 1), 8),
            num_stages=1,
        )
    return True
