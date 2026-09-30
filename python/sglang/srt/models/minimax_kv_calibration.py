"""Opt-in real-activation recorder for MiniMax-M3 FP8 attention calibration.

The recorder is deliberately dormant unless
``SGLANG_MINIMAX_KV_CALIBRATION_DIR`` is set.  It observes tensors after Q/K
normalization and RoPE but before the KV-cache write, which is the exact
quantization boundary used by the attention backends.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path

import torch

_LOCK = threading.Lock()
_LAYERS: dict[int, dict] = {}


def _rank_label() -> str:
    for name in ("RANK", "LOCAL_RANK", "SGLANG_RANK", "LOCAL_WORLD_SIZE"):
        value = os.environ.get(name)
        if value is not None:
            return f"{name.lower()}-{value}"
    return "rank-unknown"


def _amax(tensor: torch.Tensor) -> float:
    # Keep the reduction in the source dtype.  Materializing a float32 copy of
    # a 128K Q tensor can consume several GiB and is unnecessary for absmax.
    value = tensor.detach().abs().amax().item()
    if not (value >= 0.0 and value < float("inf")):
        raise RuntimeError(f"non-finite MiniMax KV calibration amax: {value}")
    return float(value)


def _flush(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    destination = (
        output_dir / f"kv-stats-{socket.gethostname()}-{_rank_label()}-pid{pid}.json"
    )
    temporary = destination.with_suffix(f".tmp-{threading.get_ident()}")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": pid,
        "rank_env": {
            name: os.environ.get(name)
            for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "SGLANG_RANK")
            if os.environ.get(name) is not None
        },
        "quantization_boundary": "post_qk_norm_rope_pre_kv_cache_write",
        "layers": {str(key): value for key, value in sorted(_LAYERS.items())},
    }
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def record_minimax_kv_activations(
    layer_id: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_mode: object,
) -> None:
    """Record one real forward observation and atomically persist it.

    This is calibration-only instrumentation.  The small host synchronization
    and JSON write are acceptable here and never occur when the env is unset.
    """

    output = os.environ.get("SGLANG_MINIMAX_KV_CALIBRATION_DIR", "").strip()
    if not output:
        return
    max_observations = int(
        os.environ.get("SGLANG_MINIMAX_KV_CALIBRATION_MAX_OBSERVATIONS_PER_LAYER", "16")
    )
    if max_observations <= 0 or q.numel() == 0 or k.numel() == 0 or v.numel() == 0:
        return

    with _LOCK:
        entry = _LAYERS.setdefault(
            int(layer_id),
            {
                "observations": 0,
                "tokens": 0,
                "q_amax": 0.0,
                "k_amax": 0.0,
                "v_amax": 0.0,
                "modes": {},
            },
        )
        if entry["observations"] >= max_observations:
            return

        q_amax, k_amax, v_amax = _amax(q), _amax(k), _amax(v)
        mode = getattr(forward_mode, "name", None) or str(forward_mode)
        entry["observations"] += 1
        entry["tokens"] += int(k.shape[0])
        entry["q_amax"] = max(entry["q_amax"], q_amax)
        entry["k_amax"] = max(entry["k_amax"], k_amax)
        entry["v_amax"] = max(entry["v_amax"], v_amax)
        entry["modes"][mode] = entry["modes"].get(mode, 0) + 1
        # Flush once per full model pass rather than once per layer.  MiniMax-M3
        # has 60 layers and calibration intentionally synchronizes amax values,
        # so 60 JSON rewrites per token would otherwise dominate decode.
        flush_layer = int(
            os.environ.get("SGLANG_MINIMAX_KV_CALIBRATION_FLUSH_LAYER", "59")
        )
        if int(layer_id) == flush_layer:
            _flush(Path(output))
