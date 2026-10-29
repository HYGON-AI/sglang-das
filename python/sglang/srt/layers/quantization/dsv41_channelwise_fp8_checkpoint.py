"""Metadata-only eligibility check for automatic DSV4.1 FP8 projections."""

import json
import struct
from functools import lru_cache
from pathlib import Path


def _projection_role(name):
    if name.endswith(".wo_a.weight"):
        return "wo_a"
    if name in ("head.weight", "lm_head.weight"):
        return "lm_head"
    if name.endswith(".markov_head.head.weight"):
        return "markov_head"
    return None


def _read_header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= 100 * 1024**2:
            raise ValueError("Invalid safetensors header length")
        return json.loads(stream.read(size))


@lru_cache(maxsize=8)
def checkpoint_supports_channelwise_fp8(model_path):
    """Require all three projections and row scales; never read tensor payloads.

    This intentionally does not guess from a model's directory name or its
    quantization config: the ordinary FP8 model has BF16 special projections.
    Nonlocal/unavailable checkpoints retain the previous behavior; explicit
    runtime=1 remains available for custom/remote loaders.
    """
    if not model_path:
        return False
    root = Path(model_path).resolve()
    if not root.is_dir():
        return False
    try:
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            if not isinstance(weight_map, dict):
                return False
            selected = {name for name in weight_map if _projection_role(name)}
            if {_projection_role(name) for name in selected} != {
                "wo_a",
                "lm_head",
                "markov_head",
            }:
                return False
            files = {weight_map[name] for name in selected}
            for name in selected:
                base = name[: -len(".weight")]
                for suffix in (".scale", ".weight_scale_inv", ".weight_scale"):
                    if base + suffix in weight_map:
                        files.add(weight_map[base + suffix])
            paths = [root / name for name in sorted(files)]
        else:
            paths = sorted(root.glob("*.safetensors"))
        headers = {}
        for path in paths:
            try:
                path.resolve().relative_to(root)
            except ValueError:
                return False
            headers.update(_read_header(path))
        projections = {
            name: meta for name, meta in headers.items() if _projection_role(name)
        }
        if {_projection_role(name) for name in projections} != {
            "wo_a",
            "lm_head",
            "markov_head",
        }:
            return False
        for name, meta in projections.items():
            if not isinstance(meta, dict):
                return False
            shape = meta.get("shape", [])
            if meta.get("dtype") not in ("F8_E4M3", "F8_E4M3FN") or len(shape) != 2:
                return False
            base = name[: -len(".weight")]
            scales = [
                headers.get(base + suffix)
                for suffix in (".scale", ".weight_scale_inv", ".weight_scale")
            ]
            if not any(
                isinstance(scale, dict)
                and scale.get("shape") == [shape[0], 1]
                and scale.get("dtype") in ("F16", "BF16", "F32", "F64")
                for scale in scales
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return False
