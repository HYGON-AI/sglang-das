#!/usr/bin/env python3
"""Fail fast when required HCU external APIs are absent from the test image."""

import importlib
import sys
from typing import List


MODULE_SYMBOLS = (
    ("causal_conv1d", "causal_conv1d_fn_hcu"),
    ("lightop", "gemma_fused_add_rmsnorm"),
    ("sgl_kernel", None),
)


def check_module_symbols() -> List[str]:
    errors = []  # type: List[str]
    for module_name, symbol in MODULE_SYMBOLS:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"cannot import {module_name}: {exc}")
            continue
        if symbol and not hasattr(module, symbol):
            errors.append(f"{module_name}.{symbol} is missing")
    return errors


def check_registered_op() -> List[str]:
    try:
        import torch
        import sglang.srt.layers.moe.topk  # noqa: F401 - registers the custom op
    except Exception as exc:
        return [f"cannot load HCU MoE op registration: {exc}"]

    if not hasattr(torch.ops.sglang, "moe_fused_gate_hcu"):
        return ["torch.ops.sglang.moe_fused_gate_hcu is missing"]
    return []


def main() -> int:
    errors = check_module_symbols() + check_registered_op()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: required HCU external APIs and custom op are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
