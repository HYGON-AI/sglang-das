#!/usr/bin/env python3
"""Reject unsafe or incomplete DCU-to-HCU mechanical replacements.

Author: Codex
"""

import re
import subprocess
import sys
from pathlib import Path


ALLOWLIST_PATH = Path("scripts/ci/hcu/hcu_rename_allowlist.txt")
SELF_EXEMPT_FILES = {
    "scripts/ci/hcu/check_hcu_runtime_text.py",
    "scripts/ci/hcu/hcu_rename_allowlist.txt",
    "scripts/ci/hcu/test_verify_hcu_rename.py",
    "scripts/ci/hcu/verify_hcu_rename.py",
}
FORBIDDEN_COMPILER_FLAGS = ("-HCUDA", "-HCUTLASS", "-HCUTE")
FORBIDDEN_PLACEHOLDERS = ("@@HCU_SYNC_",)
ALLOWED_COMPILER_DEFINES = ("-DCUTLASS", "-DCUTE", "-DCUDA_ENABLED")
ALLOWED_UPSTREAM_TEXT = ("voidcutlassdevicekernelflash",)

UNIQUE_TOP_LEVEL_FUNCTIONS = {
    "python/sglang/srt/utils/common.py": ("is_hcu", "is_hcu_native_fp8_supported"),
}
UNIQUE_IMPORTED_NAMES = {
    "python/sglang/srt/server_args.py": ("is_hcu",),
}


def normalize_path(path):
    path = str(path).replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def has_old_name(text):
    return "DCU" in text or "dcu" in text


def validate_path(path):
    path = normalize_path(path)
    if path in SELF_EXEMPT_FILES:
        return []
    if has_old_name(path):
        return ["{}: old DCU/dcu name remains in tracked path".format(path)]
    return []


def parse_allowlist(path):
    entries = {}
    if not path.exists():
        return entries
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        file_name, separator, allowed_text = line.partition(":")
        if not separator or not file_name or not allowed_text:
            raise ValueError("invalid HCU rename allowlist entry: {}".format(raw_line))
        entries.setdefault(normalize_path(file_name), []).append(allowed_text)
    return entries


def strip_allowed_technical_text(line):
    remainder = line
    for token in ALLOWED_COMPILER_DEFINES + ALLOWED_UPSTREAM_TEXT:
        remainder = remainder.replace(token, "")
    return remainder


def validate_text_line(path, lineno, line, allowlist):
    path = normalize_path(path)
    if path in SELF_EXEMPT_FILES:
        return []

    violations = []
    for flag in FORBIDDEN_COMPILER_FLAGS:
        if flag in line:
            violations.append(
                "{}:{}: invalid compiler flag {}".format(path, lineno, flag)
            )
    for placeholder in FORBIDDEN_PLACEHOLDERS:
        if placeholder in line:
            violations.append(
                "{}:{}: leaked migration placeholder {}".format(
                    path, lineno, placeholder
                )
             )

    if not has_old_name(line):
        return violations
    if not has_old_name(strip_allowed_technical_text(line)):
        return violations
    if any(token in line for token in allowlist.get(path, [])):
        return violations

    violations.append(
        "{}:{}: unexpected DCU/dcu text remains: {}".format(
            path, lineno, line.strip()[:180]
        )
    )
    return violations


def validate_python_structure(path, text):
    path = normalize_path(path)
    expected_functions = UNIQUE_TOP_LEVEL_FUNCTIONS.get(path, ())
    expected_imports = UNIQUE_IMPORTED_NAMES.get(path, ())
    if not expected_functions and not expected_imports:
        return []

    violations = []
    if expected_functions:
        top_level_names = re.findall(
            r"^(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", text, re.MULTILINE
        )
        for name in expected_functions:
            count = top_level_names.count(name)
            if count != 1:
                violations.append(
                    "{}: expected exactly one top-level {}, found {}".format(
                        path, name, count
                    )
                )

    if expected_imports:
        imported_names = []
        import_blocks = re.findall(
            r"^from\s+[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s+import\s*\((.*?)^\)",
            text,
            re.MULTILINE | re.DOTALL,
        )
        for block in import_blocks:
            for name, alias in re.findall(
                r"^\s*([A-Za-z_]\w*)(?:\s+as\s+([A-Za-z_]\w*))?\s*,?\s*(?:#.*)?$",
                block,
                re.MULTILINE,
            ):
                imported_names.append(alias or name)
        for name in expected_imports:
            count = imported_names.count(name)
            if count != 1:
                violations.append(
                    "{}: expected exactly one imported {}, found {}".format(
                        path, name, count
                    )
                )
    return violations


def tracked_files(root):
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files"], universal_newlines=True
    )
    return [line for line in output.splitlines() if line]


def find_violations(root):
    root = Path(root).resolve()
    allowlist = parse_allowlist(root / ALLOWLIST_PATH)
    violations = []
    for relative_name in tracked_files(root):
        violations.extend(validate_path(relative_name))
        path = root / relative_name
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        violations.extend(validate_python_structure(relative_name, text))
        for lineno, line in enumerate(text.splitlines(), 1):
            violations.extend(
                validate_text_line(relative_name, lineno, line, allowlist)
            )
    return violations


def main():
    root = Path(__file__).resolve().parents[3]
    violations = find_violations(root)
    if violations:
        print("HCU rename safety check failed:", file=sys.stderr)
        for item in violations:
            print("  {}".format(item), file=sys.stderr)
        return 1
    print("HCU rename safety check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
