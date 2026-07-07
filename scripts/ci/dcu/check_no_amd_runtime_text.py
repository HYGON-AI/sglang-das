#!/usr/bin/env python3
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail if HCU-owned changed files add user-visible legacy platform text.

The workflow passes only changed files from the PR or push diff. This script
does not scan the whole repository.
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


HCU_PATH_PREFIXES = (
    ".github/workflows/",
    "scripts/ci/dcu/",
    "test/registered/dcu/",
)
HCU_EXACT_FILES = {
    "python/sglang/test/dcu_utils.py",
    "requirements_dcu.txt",
}
EXEMPT_FILES = {
    "scripts/ci/dcu/check_no_amd_runtime_text.py",
}
VISIBLE_CALLS = {
    "print",
    "warning",
    "warn",
    "error",
    "exception",
    "critical",
    "info",
    "debug",
    "skip",
    "skipif",
    "xfail",
    "add_argument",
    "RuntimeError",
    "ValueError",
    "AssertionError",
    "Exception",
    "ImportError",
    "NotImplementedError",
}
TEXT_EXTS = {".py", ".sh", ".bash", ".yml", ".yaml"}
VISIBLE_TEXT_RE = re.compile(
    r"^\s*(?:-?\s*)?(?:name|description):|"
    r"\b(?:echo|printf)\b|"
    r"::(?:error|warning|notice)\b",
    re.IGNORECASE,
)
BLOCKED_TEXT_RE = re.compile(r"AMD|amd|XGMI|xgmi|DCU|dcu")
MAX_SNIPPET_LEN = 140


@dataclass(frozen=True)
class Replacement:
    source: str
    target: str


REPLACEMENTS = (
    Replacement("AMD/ROCm", "HCU/ROCm"),
    Replacement("AMD/HIP", "HCU/HIP"),
    Replacement("AMD GPUs", "HCU devices"),
    Replacement("AMD GPU", "HCU device"),
    Replacement("AMD", "HCU"),
    Replacement("amd", "hcu"),
    Replacement("XGMI", "HSL"),
    Replacement("xgmi", "hsl"),
    Replacement("DCU", "HCU"),
    Replacement("dcu", "hcu"),
)


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    location: str
    text: str

    def display_path(self) -> str:
        return normalize_path(str(self.path))

    def message(self) -> str:
        return (
            f"{self.display_path()}:{self.lineno}: user-visible legacy platform "
            f"text in {self.location}: {text_snippet(self.text)}"
        )


def normalize_path(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def is_hcu_owned(path: str) -> bool:
    path = normalize_path(path)
    if path in HCU_EXACT_FILES:
        return True
    if path.startswith(".github/workflows/"):
        return any(token in Path(path).name.lower() for token in ("dcu", "hcu"))
    return any(path.startswith(prefix) for prefix in HCU_PATH_PREFIXES[1:])


def function_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def string_values(node: ast.AST):
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child.lineno, child.value


def text_snippet(text: str) -> str:
    snippet = " ".join(text.split())
    if len(snippet) > MAX_SNIPPET_LEN:
        snippet = snippet[: MAX_SNIPPET_LEN - 3] + "..."
    return repr(snippet)


def suggested_text(text: str) -> str:
    result = text
    for replacement in REPLACEMENTS:
        result = result.replace(replacement.source, replacement.target)
    return result


def replacement_hint(text: str) -> str:
    hits: list[str] = []
    for replacement in REPLACEMENTS:
        if replacement.source in text:
            hits.append(f"{replacement.source} -> {replacement.target}")
    if not hits:
        return "Apply HCU wording: AMD->HCU, XGMI->HSL, DCU->HCU; keep ROCm/HIP."
    return "; ".join(hits)


def escape_annotation_message(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_annotation_property(text: str) -> str:
    return escape_annotation_message(text).replace(":", "%3A").replace(",", "%2C")


def emit_github_error(error: Violation) -> None:
    message = (
        "HCU-owned user-visible output contains legacy AMD/XGMI/DCU wording. "
        f"Suggested mapping: {replacement_hint(error.text)}. "
        f"Suggested text: {text_snippet(suggested_text(error.text))}"
    )
    print(
        "::error "
        f"file={escape_annotation_property(error.display_path())},"
        f"line={error.lineno},"
        "title=HCU runtime text check::"
        f"{escape_annotation_message(message)}"
    )


def violation(path: Path, lineno: int, location: str, text: str) -> Violation:
    return Violation(path=path, lineno=lineno, location=location, text=text)


def has_blocked_text(text: str) -> bool:
    return BLOCKED_TEXT_RE.search(text) is not None


def check_python(path: Path) -> list[Violation]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [violation(path, 1, "Python parser", str(exc))]

    errors: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = function_name(node.func)
            if name not in VISIBLE_CALLS:
                continue
            for lineno, value in string_values(node):
                if has_blocked_text(value):
                    errors.append(violation(path, lineno, f"{name}()", value))
        elif isinstance(node, ast.Raise) and node.exc is not None:
            for lineno, value in string_values(node.exc):
                if has_blocked_text(value):
                    errors.append(violation(path, lineno, "raise", value))
    return errors


def check_text(path: Path) -> list[Violation]:
    errors: list[Violation] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []

    for lineno, line in enumerate(lines, start=1):
        if VISIBLE_TEXT_RE.search(line) and has_blocked_text(line):
            errors.append(violation(path, lineno, "visible text", line.strip()))
    return errors


def main(argv: list[str]) -> int:
    paths = [Path(normalize_path(arg)) for arg in argv if arg.strip()]
    errors: list[Violation] = []

    for path in paths:
        normalized = normalize_path(str(path))
        if normalized in EXEMPT_FILES:
            continue
        if not is_hcu_owned(normalized):
            continue
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_EXTS and normalized not in HCU_EXACT_FILES:
            continue

        if path.suffix.lower() == ".py":
            errors.extend(check_python(path))
        else:
            errors.extend(check_text(path))

    if errors:
        print("HCU runtime text check failed.")
        print()
        print(
            "Detected user-visible AMD/amd, XGMI/xgmi, or DCU/dcu text in "
            "HCU-owned changed files. This blocks PR merge because HCU runtime "
            "logs and CI output should use HCU wording."
        )
        print()
        print("Scope:")
        print(" - Only changed files passed by quality-gate changed-files are checked.")
        print(" - Python checks only user-visible strings in print/logger/raise/skip/help.")
        print(" - Shell/YAML checks only visible output fields such as echo/printf/name/description.")
        print()
        print("How to fix:")
        print(" 1. Open each file:line reported below.")
        print(" 2. Apply these mappings only to user-visible output:")
        print("    - AMD / AMD GPU(s) -> HCU / HCU device(s)")
        print("    - amd -> hcu")
        print("    - XGMI / xgmi -> HSL / hsl")
        print("    - DCU / dcu -> HCU / hcu")
        print("    - AMD/ROCm -> HCU/ROCm")
        print("    - AMD/HIP -> HCU/HIP")
        print(" 3. Keep ROCm and HIP unchanged.")
        print(" 4. Do not rename files or change variable/function names just for this check.")
        print()
        print("Violations:")
        for error in errors:
            print(f" - {error.message()}")
            print(f"   Suggested: {text_snippet(suggested_text(error.text))}")
            emit_github_error(error)
        return 1

    print("HCU runtime text check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
