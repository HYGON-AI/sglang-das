"""Dry-run verifier for the DCU CI registration.

This script does **not** execute any test. It only:
  1. globs test/registered/**/*.py
  2. parses each file with the AST-based collector
  3. filters by HWBackend.DCU + DCU suite list
  4. prints a summary so we can confirm DCU tests are wired in correctly

Usage (from the repository root):
    python scripts/ci/dcu/verify_dcu_registration.py

Exit code 0 means the DCU registration is structurally healthy:
    - At least one DCU test was collected.
    - Every DCU test belongs to a known DCU per-commit or nightly suite.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _sanitize_dcu_log_text(text: str) -> str:
    return text.replace("AMD", "DCU").replace("amd", "dcu")


def _load_ci_register():
    """Load ci_register.py directly, bypassing sglang/__init__.py.

    This avoids pulling in the full sglang runtime (tqdm, torch, ...) just
    to run a static AST-based check.
    """
    path = os.path.join(
        REPO_ROOT, "python", "sglang", "test", "ci", "ci_register.py"
    )
    spec = importlib.util.spec_from_file_location("dcu_ci_register", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ci_register from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ci_register = _load_ci_register()
    HWBackend = ci_register.HWBackend
    collect_tests = ci_register.collect_tests

    # Import suite lists from run_suite.py so we always check against the
    # source of truth instead of duplicating the suite names here.
    test_dir = os.path.join(REPO_ROOT, "test")
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)

    # `run_suite` imports `sglang.test.ci.ci_utils` which pulls heavy deps.
    # We only need the suite dicts, so re-define them via a small parser
    # rather than importing run_suite directly.
    import ast

    with open(os.path.join(test_dir, "run_suite.py"), "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    per_commit: dict[str, list[str]] = {}
    nightly: dict[str, list[str]] = {}

    def _extract_dict(node: ast.Dict) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for k, v in zip(node.keys, node.values):
            # key is HWBackend.NAME
            if not isinstance(k, ast.Attribute) or not isinstance(k.value, ast.Name):
                continue
            if k.value.id != "HWBackend":
                continue
            backend_name = k.attr
            if not isinstance(v, ast.List):
                continue
            suites = [elt.value for elt in v.elts if isinstance(elt, ast.Constant)]
            out[backend_name] = suites
        return out

    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not stmt.targets or not isinstance(stmt.targets[0], ast.Name):
            continue
        name = stmt.targets[0].id
        if name == "PER_COMMIT_SUITES" and isinstance(stmt.value, ast.Dict):
            per_commit = _extract_dict(stmt.value)
        elif name == "NIGHTLY_SUITES" and isinstance(stmt.value, ast.Dict):
            nightly = _extract_dict(stmt.value)

    dcu_per_commit = set(per_commit.get("DCU", []))
    dcu_nightly = set(nightly.get("DCU", []))

    print("Configured DCU suites:")
    print(f"  per-commit ({len(dcu_per_commit)}):")
    for s in sorted(dcu_per_commit):
        print(f"    - {s}")
    print(f"  nightly    ({len(dcu_nightly)}):")
    for s in sorted(dcu_nightly):
        print(f"    - {s}")
    print()

    if not dcu_per_commit and not dcu_nightly:
        print("ERROR: No DCU suites configured in run_suite.py.", file=sys.stderr)
        return 2

    registered_dir = os.path.join(test_dir, "registered")
    files = [
        f
        for f in glob.glob(
            os.path.join(registered_dir, "**", "*.py"), recursive=True
        )
        if not f.endswith(("conftest.py", "__init__.py"))
    ]

    tests = collect_tests(files, sanity_check=False)
    dcu_tests = [t for t in tests if t.backend == HWBackend.DCU]

    print(f"Collected {len(dcu_tests)} DCU registered test file(s):")
    by_suite: dict[str, list[str]] = defaultdict(list)
    for t in dcu_tests:
        by_suite[t.suite].append(
            f"{os.path.relpath(t.filename, REPO_ROOT)} "
            f"(est={t.est_time}s, nightly={t.nightly}, disabled={t.disabled})"
        )

    for suite in sorted(by_suite):
        print(f"  [{suite}]")
        for line in by_suite[suite]:
            print(_sanitize_dcu_log_text(f"    - {line}"))
    print()

    unknown_suites = [
        s for s in by_suite if s not in dcu_per_commit and s not in dcu_nightly
    ]
    if unknown_suites:
        print(
            "ERROR: DCU tests reference suites that are not declared in "
            "run_suite.py:",
            file=sys.stderr,
        )
        for s in unknown_suites:
            print(f"  - {s}", file=sys.stderr)
        return 3

    if not dcu_tests:
        print(
            "ERROR: No DCU tests collected. Did you forget to add "
            "register_dcu_ci() to a test file under test/registered/dcu/?",
            file=sys.stderr,
        )
        return 4

    print("OK: DCU registration looks healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
