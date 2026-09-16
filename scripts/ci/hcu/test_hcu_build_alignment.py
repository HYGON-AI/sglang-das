# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual workflow scripts without runners, credentials or HCU."""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github/workflows"


def workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def step(job, name):
    return next(s for s in job["steps"] if s.get("name") == name)


def heredoc(script, marker):
    return script.split("\n", 1)[1].split("\n" + marker, 1)[0]


def evaluate(expression, base, ref, override="", target=""):
    code = expression[3:-2].strip()
    for key, value in {
        "github.event.pull_request.base.ref": base,
        "github.ref_name": ref,
        "inputs.torch_version": override,
        "inputs.target_branch": target,
        "vars.HCU_CI_IMAGE_0518": "image-211",
        "vars.HCU_CI_RELEASE_IMAGE": "image-210",
    }.items():
        code = code.replace(key, repr(value))
    return eval(code.replace("&&", " and ").replace("||", " or "), {"__builtins__": {}})


class TestBuildAlignment(unittest.TestCase):
    def test_temporary_branch_manual_target(self):
        build = workflow("release-pr-hcu.yml")["jobs"]["compile"]
        for target in ["main", "release/20260825_v0.5.18"]:
            self.assertEqual(
                evaluate(
                    build["container"]["image"], "", "hcu/temporary", target=target
                ),
                "image-211",
            )
            self.assertEqual(
                evaluate(
                    build["env"]["TORCH_VERSION"], "", "hcu/temporary", target=target
                ),
                "2.11.0",
            )
            self.assertEqual(
                evaluate(
                    build["env"]["SGLANG_PACKAGE_VERSION"],
                    "",
                    "hcu/temporary",
                    target=target,
                ),
                "0.5.18",
            )
        self.assertEqual(
            evaluate(
                build["env"]["TORCH_VERSION"], "v0.5.12_dev", "feature", target="main"
            ),
            "2.10.0",
        )
        guard = step(build, "Validate manual build target")["run"]
        for target, expected in [
            ("hcu/temporary", 1),
            ("main", 0),
            ("release/20260825_v0.5.18", 0),
        ]:
            result = subprocess.run(
                ["bash", "-c", guard],
                env={**os.environ, "BUILD_TARGET": target},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, expected)

    def test_pr_environment_routing(self):
        build = workflow("release-pr-hcu.yml")["jobs"]["compile"]
        wait = workflow("pr-test-hcu.yml")["jobs"]["wait-hcu-wheels"]
        release = "release/20260825_v0.5.18"
        for base, ref, expected in [
            ("main", "feature", "2.11.0"),
            (release, "main", "2.11.0"),
            ("", "main", "2.11.0"),
            ("", release, "2.11.0"),
            ("v0.5.12_dev", release, "2.10.0"),
        ]:
            with self.subTest(base=base, ref=ref):
                self.assertEqual(
                    evaluate(build["env"]["TORCH_VERSION"], base, ref), expected
                )
                self.assertEqual(
                    evaluate(build["container"]["image"], base, ref),
                    "image-211" if expected == "2.11.0" else "image-210",
                )
                self.assertEqual(
                    evaluate(
                        wait["env"]["HCU_WHEEL_EXPECTED_TORCH_VERSION"], base, ref
                    ),
                    "2.11.0" if expected == "2.11.0" else "",
                )
        self.assertEqual(
            evaluate(build["env"]["TORCH_VERSION"], "", release, "2.12.0"), "2.12.0"
        )

    def test_rust_trigger_and_classification(self):
        for filename in ["release-pr-hcu.yml", "pr-test-hcu.yml"]:
            data = workflow(filename)
            # PyYAML's YAML 1.1 loader parses the `on` key as True.
            self.assertIn(
                "rust/**",
                data.get("on", data.get(True))["pull_request_target"]["paths"],
            )
        check = workflow("pr-test-hcu.yml")["jobs"]["check-changes"]
        scripts = "\n".join(s.get("run", "") for s in check["steps"])
        hcu = re.search(r"hcu_path_pattern='([^']+)'", scripts).group(1)
        wheel = next(
            p
            for p in re.findall(r"grep -Eq '([^']+)'", scripts)
            if "requirements_hcu" in p
        )
        for path in ["rust/Cargo.lock", "rust/tree/src/lib.rs", "python/setup.py"]:
            self.assertRegex(path, hcu)
            self.assertRegex(path, wheel)

    def test_build_contract(self):
        pr = workflow("release-pr-hcu.yml")["jobs"]["compile"]
        nightly = workflow("nightly-test-hcu.yml")["jobs"]["build-hcu-wheels"]
        self.assertEqual(
            nightly["container"]["image"], "${{ needs.validate-config.outputs.image }}"
        )
        self.assertEqual(nightly["env"]["TORCH_VERSION"], "2.11.0")
        for job in [pr, nightly]:
            self.assertNotIn(
                "pip install torch", step(job, "Install build deps")["run"]
            )
            step(job, "Verify image Torch version before building extensions")
            self.assertIn(
                "cp pyproject_hcu.toml pyproject.toml",
                step(job, "Build sglang python package")["run"],
            )
        package = step(nightly, "Build sglang python package")
        self.assertEqual(nightly["env"]["SGLANG_PACKAGE_VERSION"], "0.5.18")
        self.assertEqual(
            package["env"]["SETUPTOOLS_SCM_PRETEND_VERSION"],
            "${{ env.SGLANG_PACKAGE_VERSION }}",
        )
        self.assertNotIn("0.5.15.post1", package["run"])
        gateway = step(pr, "Build sgl-model-gateway")["run"]
        self.assertIn(
            '--out "${GITHUB_WORKSPACE}/sgl-model-gateway/target/wheels"', gateway
        )
        self.assertNotIn(
            "bindings/python/target/wheels",
            (WORKFLOWS / "release-pr-hcu.yml").read_text(),
        )

    def test_all_embedded_scripts_parse(self):
        for filename in [
            "release-pr-hcu.yml",
            "pr-test-hcu.yml",
            "nightly-test-hcu.yml",
        ]:
            for job in workflow(filename)["jobs"].values():
                for item in job.get("steps", []):
                    script = item.get("run", "")
                    if not script:
                        continue
                    with self.subTest(workflow=filename, step=item.get("name")):
                        shell = re.sub(
                            r"\$\{\{.*?\}\}", "test-value", script, flags=re.S
                        )
                        result = subprocess.run(
                            ["bash", "-n"], input=shell, text=True, capture_output=True
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        for match in re.finditer(
                            r"python3? - <<'([^']+)'[^\n]*\n(.*?)\n\1(?:\n|$)",
                            script,
                            re.S,
                        ):
                            compile(
                                match.group(2), f"{filename}:{item.get('name')}", "exec"
                            )

    def test_nightly_wait_rejects_wrong_torch(self):
        job = workflow("nightly-test-hcu.yml")["jobs"]["wait-hcu-wheels"]
        script = step(job, "Wait for current commit HCU wheels")["run"]
        code = heredoc(script.split("python3 - <<'PY_WAIT'", 1)[1], "PY_WAIT")
        for torch_version, package_version in [
            ("2.11.0", "0.5.18"),
            ("2.10.0", "0.5.18"),
            ("2.11.0", "0.5.15.post1"),
        ]:
            with self.subTest(
                torch=torch_version
            ), tempfile.TemporaryDirectory() as temp:
                staged = Path(temp) / "ORG_REPO/pr-1/abcdef123456"
                wheels = staged / "wheels"
                wheels.mkdir(parents=True)
                (staged / "READY").touch()
                (staged / "manifest.json").write_text(
                    json.dumps(
                        {"commit_sha": "abcdef123456", "torch_version": torch_version}
                    )
                )
                for name in [
                    f"sglang-{package_version}-py3-none-any.whl",
                    "sglang_kernel-0.4.6.whl",
                ]:
                    (wheels / name).touch()
                with patch.dict(
                    os.environ,
                    HCU_WHEEL_STAGING_ROOT=temp,
                    HCU_WHEEL_TARGET_SHA="abcdef123456",
                    GITHUB_REPOSITORY="ORG/REPO",
                    HCU_WHEEL_WAIT_TIMEOUT="1",
                ), patch("time.time", side_effect=[0, 0, 2]), patch(
                    "time.sleep"
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    with self.assertRaises(SystemExit) as result:
                        exec(code, {})
                    self.assertEqual(
                        result.exception.code,
                        (
                            0
                            if torch_version == "2.11.0" and package_version == "0.5.18"
                            else 1
                        ),
                    )

    def test_nightly_probe_rejects_bad_manifest_and_router(self):
        job = workflow("nightly-test-hcu.yml")["jobs"]["probe-hcu-wheels"]
        script = step(job, "Probe local staged wheels")["run"]
        code = heredoc(script.split("python3 - <<'PY_PROBE'", 1)[1], "PY_PROBE")
        for kind in [
            "valid",
            "wrong-sha",
            "wrong-torch",
            "wrong-version",
            "broken",
            "missing",
            "router-only",
        ]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                staged = root / "ORG_REPO/pr-1/abcdef123456"
                wheels = staged / "wheels"
                wheels.mkdir(parents=True)
                (staged / "READY").touch()
                manifest = {"commit_sha": "abcdef123456", "torch_version": "2.11.0"}
                if kind == "wrong-sha":
                    manifest["commit_sha"] = "ffffffffff"
                if kind == "wrong-torch":
                    manifest["torch_version"] = "2.10.0"
                if kind != "missing":
                    (staged / "manifest.json").write_text(
                        "{" if kind == "broken" else json.dumps(manifest)
                    )
                for name in ["sglang_kernel-0.4.6.whl", "sglang_router-0.4.0.whl"]:
                    (wheels / name).touch()
                if kind != "router-only":
                    version = "0.5.15.post1" if kind == "wrong-version" else "0.5.18"
                    (wheels / f"sglang-{version}-py3-none-any.whl").touch()
                out = io.StringIO()
                with patch.dict(
                    os.environ,
                    HCU_WHEEL_STAGING_ROOT=temp,
                    HCU_WHEEL_TARGET_SHA="abcdef123456",
                    GITHUB_REPOSITORY="ORG/REPO",
                ), contextlib.redirect_stdout(out), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    try:
                        exec(code, {})
                    except SystemExit as result:
                        self.assertEqual(result.code, 0)
                self.assertIn(
                    "wheel_available=" + ("true" if kind == "valid" else "false"),
                    out.getvalue(),
                )
                self.assertNotIn(
                    "wheel_urls=/hcu-wheel-staging/ORG_REPO/pr-1/abcdef123456/wheels/sglang_router",
                    out.getvalue(),
                )

    def test_install_guard_before_uninstall(self):
        script = (ROOT / "scripts/ci/hcu/hcu_ci_install_dependency.sh").read_text()
        self.assertLess(script.index("PY_WHEEL_ABI"), script.index("pip uninstall"))
        code = heredoc(script.split("<<'PY_WHEEL_ABI'", 1)[1], "PY_WHEEL_ABI")
        for kind in [
            "valid",
            "wrong-sha",
            "wrong-torch",
            "wrong-tag",
            "wrong-commit",
            "wrong-metadata",
        ]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                wheels = Path(temp) / "hcu-wheel-staging/pr-1/abcdef123456/wheels"
                wheels.mkdir(parents=True)
                manifest = {"commit_sha": "abcdef123456", "torch_version": "2.11.0"}
                if kind == "wrong-sha":
                    manifest["commit_sha"] = "ffffffffff"
                if kind == "wrong-torch":
                    manifest["torch_version"] = "2.10.0"
                (wheels.parent / "manifest.json").write_text(json.dumps(manifest))
                version = "0.5.18+dtk2604.torch2110.2609151124.gabcdef"
                if kind == "wrong-tag":
                    version = version.replace("torch2110", "torch2100")
                if kind == "wrong-commit":
                    version = version.replace("gabcdef", "gffffff")
                path = wheels / f"sglang-{version}-cp310-cp310-linux_x86_64.whl"
                metadata_version = "0.5.12" if kind == "wrong-metadata" else version
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(
                        "sglang.dist-info/METADATA",
                        f"Name: sglang\nVersion: {metadata_version}\n",
                    )
                with patch.dict(
                    os.environ, HCU_CI_INSTALL_WHEEL_URLS=str(path)
                ), patch.dict(
                    sys.modules,
                    torch=types.SimpleNamespace(__version__="2.11.0+dtk2604"),
                ), patch(
                    "subprocess.check_output", return_value="abcdef123456\n"
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    if kind == "valid":
                        exec(code, {})
                    else:
                        with self.assertRaises(SystemExit):
                            exec(code, {})


if __name__ == "__main__":
    unittest.main()
