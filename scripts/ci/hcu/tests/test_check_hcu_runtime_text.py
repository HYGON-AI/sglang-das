import importlib.util
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "check_hcu_runtime_text.py"
SPEC = importlib.util.spec_from_file_location("check_hcu_runtime_text", str(SCRIPT))
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class HcuRuntimeTextCheckTest(unittest.TestCase):
    def write_file(self, root, relative_path, content):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_hcu_paths_and_files_are_owned(self):
        self.assertTrue(CHECKER.is_hcu_owned_path("scripts/ci/hcu/example.sh"))
        self.assertTrue(CHECKER.is_hcu_owned_path("test/registered/hcu/test_x.py"))
        self.assertTrue(CHECKER.is_hcu_owned_path(".github/workflows/pr-test-hcu.yml"))
        self.assertTrue(CHECKER.is_hcu_owned_path("python/sglang/test/hcu_utils.py"))
        self.assertTrue(CHECKER.is_hcu_owned_path("requirements_hcu.txt"))
        self.assertTrue(CHECKER.has_hcu_code_marker("if is_hcu():"))
        self.assertTrue(CHECKER.has_hcu_code_marker("HWBackend.HCU"))

    def test_comment_and_identifier_are_not_reported(self):
        source = "# AMD is a comment\namd_variable = 1\n"
        self.assertEqual(CHECKER.check_python(Path("sample.py"), source), [])

    def test_hcu_branch_violation_is_reported(self):
        source = '_is_hcu = True\nif _is_hcu:\n    print("AMD device")\n'
        errors = CHECKER.check_python(Path("sample.py"), source)
        self.assertEqual(len(errors), 1)
        self.assertIn("AMD device", errors[0].text)

    def test_explicit_non_hcu_branch_is_ignored(self):
        source = '_is_hcu = False\nif not _is_hcu:\n    print("AMD device")\n'
        self.assertEqual(CHECKER.check_python(Path("sample.py"), source), [])

    def test_non_hip_branch_is_ignored_for_hcu(self):
        source = 'if not _is_hip:\n    raise RuntimeError("AMD only")\n'
        self.assertEqual(CHECKER.check_python(Path("sample.py"), source), [])

    def test_amd_fallback_after_hcu_raise_is_unreachable(self):
        source = (
            "if _is_hcu:\n"
            '    raise RuntimeError("HCU error")\n'
            'raise RuntimeError("AMD error")\n'
        )
        self.assertEqual(CHECKER.check_python(Path("sample.py"), source), [])

    def test_raise_is_reported_once(self):
        source = 'raise RuntimeError("DCU runtime error")\n'
        errors = CHECKER.check_python(Path("sample.py"), source)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].location, "raise")

    def test_cli_help_only_checks_visible_help(self):
        source = (
            'parser.add_argument("--dcu-mode", dest="amd_mode", '
            'help="AMD runtime mode")\n'
        )
        errors = CHECKER.check_python(Path("sample.py"), source)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].text, "AMD runtime mode")

    def test_cpp_visible_output_is_reported(self):
        text = 'std::cerr << "XGMI link failed" << std::endl;\n'
        errors = CHECKER.check_text(Path("sample.cu"), text)
        self.assertEqual(len(errors), 1)

    def test_environment_file_list_preserves_spaces(self):
        with mock.patch.dict(os.environ, {"CHANGED_FILES": "dir/a file.py\ndir/b.py"}):
            paths = CHECKER.changed_paths([])
        self.assertEqual(
            [CHECKER.normalize_path(str(path)) for path in paths],
            ["dir/a file.py", "dir/b.py"],
        )

    def test_main_returns_violation_and_tool_error_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = self.write_file(root, "test/registered/hcu/bad.py", 'print("AMD")\n')
            syntax = self.write_file(root, "test/registered/hcu/syntax.py", "if:\n")

            previous = Path.cwd()
            os.chdir(str(root))
            try:
                with redirect_stdout(StringIO()):
                    self.assertEqual(CHECKER.main([str(bad.relative_to(root))]), 1)
                    self.assertEqual(CHECKER.main([str(syntax.relative_to(root))]), 2)
            finally:
                os.chdir(str(previous))


if __name__ == "__main__":
    unittest.main()
