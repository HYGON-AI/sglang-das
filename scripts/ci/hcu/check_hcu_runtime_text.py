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

"""Check changed HCU runtime paths for platform-sensitive visible text.

The checker is intentionally read-only. It reports violations and returns a
non-zero exit code, but never modifies source files.
"""

import ast
import os
import re
import sys
from pathlib import Path


HCU_PATH_PREFIXES = (
    "scripts/ci/hcu/",
    "test/registered/hcu/",
)
HCU_EXACT_FILES = {
    "python/sglang/test/hcu_utils.py",
    "requirements_hcu.txt",
}
EXEMPT_FILES = {
    "scripts/ci/hcu/check_hcu_runtime_text.py",
}
EXEMPT_PATH_PREFIXES = ("scripts/ci/hcu/tests/",)
HCU_CODE_MARKER_RE = re.compile(
    r"\b(?:is_hcu|_is_hcu|register_hcu_ci)\b|"
    r"\bHWBackend\.(?:HCU|DCU)\b"
)
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
    "log_info_on_rank0",
    "log_warning_on_rank0",
    "log_error_on_rank0",
    "RuntimeError",
    "ValueError",
    "AssertionError",
    "Exception",
    "ImportError",
    "NotImplementedError",
}
TEXT_EXTS = {
    ".py",
    ".sh",
    ".bash",
    ".yml",
    ".yaml",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".h",
    ".hpp",
}
VISIBLE_TEXT_RE = re.compile(
    r"^\s*(?:-?\s*)?(?:name|description):|"
    r"\b(?:echo|printf|fprintf|sprintf|snprintf|puts)\s*\(|"
    r"\bstd::(?:cerr|cout|clog)\b|"
    r"\b(?:LOG|LOG_INFO|LOG_WARNING|LOG_ERROR|SPDLOG_[A-Z]+)\s*\(|"
    r"\bthrow\s+(?:std::)?\w+|"
    r"::(?:error|warning|notice)\b",
    re.IGNORECASE,
)
BLOCKED_TEXT_RE = re.compile(r"AMD|XGMI|DCU", re.IGNORECASE)
MAX_SNIPPET_LEN = 140
UNKNOWN = object()


class Violation:
    __slots__ = ("path", "lineno", "location", "text")

    def __init__(self, path, lineno, location, text):
        self.path = path
        self.lineno = lineno
        self.location = location
        self.text = text

    def display_path(self):
        return normalize_path(str(self.path))

    def key(self):
        return (self.display_path(), self.lineno, self.location, self.text)

    def message(self):
        return (
            f"{self.display_path()}:{self.lineno}: user-visible "
            f"platform-sensitive text in {self.location}: {text_snippet(self.text)}"
        )


class ScanFailure:
    __slots__ = ("path", "message")

    def __init__(self, path, message):
        self.path = path
        self.message = message

    def display(self):
        return f"{normalize_path(str(self.path))}: {self.message}"


def normalize_path(name):
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def is_hcu_owned_path(path):
    path = normalize_path(path)
    if path in HCU_EXACT_FILES:
        return True
    if path.startswith(".github/workflows/"):
        return any(token in Path(path).name.lower() for token in ("dcu", "hcu"))
    return any(path.startswith(prefix) for prefix in HCU_PATH_PREFIXES)


def is_exempt(path):
    path = normalize_path(path)
    return path in EXEMPT_FILES or any(
        path.startswith(prefix) for prefix in EXEMPT_PATH_PREFIXES
    )


def has_hcu_code_marker(text):
    return HCU_CODE_MARKER_RE.search(text) is not None


def function_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def is_string_node(node):
    constant_cls = getattr(ast, "Constant", None)
    if constant_cls is not None and isinstance(node, constant_cls):
        return isinstance(node.value, str)
    return node.__class__.__name__ == "Str" and isinstance(
        getattr(node, "s", None), str
    )


def string_node_value(node):
    if hasattr(node, "value"):
        return node.value
    return node.s


def is_true_on_hcu(node):
    platform_names = {
        "is_hcu",
        "_is_hcu",
        "is_hip",
        "_is_hip",
    }
    if isinstance(node, ast.Name):
        return node.id in platform_names
    if isinstance(node, ast.Attribute):
        return node.attr in platform_names
    if isinstance(node, ast.Call):
        return function_name(node.func) in {"is_hcu", "is_hip"}
    return False


def evaluate_for_hcu(node):
    """Evaluate known HCU conditions while leaving unrelated values unknown."""
    if is_true_on_hcu(node):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = evaluate_for_hcu(node.operand)
        return UNKNOWN if value is UNKNOWN else not value
    if isinstance(node, ast.BoolOp):
        values = [evaluate_for_hcu(value) for value in node.values]
        if isinstance(node.op, ast.And):
            if False in values:
                return False
            return True if all(value is True for value in values) else UNKNOWN
        if isinstance(node.op, ast.Or):
            if True in values:
                return True
            return False if all(value is False for value in values) else UNKNOWN
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        if is_true_on_hcu(node.left):
            comparator = node.comparators[0]
            constant_cls = getattr(ast, "Constant", ast.AST)
            if isinstance(comparator, (ast.NameConstant, constant_cls)):
                value = getattr(comparator, "value", None)
                if isinstance(value, bool):
                    if isinstance(node.ops[0], (ast.Eq, ast.Is)):
                        return value
                    if isinstance(node.ops[0], (ast.NotEq, ast.IsNot)):
                        return not value
    return UNKNOWN


def expression_strings(node):
    if node is None:
        return
    if is_string_node(node):
        yield node.lineno, string_node_value(node)
        return
    if isinstance(node, ast.IfExp):
        value = evaluate_for_hcu(node.test)
        if value is not False:
            yield from expression_strings(node.body)
        if value is not True:
            yield from expression_strings(node.orelse)
        return
    if isinstance(node, ast.Call):
        for arg in node.args:
            yield from expression_strings(arg)
        for keyword in node.keywords:
            yield from expression_strings(keyword.value)
        return
    for child in ast.iter_child_nodes(node):
        yield from expression_strings(child)


def visible_call_strings(node, name):
    if name == "add_argument":
        for keyword in node.keywords:
            if keyword.arg in {"help", "description", "epilog"}:
                yield from expression_strings(keyword.value)
        return
    if name == "skipif":
        for keyword in node.keywords:
            if keyword.arg == "reason":
                yield from expression_strings(keyword.value)
        return
    for arg in node.args:
        yield from expression_strings(arg)
    for keyword in node.keywords:
        if keyword.arg in {"reason", "help", "description", "epilog"}:
            yield from expression_strings(keyword.value)


def text_snippet(text):
    snippet = " ".join(text.split())
    if len(snippet) > MAX_SNIPPET_LEN:
        snippet = snippet[: MAX_SNIPPET_LEN - 3] + "..."
    return repr(snippet)


def escape_annotation_message(text):
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_annotation_property(text):
    return escape_annotation_message(text).replace(":", "%3A").replace(",", "%2C")


def emit_github_error(error):
    message = (
        "HCU user-visible output contains platform-sensitive wording. "
        "Please update the reported runtime text according to the HCU wording rules above."
    )
    print(
        "::error "
        f"file={escape_annotation_property(error.display_path())},"
        f"line={error.lineno},"
        "title=HCU runtime text check::"
        f"{escape_annotation_message(message)}"
    )


def emit_github_tool_error(failure):
    print(
        "::error title=HCU runtime text checker error::"
        f"{escape_annotation_message(failure.display())}"
    )


def has_blocked_text(text):
    return BLOCKED_TEXT_RE.search(text) is not None


class HcuVisibleTextVisitor(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.errors = []

    def add_strings(self, location, values):
        for lineno, value in values:
            if has_blocked_text(value):
                self.errors.append(Violation(self.path, lineno, location, value))

    def visit_block(self, statements):
        for statement in statements:
            self.visit(statement)
            if statement_terminates_on_hcu(statement):
                break

    def visit_Module(self, node):
        self.visit_block(node.body)

    def visit_ClassDef(self, node):
        self.visit_block(node.body)

    def visit_FunctionDef(self, node):
        self.visit_block(node.body)

    def visit_AsyncFunctionDef(self, node):
        self.visit_block(node.body)

    def visit_If(self, node):
        value = evaluate_for_hcu(node.test)
        if value is not False:
            self.visit_block(node.body)
        if value is not True:
            self.visit_block(node.orelse)

    def visit_ExceptHandler(self, node):
        self.visit_block(node.body)

    def visit_Try(self, node):
        self.visit_block(node.body)
        for handler in node.handlers:
            self.visit(handler)
        self.visit_block(node.orelse)
        self.visit_block(node.finalbody)

    def visit_With(self, node):
        self.visit_block(node.body)

    def visit_AsyncWith(self, node):
        self.visit_block(node.body)

    def visit_For(self, node):
        self.visit_block(node.body)
        self.visit_block(node.orelse)

    def visit_AsyncFor(self, node):
        self.visit_block(node.body)
        self.visit_block(node.orelse)

    def visit_While(self, node):
        self.visit_block(node.body)
        self.visit_block(node.orelse)

    def visit_Raise(self, node):
        self.add_strings("raise", expression_strings(node.exc))

    def visit_Call(self, node):
        name = function_name(node.func)
        if name in VISIBLE_CALLS:
            self.add_strings(f"{name}()", visible_call_strings(node, name))
        self.generic_visit(node)


def block_terminates_on_hcu(statements):
    return any(statement_terminates_on_hcu(statement) for statement in statements)


def statement_terminates_on_hcu(statement):
    if isinstance(statement, (ast.Raise, ast.Return)):
        return True
    if isinstance(statement, ast.If):
        value = evaluate_for_hcu(statement.test)
        if value is True:
            return block_terminates_on_hcu(statement.body)
        if value is False:
            return block_terminates_on_hcu(statement.orelse)
        return (
            bool(statement.orelse)
            and block_terminates_on_hcu(statement.body)
            and block_terminates_on_hcu(statement.orelse)
        )
    return False


def check_python(path, text):
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"Python parse failed at line {exc.lineno}: {exc.msg}")
    visitor = HcuVisibleTextVisitor(path)
    visitor.visit(tree)
    return visitor.errors


def check_text(path, text):
    errors = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if VISIBLE_TEXT_RE.search(line) and has_blocked_text(line):
            errors.append(Violation(path, lineno, "visible text", line.strip()))
    return errors


def changed_paths(argv):
    values = argv if argv else os.environ.get("CHANGED_FILES", "").splitlines()
    return [Path(normalize_path(value.strip())) for value in values if value.strip()]


def deduplicate(errors):
    result = []
    seen = set()
    for error in errors:
        if error.key() not in seen:
            seen.add(error.key())
            result.append(error)
    return result


def main(argv):
    errors = []
    failures = []

    for path in changed_paths(argv):
        normalized = normalize_path(str(path))
        if is_exempt(normalized) or not path.exists() or not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in TEXT_EXTS and normalized not in HCU_EXACT_FILES:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            failures.append(ScanFailure(path, f"unable to read UTF-8 text: {exc}"))
            continue

        if not is_hcu_owned_path(normalized) and not has_hcu_code_marker(text):
            continue

        try:
            if suffix == ".py":
                errors.extend(check_python(path, text))
            else:
                errors.extend(check_text(path, text))
        except ValueError as exc:
            failures.append(ScanFailure(path, str(exc)))

    if failures:
        print("HCU runtime text checker could not complete.")
        for failure in failures:
            print(f" - {failure.display()}")
            emit_github_tool_error(failure)
        return 2

    errors = deduplicate(errors)
    if errors:
        print("HCU runtime text check failed.")
        print()
        print("This check found platform-sensitive wording in user-visible output.")
        print()
        print("Checked scope:")
        print(" - Only files changed by the pull request or push are checked.")
        print(
            " - HCU paths, HCU-specific files, and files with HCU code markers are covered."
        )
        print(" - Python: user-visible strings in print/logger/raise/skip/help.")
        print(" - Shell/YAML/C/C++: common visible output statements and fields.")
        print(
            " - Source identifiers, imports, comments, and file names are not checked."
        )
        print()
        print("Detected items:")
        for error in errors:
            print(f" - {error.message()}")
            emit_github_error(error)
        return 1

    print("HCU runtime text check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
