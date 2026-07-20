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

"""Reject sensitive platform terms added by a Git diff.

The checker is read-only. It examines destination paths and added lines only;
unchanged content, removed lines, and deleted paths are not scanned.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


SENSITIVE_TERMS = tuple(
    "".join(parts) for parts in (("d", "cu"), ("a", "md"), ("xg", "mi"))
)
SENSITIVE_RE = re.compile("|".join(map(re.escape, SENSITIVE_TERMS)), re.IGNORECASE)
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
MAX_SNIPPET_LEN = 180


class ToolFailure(Exception):
    pass


class Change:
    __slots__ = ("status", "old_path", "new_path")

    def __init__(self, status, old_path, new_path):
        self.status = status
        self.old_path = old_path
        self.new_path = new_path


class Violation:
    __slots__ = ("path", "line", "kind", "term", "text")

    def __init__(self, path, line, kind, term, text):
        self.path = path
        self.line = line
        self.kind = kind
        self.term = term
        self.text = text

    def key(self):
        return (self.path, self.line, self.kind, self.term.lower(), self.text)


def run_git(repo, args, text=False):
    command = ["git", "-C", str(repo), *args]
    try:
        run_options = {
            "check": False,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "universal_newlines": text,
        }
        if text:
            run_options["encoding"] = "utf-8"
            run_options["errors"] = "replace"
        result = subprocess.run(command, **run_options)
    except OSError as exc:
        raise ToolFailure(f"unable to execute Git: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() if text else result.stderr.decode(
            "utf-8", errors="replace"
        ).strip()
        raise ToolFailure(
            f"Git command failed ({' '.join(command)}): {stderr or 'unknown error'}"
        )
    return result.stdout


def decode_path(value):
    return value.decode("utf-8", errors="surrogateescape").replace("\\", "/")


def collect_changes(repo, base, head):
    output = run_git(
        repo,
        [
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            "--find-copies",
            "--diff-filter=ACMRT",
            base,
            head,
            "--",
        ],
    )
    tokens = output.split(b"\0")
    if tokens and not tokens[-1]:
        tokens.pop()

    changes = []
    index = 0
    while index < len(tokens):
        status = decode_path(tokens[index])
        index += 1
        code = status[:1]

        if code in {"R", "C"}:
            if index + 1 >= len(tokens):
                raise ToolFailure("unable to parse renamed or copied path from Git diff")
            old_path = decode_path(tokens[index])
            new_path = decode_path(tokens[index + 1])
            index += 2
        else:
            if index >= len(tokens):
                raise ToolFailure("unable to parse changed path from Git diff")
            new_path = decode_path(tokens[index])
            old_path = new_path
            index += 1

        changes.append(Change(status, old_path, new_path))
    return changes


def find_terms(text):
    return [(match.group(0), match.start()) for match in SENSITIVE_RE.finditer(text)]


def added_lines_for_change(repo, base, head, change):
    pathspecs = [change.new_path]
    if change.old_path != change.new_path:
        pathspecs.insert(0, change.old_path)

    patch = run_git(
        repo,
        [
            "diff",
            "--unified=0",
            "--no-color",
            "--no-ext-diff",
            "--find-renames",
            "--find-copies",
            "--diff-filter=ACMRT",
            base,
            head,
            "--",
            *pathspecs,
        ],
        text=True,
    )

    in_hunk = False
    new_line = 0
    for patch_line in patch.splitlines():
        match = HUNK_RE.match(patch_line)
        if match:
            in_hunk = True
            new_line = int(match.group(1))
            continue
        if patch_line.startswith("diff --git "):
            in_hunk = False
            continue
        if not in_hunk or patch_line.startswith("\\"):
            continue
        if patch_line.startswith("+"):
            yield new_line, patch_line[1:]
            new_line += 1
        elif patch_line.startswith("-"):
            continue
        else:
            new_line += 1


def scan_diff(repo, base, head):
    violations = []
    for change in collect_changes(repo, base, head):
        for term, _ in find_terms(change.new_path):
            violations.append(
                Violation(
                    change.new_path,
                    1,
                    "changed file path",
                    term,
                    change.new_path,
                )
            )

        for line, text in added_lines_for_change(repo, base, head, change):
            for term, _ in find_terms(text):
                violations.append(
                    Violation(change.new_path, line, "added content", term, text)
                )
    return deduplicate(violations)


def deduplicate(violations):
    result = []
    seen = set()
    for violation in violations:
        if violation.key() not in seen:
            seen.add(violation.key())
            result.append(violation)
    return result


def text_snippet(text):
    snippet = " ".join(text.split())
    if len(snippet) > MAX_SNIPPET_LEN:
        snippet = snippet[: MAX_SNIPPET_LEN - 3] + "..."
    return repr(snippet)


def escape_annotation_data(text):
    return (
        text.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def escape_annotation_property(text):
    return (
        escape_annotation_data(text)
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def emit_github_error(violation):
    message = (
        f"sensitive term {violation.term!r} found in {violation.kind}: "
        f"{text_snippet(violation.text)}"
    )
    print(
        "::error "
        f"file={escape_annotation_property(violation.path)},"
        f"line={violation.line},"
        "title=Sensitive diff text check::"
        f"{escape_annotation_data(message)}"
    )


def print_failure_guidance():
    print("This check found sensitive platform terms in added paths or diff content.")
    print()
    print("Checked scope:")
    print(" - Only destination paths and lines added by this diff are checked.")
    print(
        " - Paths, file names, identifiers, imports, comments, strings, logs, "
        "configuration, and workflow text are covered."
    )
    print(" - Unchanged content, removed lines, and deleted paths are ignored.")
    print()
    print("Recommended remediation:")
    print(
        " - Rename reported sensitive terms in added paths, file names, and "
        "identifiers using HCU / hcu wording."
    )
    print(" - For HCU hardware wording, use HCU / HCU device(s).")
    print(" - For HCU fabric or link wording, use HSL / hsl.")
    print(" - Keep ROCm and HIP unchanged for software or runtime stack references.")
    print(
        " - Use HCU/ROCm or HCU/HIP only for combined hardware and software "
        "stack descriptions."
    )


def verify_revision(repo, revision, label):
    try:
        run_git(repo, ["cat-file", "-e", f"{revision}^{{tree}}"])
    except ToolFailure as exc:
        raise ToolFailure(f"invalid {label} revision {revision!r}: {exc}") from exc


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Check added Git paths and lines for sensitive platform terms."
    )
    parser.add_argument("--base", required=True, help="Base Git revision or tree.")
    parser.add_argument("--head", required=True, help="Head Git revision or tree.")
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository directory. Defaults to the current directory.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo = Path(args.repo).resolve()

    try:
        if not (repo / ".git").exists():
            raise ToolFailure(f"not a Git repository: {repo}")
        verify_revision(repo, args.base, "base")
        verify_revision(repo, args.head, "head")
        violations = scan_diff(repo, args.base, args.head)
    except ToolFailure as exc:
        print(f"Sensitive diff text checker could not complete: {exc}")
        print(
            "::error title=Sensitive diff text checker error::"
            f"{escape_annotation_data(str(exc))}"
        )
        return 2

    if violations:
        print("Sensitive diff text check failed.")
        print()
        print_failure_guidance()
        print()
        print("Detected items:")
        for violation in violations:
            print(
                f" - {violation.path}:{violation.line}: {violation.kind} contains "
                f"{violation.term!r}: {text_snippet(violation.text)}"
            )
            emit_github_error(violation)
        return 1

    print("Sensitive diff text check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
