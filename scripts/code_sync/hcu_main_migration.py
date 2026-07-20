#!/usr/bin/env python3
"""Read-only helper for the DCU main migration.

This script intentionally does not create branches, merge commits, tags, or
pushes. It reports migration state and prints templates that humans can use
when running the checkpoint workflow.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


BASE_SHA = "3117415c9bcd00ec06835eaae06690d53d18a334"
DEFAULT_OFFICIAL_REPO = Path(os.environ.get("SGLANG_OFFICIAL_REPO", "/home/officials/sglang"))


@dataclass(frozen=True)
class Checkpoint:
    cid: str
    cutoff_utc: str
    sha: str
    delta_commits: int
    risk: str
    notes: str

    @property
    def short_sha(self) -> str:
        return self.sha[:12]

    @property
    def tag_name(self) -> str:
        return f"dcu-main-bootstrap-{self.cid}-official-{self.cutoff_utc.replace('-', '')}"


CHECKPOINTS = [
    Checkpoint("C00", "2026-05-15", BASE_SHA, 0, "base", "Common merge base"),
    Checkpoint("C01", "2026-05-17", "c67b2870569a", 77, "high", "Heavy test and CI overlap"),
    Checkpoint("C02", "2026-05-19", "425dffbde339", 140, "medium", "DeepSeek V4 MTP and attention"),
    Checkpoint("C03", "2026-05-21", "7cf193fe1faf", 104, "high", "Cache, model, attention"),
    Checkpoint("C04", "2026-05-23", "af8f66940e9b", 66, "medium", "AMD DSV4 runtime and jit-kernel"),
    Checkpoint("C05", "2026-05-25", "8805f4cf1666", 50, "low", "PD and scheduler"),
    Checkpoint("C06", "2026-05-27", "0abe6a85a51f", 74, "medium", "Model and mem_cache"),
    Checkpoint("C07", "2026-05-29", "a5e6a8887a94", 113, "high", "Attention and test"),
    Checkpoint("C08", "2026-05-31", "373cadc92ea4", 57, "low", "Mooncake and CI"),
    Checkpoint("C09", "2026-06-02", "c55548ba115c", 103, "medium", "Embedding, mem_cache, attention"),
    Checkpoint("C10", "2026-06-04", "47377525cb32", 115, "high", "CI, mem_cache, attention"),
    Checkpoint("C11", "2026-06-06", "5160f7914ebf", 81, "medium", "MLA EAGLE and CUDA graph"),
    Checkpoint("C12", "2026-06-08", "3fe6bc390bdc", 76, "medium", "Spec naming cleanup"),
    Checkpoint("C13", "2026-06-10", "125ef888921b", 105, "high", "Model, MoE, jit-kernel"),
    Checkpoint("C14", "2026-06-12", "fda795589097", 109, "medium", "AMD DFlash and fused KV"),
    Checkpoint("C15", "2026-06-14", "000fc975c7b3", 70, "medium", "Docker and mem_cache"),
    Checkpoint("C16", "2026-06-16", "2ad00faae1f4", 92, "low", "Nightly tests"),
    Checkpoint("C17", "2026-06-18", "62ab09a47886", 107, "high", "AMD spec tests and models"),
    Checkpoint("C18", "2026-06-20", "f42ec350b431", 64, "low", "MTP rejection sampling"),
    Checkpoint("C19", "2026-06-22", "62b3c8e17781", 38, "low", "XPU import guard"),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git(repo: Path, *args: str, extra_env: dict[str, str] | None = None, check: bool = True) -> str:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout.strip()


def git_success(repo: Path, *args: str, extra_env: dict[str, str] | None = None) -> bool:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.returncode == 0


def alternate_env(official_repo: Path) -> dict[str, str]:
    return {"GIT_ALTERNATE_OBJECT_DIRECTORIES": str(official_repo / ".git" / "objects")}


def checkpoint_by_id(cid: str) -> Checkpoint:
    normalized = cid.upper()
    for checkpoint in CHECKPOINTS:
        if checkpoint.cid == normalized:
            return checkpoint
    raise SystemExit(f"Unknown checkpoint: {cid}")


def branch_exists(repo: Path, branch: str) -> bool:
    return git_success(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}")


def tag_exists(repo: Path, tag: str) -> bool:
    return git_success(repo, "show-ref", "--verify", "--quiet", f"refs/tags/{tag}")


def changed_files(repo: Path, rev_range: str, extra_env: dict[str, str] | None = None) -> set[str]:
    output = git(repo, "diff", "--name-only", rev_range, extra_env=extra_env)
    return {line for line in output.splitlines() if line}


def cmd_status(args: argparse.Namespace) -> None:
    dcu_repo = args.repo
    official_repo = args.official_repo
    official_head = git(official_repo, "rev-parse", "HEAD")
    dcu_head = git(dcu_repo, "rev-parse", "HEAD")
    merge_base = git(
        dcu_repo,
        "merge-base",
        "HEAD",
        official_head,
        extra_env=alternate_env(official_repo),
    )
    dcu_count = git(dcu_repo, "rev-list", "--count", f"{BASE_SHA}..HEAD")
    official_count = git(official_repo, "rev-list", "--count", f"{BASE_SHA}..HEAD")
    dcu_files = changed_files(dcu_repo, f"{BASE_SHA}..HEAD", extra_env=alternate_env(official_repo))
    official_files = changed_files(official_repo, f"{BASE_SHA}..HEAD")
    overlap = dcu_files & official_files

    print(f"DCU repo: {dcu_repo}")
    print(f"Official repo: {official_repo}")
    print(f"Current branch: {git(dcu_repo, 'branch', '--show-current')}")
    print(f"DCU HEAD: {dcu_head[:12]}")
    print(f"Official HEAD: {official_head[:12]}")
    print(f"Merge base: {merge_base[:12]}")
    print(f"DCU commits after base: {dcu_count}")
    print(f"Official commits after base: {official_count}")
    print(f"Overlapping changed files: {len(overlap)}")
    print(f"Local main exists: {'yes' if branch_exists(dcu_repo, 'main') else 'no'}")
    print(
        "Bootstrap branch exists: "
        f"{'yes' if branch_exists(dcu_repo, 'sync/official-main-bootstrap') else 'no'}"
    )
    rerere = git(dcu_repo, "config", "--get", "rerere.enabled", check=False) or "unset"
    print(f"rerere.enabled: {rerere}")


def cmd_checkpoints(args: argparse.Namespace) -> None:
    dcu_repo = args.repo
    print("| ID | Cutoff UTC | Checkpoint | Delta | Risk | Tag | Notes |")
    print("|---|---:|---|---:|---|---|---|")
    for checkpoint in CHECKPOINTS:
        if checkpoint.cid == "C00":
            state = "base"
        else:
            state = "done" if tag_exists(dcu_repo, checkpoint.tag_name) else "pending"
        print(
            f"| {checkpoint.cid} | {checkpoint.cutoff_utc} | `{checkpoint.short_sha}` | "
            f"{checkpoint.delta_commits} | {checkpoint.risk} | {state} | {checkpoint.notes} |"
        )


def cmd_next(args: argparse.Namespace) -> None:
    dcu_repo = args.repo
    for checkpoint in CHECKPOINTS:
        if checkpoint.cid == "C00":
            continue
        if not tag_exists(dcu_repo, checkpoint.tag_name):
            print(f"Next checkpoint: {checkpoint.cid}")
            print(f"Official SHA: {checkpoint.short_sha}")
            print(f"Cutoff UTC: {checkpoint.cutoff_utc}")
            print(f"Risk: {checkpoint.risk}")
            print(f"Merge branch: sync/official-main-{checkpoint.cid}-{checkpoint.cutoff_utc.replace('-', '')}")
            print(f"Tag name: {checkpoint.tag_name}")
            print(f"Notes: {checkpoint.notes}")
            return
    print("All bootstrap checkpoints have milestone tags.")


def cmd_tag_message(args: argparse.Namespace) -> None:
    checkpoint = checkpoint_by_id(args.checkpoint)
    dcu_sha = git(args.repo, "rev-parse", "HEAD")
    official_sha = git(args.official_repo, "rev-parse", checkpoint.sha)
    print(f"Tag: {checkpoint.tag_name}")
    print()
    print(f"Official checkpoint: {official_sha}")
    print("DCU base branch: v0.5.12_dev")
    print(f"DCU main sha: {dcu_sha}")
    print(f"Validation: {args.validation}")
    print("Known issues:")
    if args.known_issue:
        for issue in args.known_issue:
            print(f"- {issue}")
    else:
        print("- none")


def cmd_validation(args: argparse.Namespace) -> None:
    print("Phase 0:")
    print("  python3 scripts/ci/dcu/verify_dcu_registration.py")
    print()
    print("Phase 1:")
    print("  # Fill in internal DCU CI dry-run command")
    print("  python3 scripts/ci/dcu/verify_dcu_registration.py")
    print()
    print("Phase 2:")
    print("  # Fill in DCU stage-b small model smoke command")
    print("  # Fill in Qwen2.5 dense, VLM, embedding, reranker smoke commands")
    print("  # Fill in sgl-kernel DCU smoke whitelist command")
    print()
    print("Phase 3:")
    print("  # Fill in Qwen3 MoE, DeepEP, DeepSeek V4, nightly-dcu commands")
    print()
    print("Phase 4:")
    print("  # Fill in daily sync smoke gate and weekly nightly commands")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo_root(), help="Path to dcu-sglang repo")
    parser.add_argument(
        "--official-repo",
        type=Path,
        default=DEFAULT_OFFICIAL_REPO,
        help="Path to official SGLang repo",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Print current migration status").set_defaults(func=cmd_status)
    subparsers.add_parser("checkpoints", help="Print checkpoint table with local tag state").set_defaults(
        func=cmd_checkpoints
    )
    subparsers.add_parser("next", help="Print the next untagged checkpoint").set_defaults(func=cmd_next)
    subparsers.add_parser("validation", help="Print validation command placeholders").set_defaults(
        func=cmd_validation
    )

    tag_parser = subparsers.add_parser("tag-message", help="Print annotated tag message template")
    tag_parser.add_argument("checkpoint", help="Checkpoint ID, for example C01")
    tag_parser.add_argument("--validation", default="passed", help="Validation field value")
    tag_parser.add_argument("--known-issue", action="append", help="Known issue line; repeatable")
    tag_parser.set_defaults(func=cmd_tag_message)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
