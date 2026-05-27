from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = REPO_ROOT / "sglang.csv"

CSV_UNVERIFIED_MARKERS = (
    "DCU_CSV_CI_UNVERIFIED",
    "DCU_CSV_COVERED_UNVERIFIED",
)

MANUAL_DCU_CASEFILES = {
    "lora/test_lora_llama4.py",
}

MISSING_OR_OBSOLETE_CASEFILES = {
    "hicache/test_hicache.py",
    "hicache/test_hicache_eagle.py",
    "hicache/test_hicache_mla.py",
    "lora/test_lora.py",
    "openai_server/features/test_json_constrained.py",
    "test_ebnf_constrained.py",
    "test_regex_constrained.py",
    "test_vllm_dependency.py",
}


@dataclass
class CoverageRow:
    casefile: str
    tag: str
    gpu_need: str
    timeout: str
    status: str
    dcu_file: str
    suite: str
    nightly: bool | str
    disabled: str
    marker: str


def _load_ci_register():
    path = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
    spec = importlib.util.spec_from_file_location("dcu_ci_register", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ci_register from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_registries():
    ci_register = _load_ci_register()
    files = [
        str(path)
        for path in (REPO_ROOT / "test" / "registered").rglob("test_*.py")
    ]
    registries = ci_register.collect_tests(files, sanity_check=False)
    return ci_register, registries


def _build_indexes(registries):
    suffix_index: dict[str, list] = defaultdict(list)
    basename_index: dict[str, list] = defaultdict(list)

    for registry in registries:
        rel_path = os.path.relpath(registry.filename, REPO_ROOT)
        path = Path(rel_path)
        basename_index[path.name].append(registry)
        parts = path.parts
        for index in range(len(parts)):
            suffix_index["/".join(parts[index:])].append(registry)

    return suffix_index, basename_index


def _find_registration(casefile: str, suffix_index, basename_index):
    candidates = []
    for key in (casefile, f"test/registered/{casefile}"):
        matches = suffix_index.get(key, [])
        if len(matches) == 1:
            return matches[0]
        candidates.extend(matches)

    basename_matches = basename_index.get(Path(casefile).name, [])
    if len(basename_matches) == 1:
        return basename_matches[0]

    return None


def _has_marker(filename: str) -> str:
    text = Path(filename).read_text(encoding="utf-8", errors="ignore")
    for marker in CSV_UNVERIFIED_MARKERS:
        if marker in text:
            return marker
    return ""


def _read_csv_rows():
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_csv_matrix() -> list[CoverageRow]:
    ci_register, registries = _collect_registries()
    dcu_registries = [
        registry
        for registry in registries
        if registry.backend == ci_register.HWBackend.DCU
    ]
    suffix_index, basename_index = _build_indexes(dcu_registries)

    matrix: list[CoverageRow] = []
    for row in _read_csv_rows():
        casefile = row["casefile"]
        if casefile in MANUAL_DCU_CASEFILES:
            matrix.append(
                CoverageRow(
                    casefile=casefile,
                    tag=row["tag"],
                    gpu_need=row["gpu_need"],
                    timeout=row["timeout"],
                    status="manual_dcu",
                    dcu_file="test/manual/lora/test_lora_llama4.py",
                    suite="manual",
                    nightly="manual",
                    disabled="manual test; not part of automated DCU CI",
                    marker="manual",
                )
            )
            continue

        if casefile in MISSING_OR_OBSOLETE_CASEFILES:
            matrix.append(
                CoverageRow(
                    casefile=casefile,
                    tag=row["tag"],
                    gpu_need=row["gpu_need"],
                    timeout=row["timeout"],
                    status="missing_or_obsolete",
                    dcu_file="",
                    suite="",
                    nightly="",
                    disabled="not found in current repository; excluded from this automation pass",
                    marker="missing",
                )
            )
            continue

        registry = _find_registration(casefile, suffix_index, basename_index)
        if registry is None:
            matrix.append(
                CoverageRow(
                    casefile=casefile,
                    tag=row["tag"],
                    gpu_need=row["gpu_need"],
                    timeout=row["timeout"],
                    status="unregistered_ci",
                    dcu_file="",
                    suite="",
                    nightly="",
                    disabled="repository file exists or matching is ambiguous, but no DCU registration was found",
                    marker="",
                )
            )
            continue

        status = "enabled" if registry.disabled is None else "registered_disabled"
        matrix.append(
            CoverageRow(
                casefile=casefile,
                tag=row["tag"],
                gpu_need=row["gpu_need"],
                timeout=row["timeout"],
                status=status,
                dcu_file=os.path.relpath(registry.filename, REPO_ROOT),
                suite=registry.suite,
                nightly=registry.nightly,
                disabled=registry.disabled or "",
                marker=_has_marker(registry.filename),
            )
        )

    return matrix


def build_intersection_matrix():
    ci_register, registries = _collect_registries()
    by_backend: dict[object, dict[str, object]] = defaultdict(dict)
    for registry in registries:
        rel_path = os.path.relpath(registry.filename, REPO_ROOT)
        by_backend[registry.backend][rel_path] = registry

    amd = set(by_backend[ci_register.HWBackend.AMD])
    cuda = set(by_backend[ci_register.HWBackend.CUDA])
    dcu = by_backend[ci_register.HWBackend.DCU]
    csv_files = {row.casefile for row in build_csv_matrix()}

    rows = []
    for filename in sorted(amd & cuda):
        dcu_registry = dcu.get(filename)
        if dcu_registry is None:
            status = "missing_dcu"
            disabled = ""
            suite = ""
            nightly = ""
        elif dcu_registry.disabled is None:
            status = "enabled"
            disabled = ""
            suite = dcu_registry.suite
            nightly = dcu_registry.nightly
        else:
            status = "disabled"
            disabled = dcu_registry.disabled
            suite = dcu_registry.suite
            nightly = dcu_registry.nightly

        path = Path(filename)
        csv_overlap = (
            "yes"
            if any(
                Path(csv_case).name == path.name
                or filename.endswith(csv_case)
                or f"test/registered/{csv_case}" == filename
                for csv_case in csv_files
            )
            else "no"
        )
        rows.append(
            {
                "file": filename,
                "status": status,
                "suite": suite,
                "nightly": nightly,
                "csv_overlap": csv_overlap,
                "disabled": disabled,
            }
        )
    return rows


def _markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item).replace("\n", " ") for item in row) + " |")
    return "\n".join(lines)


def write_markdown(csv_matrix: list[CoverageRow], intersection_rows: list[dict]):
    csv_counts = Counter(row.status for row in csv_matrix)
    intersection_counts = Counter(row["status"] for row in intersection_rows)

    csv_doc = [
        "# DCU CSV Coverage Matrix",
        "",
        "Generated by `python3 scripts/ci/dcu/analyze_dcu_csv_coverage.py --write-md`.",
        "",
        "## Summary",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["CSV rows", len(csv_matrix)],
                ["Unique CSV casefiles", len({row.casefile for row in csv_matrix})],
                ["Enabled", csv_counts["enabled"]],
                ["Registered disabled", csv_counts["registered_disabled"]],
                ["Manual DCU", csv_counts["manual_dcu"]],
                ["Missing or obsolete", csv_counts["missing_or_obsolete"]],
                ["Unregistered CI", csv_counts["unregistered_ci"]],
            ],
        ),
        "",
        "## Matrix",
        "",
        _markdown_table(
            [
                "Casefile",
                "Tag",
                "Status",
                "DCU file",
                "Suite",
                "Nightly",
                "Marker",
                "Disabled reason",
            ],
            [
                [
                    row.casefile,
                    row.tag,
                    row.status,
                    row.dcu_file,
                    row.suite,
                    row.nightly,
                    row.marker,
                    row.disabled,
                ]
                for row in csv_matrix
            ],
        ),
        "",
    ]
    (REPO_ROOT / "DCU_CSV覆盖矩阵.md").write_text("\n".join(csv_doc), encoding="utf-8")

    intersection_doc = [
        "# DCU AMD/NVIDIA Intersection Coverage Matrix",
        "",
        "Generated by `python3 scripts/ci/dcu/analyze_dcu_csv_coverage.py --write-md`.",
        "",
        "Policy: intersection tests with `CSV overlap = no` must be validated on DCU before they can be considered complete; they cannot use historical CSV coverage as an enablement reason.",
        "",
        "## Summary",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["AMD/NVIDIA intersection", len(intersection_rows)],
                ["Enabled", intersection_counts["enabled"]],
                ["Disabled", intersection_counts["disabled"]],
                ["Missing DCU", intersection_counts["missing_dcu"]],
                [
                    "CSV overlap",
                    sum(1 for row in intersection_rows if row["csv_overlap"] == "yes"),
                ],
                [
                    "Not in CSV",
                    sum(1 for row in intersection_rows if row["csv_overlap"] == "no"),
                ],
                [
                    "Not in CSV and disabled",
                    sum(
                        1
                        for row in intersection_rows
                        if row["csv_overlap"] == "no" and row["status"] == "disabled"
                    ),
                ],
            ],
        ),
        "",
        "## Matrix",
        "",
        _markdown_table(
            ["File", "Status", "Suite", "Nightly", "CSV overlap", "Disabled reason"],
            [
                [
                    row["file"],
                    row["status"],
                    row["suite"],
                    row["nightly"],
                    row["csv_overlap"],
                    row["disabled"],
                ]
                for row in intersection_rows
            ],
        ),
        "",
    ]
    (REPO_ROOT / "DCU_AMD_NVIDIA交集覆盖矩阵.md").write_text(
        "\n".join(intersection_doc), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-md", action="store_true")
    args = parser.parse_args()

    csv_matrix = build_csv_matrix()
    intersection_rows = build_intersection_matrix()

    csv_counts = Counter(row.status for row in csv_matrix)
    intersection_counts = Counter(row["status"] for row in intersection_rows)

    print("CSV Coverage Matrix")
    print(f"  csv_rows: {len(csv_matrix)}")
    print(f"  unique_casefiles: {len({row.casefile for row in csv_matrix})}")
    for key in sorted(csv_counts):
        print(f"  {key}: {csv_counts[key]}")
    print()

    print("AMD/NVIDIA Intersection Matrix")
    print(f"  intersection: {len(intersection_rows)}")
    for key in sorted(intersection_counts):
        print(f"  {key}: {intersection_counts[key]}")
    print(
        "  csv_overlap:",
        sum(1 for row in intersection_rows if row["csv_overlap"] == "yes"),
    )
    print(
        "  not_in_csv:",
        sum(1 for row in intersection_rows if row["csv_overlap"] == "no"),
    )

    if args.write_md:
        write_markdown(csv_matrix, intersection_rows)
        print()
        print("Wrote DCU_CSV覆盖矩阵.md")
        print("Wrote DCU_AMD_NVIDIA交集覆盖矩阵.md")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
