#!/usr/bin/env python3
"""Verify the application/json documentation versus runtime acceptance gap."""

from __future__ import annotations

import argparse

from _openapi_probe import (
    ProbeError,
    add_common_arguments,
    fetch_openapi,
    finish,
    main_guard,
    request_content,
    runtime_missing_task_probe,
    video_post_operation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = fetch_openapi(args)
    content = request_content(video_post_operation(document))
    json_declared = "application/json" in content

    runtime = runtime_missing_task_probe(args)
    detail = runtime.body.get("detail") if isinstance(runtime.body, dict) else None
    json_reached_h3_admission = (
        runtime.status in {400, 422} and "task" in str(detail).lower()
    )
    if not json_reached_h3_admission:
        raise ProbeError(
            "JSON probe did not reach MiniMax-H3 task admission; "
            f"HTTP {runtime.status}: {runtime.raw_body[:300]}"
        )

    return finish(
        issue="02_json_content_type_undocumented",
        expected=args.expect,
        observed=(
            "fixed" if json_declared and json_reached_h3_admission else "gap"
        ),
        evidence={
            "declared_media_types": sorted(content),
            "application_json_declared": json_declared,
            "runtime_json_status": runtime.status,
            "runtime_json_detail": detail,
            "runtime_json_reached_h3_admission": json_reached_h3_admission,
        },
    )


if __name__ == "__main__":
    main_guard(main)
