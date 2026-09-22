#!/usr/bin/env python3
"""Verify that runtime H3 errors are represented by OpenAPI response schemas."""

from __future__ import annotations

import argparse

from _openapi_probe import (
    add_common_arguments,
    detail_types,
    fetch_openapi,
    finish,
    main_guard,
    response_schema,
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
    operation = video_post_operation(document)
    responses = operation.get("responses", {})

    runtime = runtime_missing_task_probe(args)
    detail = runtime.body.get("detail") if isinstance(runtime.body, dict) else None
    if isinstance(detail, str):
        runtime_detail_type = "string"
    elif isinstance(detail, list):
        runtime_detail_type = "array"
    else:
        runtime_detail_type = type(detail).__name__

    matching_schema = response_schema(operation, runtime.status)
    matching_detail_types = detail_types(document, matching_schema)
    runtime_error_documented = matching_schema is not None and (
        runtime_detail_type in matching_detail_types
    )
    response_422_types = detail_types(document, response_schema(operation, 422))
    observed = "fixed" if runtime_error_documented else "gap"

    return finish(
        issue="04_error_response_schema",
        expected=args.expect,
        observed=observed,
        evidence={
            "declared_response_codes": sorted(responses),
            "runtime_status": runtime.status,
            "runtime_detail_type": runtime_detail_type,
            "runtime_detail": detail,
            "matching_response_detail_types": sorted(matching_detail_types),
            "response_422_detail_types": sorted(response_422_types),
            "runtime_error_documented": runtime_error_documented,
        },
    )


if __name__ == "__main__":
    main_guard(main)
