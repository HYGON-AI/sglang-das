#!/usr/bin/env python3
"""Verify whether MiniMax-H3 request fields are represented in OpenAPI."""

from __future__ import annotations

import argparse
from typing import Any

from _openapi_probe import (
    ProbeError,
    add_common_arguments,
    content_schema,
    fetch_openapi,
    finish,
    main_guard,
    request_content,
    runtime_missing_task_probe,
    schema_enum_values,
    schema_properties,
    schema_required,
    schema_types,
    video_post_operation,
    walk_schema,
)


H3_FIELDS = {"task", "conditions", "target", "audio_flow_shift"}
H3_TASKS = {"t2va", "fl2va", "ref2va"}
TARGET_FIELDS = {"short_edge", "aspect_ratio", "duration_seconds"}
CONDITION_FIELDS = {"type", "uri", "role", "frame_index", "start_time_seconds"}


def _item_properties(document: dict[str, Any], schema: Any) -> set[str]:
    properties: set[str] = set()
    for node in walk_schema(document, schema):
        if "items" in node:
            properties.update(schema_properties(document, node["items"]))
    return properties


def _has_positive_lower_bound(document: dict[str, Any], schema: Any) -> bool:
    for node in walk_schema(document, schema):
        exclusive_minimum = node.get("exclusiveMinimum")
        minimum = node.get("minimum")
        if isinstance(exclusive_minimum, (int, float)) and exclusive_minimum >= 0:
            return True
        if isinstance(minimum, (int, float)) and minimum > 0:
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = fetch_openapi(args)
    operation = video_post_operation(document)
    content = request_content(operation)

    declared: set[str] = set()
    per_media_type: dict[str, Any] = {}
    for media_type, media in content.items():
        if not isinstance(media, dict) or "schema" not in media:
            continue
        schema = content_schema(content, media_type)
        media_fields = set(schema_properties(document, schema))
        media_required = schema_required(document, schema)
        declared.update(media_fields)
        per_media_type[media_type] = {
            "h3_fields": sorted(H3_FIELDS & media_fields),
            "required": sorted(H3_FIELDS & media_required),
        }

    json_schema = None
    if "application/json" in content:
        json_schema = content_schema(content, "application/json")
    json_properties = (
        schema_properties(document, json_schema) if json_schema is not None else {}
    )
    json_required = (
        schema_required(document, json_schema) if json_schema is not None else set()
    )

    task_schema = json_properties.get("task")
    conditions_schema = json_properties.get("conditions")
    target_schema = json_properties.get("target")
    audio_flow_shift_schema = json_properties.get("audio_flow_shift")
    task_enum = schema_enum_values(document, task_schema)
    conditions_types = schema_types(document, conditions_schema)
    condition_fields = _item_properties(document, conditions_schema)
    target_types = schema_types(document, target_schema)
    target_fields = set(schema_properties(document, target_schema))
    target_required = schema_required(document, target_schema)
    audio_flow_shift_types = schema_types(document, audio_flow_shift_schema)

    structural_checks = {
        "task_required": "task" in json_required,
        "task_enum_complete": H3_TASKS <= task_enum,
        "conditions_is_array": "array" in conditions_types,
        "condition_fields_complete": CONDITION_FIELDS <= condition_fields,
        "target_is_object": "object" in target_types,
        "target_fields_complete": TARGET_FIELDS <= target_fields,
        "target_core_fields_required": {
            "short_edge",
            "aspect_ratio",
        }
        <= target_required,
        "audio_flow_shift_is_number": bool(
            {"number", "integer"} & audio_flow_shift_types
        ),
        "audio_flow_shift_is_positive": _has_positive_lower_bound(
            document, audio_flow_shift_schema
        ),
    }

    runtime = runtime_missing_task_probe(args)
    detail = runtime.body.get("detail") if isinstance(runtime.body, dict) else None
    runtime_requires_task = runtime.status in {400, 422} and "task" in str(detail).lower()
    missing = H3_FIELDS - declared
    if not runtime_requires_task:
        raise ProbeError(
            "runtime did not reject the no-task H3 probe as expected; "
            f"HTTP {runtime.status}: {runtime.raw_body[:300]}"
        )

    json_missing = H3_FIELDS - set(json_properties)
    if json_missing:
        observed = "gap"
    elif all(structural_checks.values()):
        observed = "fixed"
    else:
        observed = "partial"

    return finish(
        issue="01_h3_request_schema",
        expected=args.expect,
        observed=observed,
        evidence={
            "declared_media_types": sorted(content),
            "missing_h3_fields": sorted(missing),
            "json_missing_h3_fields": sorted(json_missing),
            "structural_checks": structural_checks,
            "task_enum": sorted(str(value) for value in task_enum),
            "conditions_types": sorted(conditions_types),
            "condition_fields": sorted(condition_fields),
            "target_types": sorted(target_types),
            "target_fields": sorted(target_fields),
            "target_required": sorted(target_required),
            "audio_flow_shift_types": sorted(audio_flow_shift_types),
            "per_media_type": per_media_type,
            "runtime_missing_task_status": runtime.status,
            "runtime_missing_task_detail": detail,
        },
    )


if __name__ == "__main__":
    main_guard(main)
