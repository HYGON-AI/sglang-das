#!/usr/bin/env python3
"""Verify the VideoResponse.model schema default against an existing H3 job."""

from __future__ import annotations

import argparse
from typing import Any

from _openapi_probe import (
    ProbeError,
    add_common_arguments,
    fetch_openapi,
    finish,
    main_guard,
    request_json,
    resolve_ref,
    response_schema,
    schema_properties,
    video_post_operation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--video-id",
        help="Existing MiniMax-H3 video job ID; defaults to the newest listed job",
    )
    return parser.parse_args()


def _response_model_property(document: dict[str, Any]) -> dict[str, Any]:
    operation = video_post_operation(document)
    schema = response_schema(operation, 200)
    if schema is None:
        raise ProbeError("OpenAPI has no JSON schema for POST /v1/videos HTTP 200")
    model_property = schema_properties(document, schema).get("model")
    if not isinstance(model_property, dict):
        raise ProbeError("VideoResponse schema has no model property")
    resolved = resolve_ref(document, model_property)
    return resolved if isinstance(resolved, dict) else model_property


def _runtime_job(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.video_id:
        result = request_json(
            args.base_url,
            f"/v1/videos/{args.video_id}",
            timeout=args.timeout,
            api_key=args.api_key,
        )
        if result.status != 200 or not isinstance(result.body, dict):
            raise ProbeError(
                f"video job {args.video_id!r} returned HTTP {result.status}: "
                f"{result.raw_body[:300]}"
            )
        return result.body

    result = request_json(
        args.base_url,
        "/v1/videos?limit=1&order=desc",
        timeout=args.timeout,
        api_key=args.api_key,
    )
    if result.status != 200 or not isinstance(result.body, dict):
        raise ProbeError(
            f"GET /v1/videos returned HTTP {result.status}: {result.raw_body[:300]}"
        )
    data = result.body.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def main() -> int:
    args = parse_args()
    document = fetch_openapi(args)
    model_property = _response_model_property(document)
    schema_default = model_property.get("default")
    job = _runtime_job(args)

    if job is None and not args.allow_empty:
        raise ProbeError(
            "the video store is empty; first submit the documented T2VA request, "
            "or pass --video-id. Use --allow-empty only for a schema-only check"
        )

    runtime_model = job.get("model") if job else None
    if job is not None and not isinstance(runtime_model, str):
        raise ProbeError(f"video response has no string model field: {job!r}")

    runtime_disagrees = runtime_model is None or runtime_model != schema_default
    observed = (
        "gap" if schema_default == "sora-2" and runtime_disagrees else "fixed"
    )
    return finish(
        issue="03_video_response_model_default",
        expected=args.expect,
        observed=observed,
        evidence={
            "schema_model_default": schema_default,
            "runtime_job_id": job.get("id") if job else None,
            "runtime_model": runtime_model,
            "schema_only": job is None,
        },
    )


if __name__ == "__main__":
    main_guard(main)
