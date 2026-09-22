#!/usr/bin/env python3
"""Dependency-free helpers for MiniMax-H3 OpenAPI/runtime probes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: Any
    raw_body: str


class ProbeError(RuntimeError):
    pass


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30010",
        help="SGLang server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--expect",
        choices=("gap", "partial", "fixed"),
        default="gap",
        help=(
            "Expected server state: gap, partial, or fixed "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SGLANG_API_KEY"),
        help="Optional bearer token (default: SGLANG_API_KEY)",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help=(
            "Allow schema-only checks when applicable; ignored by probes "
            "that do not inspect stored jobs"
        ),
    )


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def request_json(
    base_url: str,
    path: str,
    *,
    timeout: float,
    api_key: str | None,
    method: str = "GET",
    payload: Any = None,
) -> HttpResult:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    data = None
    headers = _headers(api_key)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw_body = exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProbeError(f"request failed for {url}: {exc}") from exc

    try:
        body = json.loads(raw_body) if raw_body else None
    except json.JSONDecodeError:
        body = None
    return HttpResult(status=status, body=body, raw_body=raw_body)


def fetch_openapi(args: argparse.Namespace) -> dict[str, Any]:
    result = request_json(
        args.base_url,
        "/openapi.json",
        timeout=args.timeout,
        api_key=args.api_key,
    )
    if result.status != 200 or not isinstance(result.body, dict):
        raise ProbeError(
            f"GET /openapi.json returned HTTP {result.status}: {result.raw_body[:300]}"
        )
    return result.body


def resolve_ref(document: dict[str, Any], value: Any) -> Any:
    if not isinstance(value, dict) or "$ref" not in value:
        return value
    ref = value["$ref"]
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return value
    current: Any = document
    for token in ref[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            raise ProbeError(f"unresolvable OpenAPI reference: {ref}")
        current = current[token]
    return current


def walk_schema(
    document: dict[str, Any], schema: Any, seen_refs: set[str] | None = None
) -> Iterable[dict[str, Any]]:
    if not isinstance(schema, dict):
        return
    seen_refs = set() if seen_refs is None else seen_refs
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen_refs:
            return
        seen_refs.add(ref)
        yield from walk_schema(document, resolve_ref(document, schema), seen_refs)
        return

    yield schema
    for keyword in ("allOf", "anyOf", "oneOf"):
        for child in schema.get(keyword, []):
            yield from walk_schema(document, child, set(seen_refs))


def schema_properties(document: dict[str, Any], schema: Any) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for node in walk_schema(document, schema):
        node_properties = node.get("properties")
        if isinstance(node_properties, dict):
            properties.update(node_properties)
    return properties


def schema_required(document: dict[str, Any], schema: Any) -> set[str]:
    required: set[str] = set()
    for node in walk_schema(document, schema):
        values = node.get("required")
        if isinstance(values, list):
            required.update(value for value in values if isinstance(value, str))
    return required


def schema_types(document: dict[str, Any], schema: Any) -> set[str]:
    types: set[str] = set()
    for node in walk_schema(document, schema):
        value = node.get("type")
        if isinstance(value, str):
            types.add(value)
        elif isinstance(value, list):
            types.update(item for item in value if isinstance(item, str))
    return types


def schema_enum_values(document: dict[str, Any], schema: Any) -> set[Any]:
    values: set[Any] = set()
    for node in walk_schema(document, schema):
        enum = node.get("enum")
        if isinstance(enum, list):
            for value in enum:
                try:
                    values.add(value)
                except TypeError:
                    continue
        if "const" in node:
            try:
                values.add(node["const"])
            except TypeError:
                pass
    return values


def video_post_operation(document: dict[str, Any]) -> dict[str, Any]:
    try:
        operation = document["paths"]["/v1/videos"]["post"]
    except (KeyError, TypeError) as exc:
        raise ProbeError("OpenAPI has no POST /v1/videos operation") from exc
    if not isinstance(operation, dict):
        raise ProbeError("POST /v1/videos operation is not an object")
    return operation


def request_content(operation: dict[str, Any]) -> dict[str, Any]:
    content = operation.get("requestBody", {}).get("content", {})
    if not isinstance(content, dict):
        raise ProbeError("POST /v1/videos requestBody.content is not an object")
    return content


def content_schema(content: dict[str, Any], media_type: str) -> Any:
    media = content.get(media_type)
    if not isinstance(media, dict) or "schema" not in media:
        raise ProbeError(f"OpenAPI has no schema for media type {media_type}")
    return media["schema"]


def response_schema(operation: dict[str, Any], status: int) -> Any | None:
    response = operation.get("responses", {}).get(str(status))
    if not isinstance(response, dict):
        return None
    content = response.get("content", {})
    if not isinstance(content, dict):
        return None
    media = content.get("application/json")
    if not isinstance(media, dict):
        return None
    return media.get("schema")


def detail_types(document: dict[str, Any], schema: Any) -> set[str]:
    if schema is None:
        return set()
    types: set[str] = set()
    for node in walk_schema(document, schema):
        properties = node.get("properties")
        if isinstance(properties, dict) and "detail" in properties:
            types.update(schema_types(document, properties["detail"]))
    return types


def runtime_missing_task_probe(args: argparse.Namespace) -> HttpResult:
    # This reaches H3 admission but cannot enqueue inference because task is absent.
    payload = {
        "prompt": "OpenAPI contract probe; this request must not be queued.",
        "conditions": [],
        "target": {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
        },
        "seed": 1101,
        "n": 1,
        "num_inference_steps": 50,
        "flow_shift": 12,
        "audio_flow_shift": 3,
    }
    return request_json(
        args.base_url,
        "/v1/videos",
        timeout=args.timeout,
        api_key=args.api_key,
        method="POST",
        payload=payload,
    )


def finish(
    *,
    issue: str,
    expected: str,
    observed: str,
    evidence: dict[str, Any],
) -> int:
    if observed not in {"gap", "partial", "fixed"}:
        raise ValueError(f"invalid observed state: {observed!r}")
    matched = observed == expected
    payload = {
        "issue": issue,
        "expected": expected,
        "observed": observed,
        "matched": matched,
        "evidence": evidence,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if matched else 1


def main_guard(main) -> None:
    try:
        raise SystemExit(main())
    except ProbeError as exc:
        print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
