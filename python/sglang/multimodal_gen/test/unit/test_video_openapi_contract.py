import json

from fastapi import FastAPI

from sglang.multimodal_gen.runtime.entrypoints.openai import video_api


def _resolve_ref(document, schema):
    if not isinstance(schema, dict) or "$ref" not in schema:
        return schema
    current = document
    for token in schema["$ref"][2:].split("/"):
        current = current[token.replace("~1", "/").replace("~0", "~")]
    return current


def _schema_types(schema):
    if "type" in schema:
        return {schema["type"]}
    return {
        branch["type"]
        for branch in schema.get("anyOf", [])
        if isinstance(branch, dict) and "type" in branch
    }


def test_minimax_h3_video_openapi_contract():
    app = FastAPI()
    app.include_router(video_api.router)
    video_api.configure_video_openapi(app, minimax_h3=True)

    document = app.openapi()
    operation = document["paths"]["/v1/videos"]["post"]
    content = operation["requestBody"]["content"]
    schema = content["application/json"]["schema"]
    properties = schema["properties"]

    assert '"$defs"' not in json.dumps(schema)
    assert '"$ref"' not in json.dumps(schema)
    assert set(schema["required"]) >= {"prompt", "task", "target"}
    assert set(properties["task"]["enum"]) == {"t2va", "fl2va", "ref2va"}

    multipart_schema = _resolve_ref(
        document,
        content["multipart/form-data"]["schema"],
    )
    assert set(multipart_schema["properties"]) >= {
        "task",
        "conditions",
        "target",
        "audio_flow_shift",
    }

    conditions = properties["conditions"]
    assert "array" in _schema_types(conditions)
    condition = conditions["items"]
    assert set(condition["properties"]) >= {
        "type",
        "uri",
        "role",
        "frame_index",
        "start_time_seconds",
    }

    target = properties["target"]
    assert set(target["required"]) >= {"short_edge", "aspect_ratio"}
    assert set(target["properties"]) >= {
        "short_edge",
        "aspect_ratio",
        "duration_seconds",
    }

    audio_flow_shift = properties["audio_flow_shift"]
    numeric_branches = [
        branch
        for branch in audio_flow_shift["anyOf"]
        if branch.get("type") in {"number", "integer"}
    ]
    assert numeric_branches[0]["exclusiveMinimum"] == 0

    error_schema = _resolve_ref(
        document,
        operation["responses"]["400"]["content"]["application/json"]["schema"],
    )
    assert error_schema["properties"]["detail"]["type"] == "string"


def test_generic_video_openapi_does_not_require_h3_task():
    app = FastAPI()
    app.include_router(video_api.router)
    video_api.configure_video_openapi(app, minimax_h3=False)

    document = app.openapi()
    schema = document["paths"]["/v1/videos"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert "task" not in schema.get("required", [])
