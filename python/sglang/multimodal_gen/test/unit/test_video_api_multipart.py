import json

import pytest
from fastapi import HTTPException

from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _merge_multipart_declared_values,
    _parse_form_extra_value,
    _parse_multipart_extra_body,
)


def test_parse_multipart_extra_body_preserves_minimax_h3_contract():
    conditions = [
        {
            "role": "keyframe",
            "type": "image",
            "uri": "file:///tmp/reference.jpg",
            "frame_index": 0,
        }
    ]
    target = {
        "short_edge": 768,
        "aspect_ratio": "auto",
        "duration_seconds": 4.0,
    }
    parsed = _parse_multipart_extra_body(
        json.dumps(
            {
                "task": "fl2va",
                "conditions": conditions,
                "target": target,
                "flow_shift": 12.0,
                "audio_flow_shift": 3.0,
            }
        )
    )

    assert parsed["task"] == "fl2va"
    assert parsed["conditions"] == conditions
    assert parsed["target"] == target
    assert parsed["flow_shift"] == 12.0
    assert parsed["audio_flow_shift"] == 3.0


@pytest.mark.parametrize("value", ["not-json", "[]", '"string"'])
def test_parse_multipart_extra_body_rejects_invalid_objects(value):
    with pytest.raises(HTTPException) as exc_info:
        _parse_multipart_extra_body(value)

    assert exc_info.value.status_code == 400


def test_multipart_explicit_form_fields_override_extra_body():
    merged = _merge_multipart_declared_values(
        {"seed": 7, "num_inference_steps": None},
        {"seed": 11, "num_inference_steps": 4},
        {"seed"},
    )

    assert merged == {"seed": 7, "num_inference_steps": 4}


def test_parse_form_extra_value_keeps_json_and_plain_strings():
    assert _parse_form_extra_value('{"enabled": true}') == {"enabled": True}
    assert _parse_form_extra_value("plain-text") == "plain-text"
