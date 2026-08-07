# SPDX-License-Identifier: Apache-2.0
"""High-value task, partition, and public request admission contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    prequeue,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    partition_for_task,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import Backend

TARGET = {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 5.0,
}


def _ref_conditions(*, images: int = 0, videos: int = 0, audios: int = 0):
    conditions = [
        {
            "type": "image",
            "uri": f"file:///image-{index}.png",
            "role": "reference",
        }
        for index in range(images)
    ]
    conditions.extend(
        {
            "type": "video",
            "uri": f"file:///video-{index}.mp4",
            "role": "reference",
        }
        for index in range(videos)
    )
    conditions.extend(
        {
            "type": "audio",
            "uri": f"file:///audio-{index}.wav",
            "role": "reference",
        }
        for index in range(audios)
    )
    return conditions


@pytest.mark.parametrize(
    ("task", "conditions", "partition", "visual", "audio", "chains"),
    [
        ("t2va", [], "fl2va", [], [], []),
        (
            "fl2va",
            [
                {
                    "type": "image",
                    "uri": "file:///first.png",
                    "role": "keyframe",
                    "frame_index": 0,
                },
                {
                    "type": "image",
                    "uri": "file:///last.png",
                    "role": "keyframe",
                    "frame_index": -1,
                },
            ],
            "fl2va",
            [0, 1],
            [],
            ["image.target_canvas", "image.target_canvas"],
        ),
        (
            "ref2va",
            [
                {
                    "type": "image",
                    "uri": "file:///image.png",
                    "role": "reference",
                },
                {
                    "type": "video",
                    "uri": "file:///video.mp4",
                    "role": "reference",
                    "start_time_seconds": 12.5,
                },
                {
                    "type": "audio",
                    "uri": "file:///audio.wav",
                    "role": "reference",
                },
                {
                    "type": "video_audio",
                    "uri": "file:///av.mp4",
                    "role": "reference",
                },
            ],
            "ref2va",
            [0, 1, 3],
            [1, 2, 3],
            [
                "image.reference_preserve",
                "video.reference_preserve",
                "audio",
                "video_audio.reference_preserve",
            ],
        ),
    ],
)
def test_public_tasks_resolve_to_exact_partition_and_encoder_plan(
    task, conditions, partition, visual, audio, chains
):
    canonical = minimax_h3_validate_canonical_request(
        task=task,
        prompt="contract",
        conditions=conditions,
        target=TARGET,
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)

    assert partition_for_task(task) == partition
    assert plan.task == task
    assert plan.encoders["visual"] == visual
    assert plan.encoders["audio"] == audio
    assert [material.material_chain for material in plan.materials] == chains
    if task == "ref2va":
        assert plan.materials[1].start_time_seconds == 12.5
    assert plan.shape["frame_count"] == 124
    assert plan.shape["video_latent_t"] == 37


@pytest.mark.parametrize(
    ("partition", "tasks"),
    [("fl2va", ["t2va", "fl2va"]), ("ref2va", ["ref2va"])],
)
def test_loaded_weight_partition_admits_only_its_declared_tasks(partition, tasks):
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": partition,
                "tasks": tasks,
                "task_aliases": {},
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        }
    )

    assert [metadata.canonical_task(task) for task in tasks] == tasks
    rejected = "ref2va" if partition == "fl2va" else "t2va"
    with pytest.raises(ValueError):
        metadata.canonical_task(rejected)


def test_duration_admission_accepts_released_4_to_15_second_range():
    for duration in (4.0, 15.0):
        target = {**TARGET, "duration_seconds": duration}
        canonical = minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="duration contract",
            conditions=[],
            target=target,
            seed=0,
        )
        assert canonical["target"]["duration_seconds"] == duration

    for duration in (3.9, 15.1):
        target = {**TARGET, "duration_seconds": duration}
        with pytest.raises(ValueError, match=r"\[4, 15\]"):
            minimax_h3_validate_canonical_request(
                task="t2va",
                prompt="duration contract",
                conditions=[],
                target=target,
                seed=0,
            )


def test_ref2va_accepts_released_twelve_file_mixed_boundary():
    conditions = _ref_conditions(images=6, videos=3, audios=3)

    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="twelve-file boundary",
        conditions=conditions,
        target=TARGET,
        seed=0,
    )

    assert len(canonical["conditions"]) == 12


@pytest.mark.parametrize(
    ("conditions", "message"),
    [
        (_ref_conditions(images=10), "at most 9 image files"),
        (_ref_conditions(images=1, videos=4), "at most 3 video files"),
        (_ref_conditions(images=1, audios=4), "at most 3 standalone audio files"),
        (_ref_conditions(images=7, videos=3, audios=3), "at most 12 entries"),
        (_ref_conditions(audios=3), "cannot contain audio as the sole input"),
    ],
)
def test_ref2va_rejects_released_file_count_and_audio_only_violations(
    conditions, message
):
    with pytest.raises(ValueError, match=message):
        minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="invalid reference boundary",
            conditions=conditions,
            target=TARGET,
            seed=0,
        )


def _duration_plan_and_facts(entries):
    materials = []
    facts = {}
    for index, (condition_type, duration) in enumerate(entries):
        materials.append(
            SimpleNamespace(
                condition_index=index,
                condition_type=condition_type,
            )
        )
        key = (
            "audio_duration_seconds"
            if condition_type == "audio"
            else "video_duration_seconds"
        )
        facts[index] = {key: duration}
    return SimpleNamespace(task="ref2va", materials=materials), facts


def test_ref2va_accepts_released_per_clip_and_total_duration_boundaries():
    plan, facts = _duration_plan_and_facts([("video", 5.0)] * 3 + [("audio", 5.0)] * 3)

    prequeue._validate_ref2va_reference_durations(plan, facts)


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([("video", 1.9)], r"video duration must be in \[2, 15\]"),
        ([("audio", 15.1)], r"audio duration must be in \[2, 15\]"),
        ([("video", 8.0), ("video_audio", 8.0)], "video total duration"),
        ([("audio", 8.0), ("audio", 8.0)], "audio total duration"),
    ],
)
def test_ref2va_rejects_released_reference_duration_violations(entries, message):
    plan, facts = _duration_plan_and_facts(entries)

    with pytest.raises(ValueError, match=message):
        prequeue._validate_ref2va_reference_durations(plan, facts)


def test_ref2va_prequeue_applies_total_duration_gate_before_queue(monkeypatch):
    conditions = _ref_conditions(images=1, videos=2)
    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="prequeue duration gate",
        conditions=conditions,
        target=TARGET,
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)
    probe_facts = {
        0: {
            "condition_type": "image",
            "display_width": 1024,
            "display_height": 1024,
        },
        1: {
            "condition_type": "video",
            "video_duration_seconds": 8.0,
            "display_width": 1280,
            "display_height": 720,
        },
        2: {
            "condition_type": "video",
            "video_duration_seconds": 8.0,
            "display_width": 1280,
            "display_height": 720,
        },
    }
    batch = SimpleNamespace(extra={})
    monkeypatch.setattr(prequeue, "minimax_h3_plan_from_batch", lambda _batch: plan)
    monkeypatch.setattr(
        prequeue,
        "minimax_h3_probe_material",
        lambda _batch, _uri, *, condition_type, condition_index: probe_facts[
            condition_index
        ],
    )

    with pytest.raises(ValueError, match="video total duration"):
        prequeue.minimax_h3_prepare_for_queue(batch)


def test_video_adapter_lowers_only_native_fields_and_rejects_cfg():
    request = VideoGenerationsRequest(
        prompt="contract",
        task="t2va",
        conditions=[],
        target=TARGET,
        flow_shift=8.0,
        audio_flow_shift=2.0,
        quality="high",
        imgvid_cond_noise_aug_for_inference=0.75,
        audio_cond_noise_aug_for_inference=0.5,
    )
    generic = {
        "prompt": request.prompt,
        "seed": request.seed,
        "flow_shift": request.flow_shift,
    }

    lowered = MiniMaxH3SamplingParams.lower_video_request_kwargs(request, generic)
    assert lowered == {
        "prompt": "contract",
        "seed": request.seed,
        "task": "t2va",
        "conditions": [],
        "target": TARGET,
        "flow_shift": 8.0,
        "audio_flow_shift": 2.0,
        "quality": "high",
        "imgvid_cond_noise_aug_for_inference": 0.75,
        "audio_cond_noise_aug_for_inference": 0.5,
    }

    with pytest.raises(ValueError):
        MiniMaxH3SamplingParams.lower_video_request_kwargs(
            request, {**generic, "guidance_scale": 7.5}
        )


class _HopperCapability:
    def to_int(self) -> int:
        return 90


def _quality_server_args():
    return SimpleNamespace(
        attention_backend=None,
        model_variant="fl2va",
        num_gpus=4,
        backend=Backend.AUTO,
        component_attention_backends={},
        enable_breakable_cuda_graph=False,
        enable_torch_compile=False,
        is_dit_layerwise_offload_selected=False,
        performance_mode="speed",
        quantization=None,
        regional_compile=False,
        ring_degree=1,
        sp_degree=4,
        tp_size=1,
        ulysses_degree=4,
        use_fsdp_inference=False,
    )


def test_quality_admission_fails_closed_outside_validated_request():
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": "fl2va",
                "tasks": ["t2va", "fl2va"],
                "task_aliases": {},
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        }
    )
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="quality",
        conditions=[],
        target=TARGET,
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(task="t2va", quality="high"),
        num_inference_steps=50,
        is_warmup=False,
    )
    stage = MiniMaxH3PartitionAdmissionStage(metadata)
    config = MiniMaxH3PipelineConfig()
    server_args = _quality_server_args()
    server_args.pipeline_config = config

    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "get_device_name", return_value="NVIDIA H200"),
        patch.object(
            current_platform,
            "get_device_capability",
            return_value=_HopperCapability(),
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
            "minimax_h3.release_metadata.minimax_h3_plan_from_batch",
            return_value=plan,
        ),
    ):
        assert stage.forward(batch, server_args) is batch
        batch.num_inference_steps = 40
        with pytest.raises(ValueError, match="validated only"):
            stage.forward(batch, server_args)

    batch.sampling_params.quality = "lossless"
    batch.num_inference_steps = 50
    server_args.attention_backend = "sage_attn"
    with pytest.raises(ValueError, match="does not support SageAttention"):
        stage.forward(batch, server_args)

    batch.sampling_params.quality = "unsupported"
    server_args.attention_backend = None
    with pytest.raises(ValueError, match="unsupported MiniMax-H3 quality profile"):
        stage.forward(batch, server_args)
