# SPDX-License-Identifier: Apache-2.0
"""Shared GLM-V vision encoder helpers."""

import math
from typing import List

import torch

from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model
from sglang.srt.utils import get_int_env_var


class GlmVisualEncoderMixin:
    """ViT feature extraction shared by GLM-V conditional-generation models."""

    def _run_visual_chunked(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        def _run(px: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.visual, px, g.tolist(), rope_type="rope_3d"
                )
            return self.visual(px, grid_thw=g)

        max_patches = get_int_env_var("SGLANG_VLM_MAX_PATCHES_PER_VIT", 0)
        max_images = get_int_env_var("SGLANG_VLM_MAX_IMAGES_PER_VIT", 0)

        if max_patches <= 0 and max_images <= 0:
            return _run(pixel_values, grid_thw)

        patches_per = [int(math.prod(g)) for g in grid_thw.tolist()]
        n = len(patches_per)
        cum = [0]
        for patches in patches_per:
            cum.append(cum[-1] + patches)
        assert pixel_values.size(0) == cum[-1], (pixel_values.size(0), cum[-1])

        outs = []
        start = 0
        while start < n:
            end, packed_patches, packed_images = start, 0, 0
            while end < n:
                next_patches = patches_per[end]
                if max_patches > 0 and packed_patches + next_patches > max_patches:
                    break
                if max_images > 0 and packed_images + 1 > max_images:
                    break
                packed_patches += next_patches
                packed_images += 1
                end += 1
            if end == start:
                end = start + 1
            outs.append(_run(pixel_values[cum[start] : cum[end]], grid_thw[start:end]))
            start = end

        return torch.cat(outs, dim=0)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if getattr(self, "visual", None) is None:
            raise RuntimeError(
                "GLM visual encoder is not initialized; multimodal embeddings must "
                "be provided by the disaggregated encoder before prefill."
            )
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        return self._run_visual_chunked(pixel_values, image_grid_thw)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        if getattr(self, "visual", None) is None:
            raise RuntimeError(
                "GLM visual encoder is not initialized; multimodal embeddings must "
                "be provided by the disaggregated encoder before prefill."
            )
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)

        temp_frames_hw = []
        for t, h, w in video_grid_thw:
            repeated_row = (
                torch.tensor([1, h.item(), w.item()]).unsqueeze(0).repeat(t, 1)
            )
            temp_frames_hw.append(repeated_row)
        flattened_video_grid_thw = torch.cat(temp_frames_hw, dim=0)

        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        return self._run_visual_chunked(pixel_values, flattened_video_grid_thw)
