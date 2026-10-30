"""DeepSeek-V4.1 image preprocessing, preserving raw token IDs for Engram."""

import asyncio
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.multimodal.deepseek_v41_image_processing import (
    num_image_tokens,
    patchify_for_vit,
    resolve_patchify_backend,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.runtime_context import get_mm

logger = logging.getLogger(__name__)


class DeepseekV41ImageProcessor(BaseMultimodalProcessor):
    models = [DeepseekV4ForCausalLM]
    preserve_processor_input_ids = True
    prefer_tokenized_input = True
    gpu_image_decode = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        backend = get_mm().image_processor_backend
        self.preprocess_backend, self.cpu_patchify = resolve_patchify_backend(
            backend,
            get_mm().disable_fast_image_processor,
            envs.SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU.get(),
        )
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_token_id = hf_config.image_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=_processor.convert_ids_to_tokens(self.image_token_id),
            image_token_id=self.image_token_id,
        ).build(_processor)

    def preprocess_fingerprint_payload(self):
        payload = super().preprocess_fingerprint_payload()
        payload["dsv41_preprocess_backend"] = self.preprocess_backend
        return payload

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, *args, **kwargs
    ):
        base = await self.load_mm_data(
            input_text, image_data=image_data, multimodal_tokens=self.mm_tokens
        )
        ids = (
            input_text
            if isinstance(input_text, list)
            else self._processor.encode(input_text)
        )
        # Reject a placeholder/image mismatch before paying for patchify.
        self._check_placeholder_count(ids, len(base.images))
        prepared = [
            await asyncio.to_thread(
                patchify_for_vit,
                image,
                self.hf_config,
                backend=self.preprocess_backend,
                cpu_patchify=self.cpu_patchify,
            )
            for image in base.images
        ]
        tokens, offsets = self._expand_image_spans(
            ids, [num_image_tokens(lh, lw) for _, _, lh, lw in prepared]
        )
        items = []
        for (patches, metadata, _, _), offset in zip(prepared, offsets):
            if self.keep_mm_features_on_device:
                patches = patches.to(torch.device("cuda", self.server_args.base_gpu_id))
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=patches,
                    offsets=[offset],
                    model_specific_data=metadata,
                )
            )
        return MultimodalProcessorOutput(
            input_ids=tokens,
            mm_items=self._prepare_mm_items_for_transport(items),
            im_token_id=self.image_token_id,
        )

    def _check_placeholder_count(self, ids, num_images):
        num_placeholders = ids.count(self.image_token_id)
        if num_placeholders != num_images:
            raise ValueError(
                "Image placeholders and images must match: "
                f"{num_placeholders} placeholders != {num_images} images"
            )

    def _expand_image_spans(self, ids, counts):
        """Expand the i-th image placeholder in ``ids`` into ``counts[i]`` copies.

        Returns ``(tokens, offsets)`` with one inclusive ``(start, end)`` per image.
        """
        self._check_placeholder_count(ids, len(counts))
        remaining = iter(counts)
        tokens, offsets = [], []
        for token in ids:
            if token != self.image_token_id:
                tokens.append(token)
                continue
            count = next(remaining)
            start = len(tokens)
            tokens.extend([self.image_token_id] * count)
            offsets.append((start, start + count - 1))
        return tokens, offsets

    def get_mm_data(self, prompt, embeddings, **kwargs):
        """Rebuild EPD inputs on the language node from encoder-returned spans.

        The prompt carries one image placeholder per image; each grid
        ``(n_llm_h, n_llm_w)`` sets its span length, matching the IFB layout.
        Row-count checks against the received embeddings live in
        ``get_validated_mm_data``.
        """
        img_grid_thw = kwargs.get("img_grid_thw", None)
        if img_grid_thw is None:
            raise ValueError(
                "DeepSeek-V4.1 EPD assembly requires img_grid_thw metadata"
            )
        grids = torch.as_tensor(img_grid_thw).reshape(-1, 2).tolist()
        ids = (
            self._processor.encode(prompt) if isinstance(prompt, str) else list(prompt)
        )

        embedding = embeddings[Modality.IMAGE]
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)
        if embedding.ndim < 2:
            raise ValueError(
                "DeepSeek-V4.1 EPD image embeddings must have at least 2 dimensions"
            )
        if embedding.ndim != 2:
            embedding = embedding.reshape(-1, embedding.shape[-1])
        expected_width = self.hf_config.hidden_size
        if embedding.shape[1] != expected_width:
            raise ValueError(
                "DeepSeek-V4.1 EPD embedding width mismatch: "
                f"got {embedding.shape[1]}, expected {expected_width}"
            )

        tokens, offsets = self._expand_image_spans(
            ids, [num_image_tokens(int(lh), int(lw)) for lh, lw in grids]
        )
        items, consumed = [], 0
        for start, end in offsets:
            count = end - start + 1
            items.append(
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    offsets=[(start, end)],
                    precomputed_embeddings=embedding[consumed : consumed + count],
                )
            )
            consumed += count
        return MultimodalProcessorOutput(
            input_ids=tokens,
            mm_items=items,
            im_token_id=self.image_token_id,
        )
