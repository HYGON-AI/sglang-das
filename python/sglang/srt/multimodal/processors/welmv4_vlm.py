import dataclasses
import re
import time
from typing import List, Union

from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.welmv4_vlm import WeLMV4VLMForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.utils import logger


class WeLMV4VLMImageProcessor(SGLangBaseProcessor):
    models = [WeLMV4VLMForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        tokenizer = _processor.tokenizer
        self.image_token = getattr(hf_config, "image_token", "<|image_pad|>")
        self.image_token_id = getattr(hf_config, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(
            self.vision_start_token
        )
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids(
            self.vision_end_token
        )

        # Required by base class for language_only EPD mode (encode_receiver
        # uses these to reconstruct MultimodalInputs from embeddings).
        self.IM_START_TOKEN_ID = self.vision_start_token_id
        self.IM_END_TOKEN_ID = self.vision_end_token_id
        self.IM_TOKEN_ID = self.image_token_id

        self.mm_tokens = MultimodalSpecialTokens(
            image_token=(
                f"{self.vision_start_token}{self.image_token}{self.vision_end_token}"
            ),
            image_token_id=self.image_token_id,
            image_token_regex=re.compile(
                rf"{re.escape(self.vision_start_token)}"
                rf"(?:{re.escape(self.image_token)})+"
                rf"{re.escape(self.vision_end_token)}"
            ),
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data,
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        del audio_data, args
        entry_time = time.perf_counter()
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        load_time = time.perf_counter()

        # The public chat template carries the full wrapped token. The WeLMV4 HF
        # processor expands a bare image token into the wrapped repeated form.
        processor_input_text = base_output.input_text.replace(
            self.mm_tokens.image_token, self.image_token
        )
        processor_output = dataclasses.replace(
            base_output, input_text=processor_input_text
        )
        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            processor_output, self.mm_tokens
        )
        process_time = time.perf_counter()
        rid = getattr(request_obj, "rid", "anonymous_rid")
        logger.debug(
            f"[WeLMV4VLMProcessor Perf] {rid=}, "
            f"load_time: {(load_time - entry_time) * 1000:.2f} ms, "
            f"process_time: {(process_time - load_time) * 1000:.2f} ms, "
            f"total_time: {(process_time - entry_time) * 1000:.2f} ms"
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.flatten().tolist(),
            mm_items=mm_items,
            im_start_id=self.vision_start_token_id,
            im_end_id=self.vision_end_token_id,
            im_token_id=self.mm_tokens.image_token_id,
        )
