"""Response-mode policy for the EPD language-side prefill server."""

from sglang.srt.runtime_context import get_disagg


def epd_prefill_forces_non_streaming() -> bool:
    """Whether this server is an EPD prefill that must answer non-streaming.

    The PD router judges prefill failure by HTTP status alone. A streamed
    prefill reports a failure during multimodal assembly (encoder unreachable,
    embedding timeout) as HTTP 200 plus an SSE error event, so the router never
    aborts the paired decode, which waits for KV until the disaggregation
    timeout. Prefill output never reaches the client, so answering
    non-streaming turns those failures into a real error status and changes
    nothing else.
    """
    disagg = get_disagg()
    return disagg.disaggregation_mode == "prefill" and disagg.language_only


def drop_streaming_for_epd_prefill(request) -> None:
    """Make a chat/completions request non-streaming on an EPD prefill server."""
    # Local import: entrypoints.openai.protocol is heavy and this module is
    # also imported by the native HTTP server.
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        CompletionRequest,
    )

    if epd_prefill_forces_non_streaming() and isinstance(
        request, (ChatCompletionRequest, CompletionRequest)
    ):
        request.stream = False
        # OpenAI requests reject stream_options without stream=true.
        request.stream_options = None
