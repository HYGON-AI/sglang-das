import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-c-test-cpu")


def test_breakable_cuda_graph_supports_logits_processor_output():
    backend = object.__new__(BreakableCudaGraphBackend)
    output = LogitsProcessorOutput(
        next_token_logits=torch.arange(15, dtype=torch.float32).view(3, 5),
        hidden_states=torch.ones((3, 2), dtype=torch.float32),
    )

    buffer = backend._alloc_full_buffer(output, size=8)
    assert isinstance(buffer, LogitsProcessorOutput)
    assert buffer.next_token_logits.shape == (8, 5)
    assert buffer.hidden_states.shape == (8, 2)
    assert backend._output_rows(output, cap=8) == 3

    backend._copy_output_to_buffer(output, buffer, num_tokens=3)
    sliced = backend._slice_output(buffer, num_tokens=3)
    torch.testing.assert_close(sliced.next_token_logits, output.next_token_logits)
    torch.testing.assert_close(sliced.hidden_states, output.hidden_states)
