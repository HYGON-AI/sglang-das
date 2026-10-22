"""CPU regression for SGLANG_DSV41_SKIP_MID_CHUNK_DECODER.

The gate is loaded from source with its collaborators mocked; the equivalence
check replays chunked prefill through the real late_layer_tail_layout. This does
not replace service-level accuracy validation.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SOURCE = Path(
    os.environ.get(
        "SWA_TEST_SOURCE", str(Path(__file__).resolve().parents[4] / "python" / "sglang")
    )
)
SWA_WINDOW = 128


def load(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name
        ).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    tree = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[name]


class _Mode:
    EXTEND = "extend"
    MIXED = "mixed"
    DECODE = "decode"


def make_gate():
    return load(
        SOURCE / "srt/layers/attention/deepseek_v4_backend.py",
        "_skips_late_layers",
        dict(SWA_WINDOW=SWA_WINDOW, ForwardMode=_Mode),
        "DeepseekV4AttnBackend",
    )


def decide(gate, orig, seq, *, enabled=True, mode=_Mode.EXTEND, extend=None):
    backend = SimpleNamespace(skip_mid_chunk_decoder=enabled)
    batch = SimpleNamespace(
        forward_mode=mode,
        orig_seq_lens_cpu=orig,
        extend_seq_lens_cpu=extend if extend is not None else [1] * len(seq),
        seq_lens_cpu=torch.tensor(seq),
    )
    return gate(backend, batch)


class TestSkipGate(CustomTestCase):
    gate = staticmethod(make_gate())

    def decide(self, orig, seq, **kw):
        return decide(self.gate, orig, seq, **kw)

    def test_every_request_mid_chunk(self):
        self.assertTrue(self.decide([100000], [32768]))
        self.assertTrue(self.decide([100000, 70000], [65536, 4096]))

    def test_window_boundary(self):
        self.assertTrue(self.decide([32768 + SWA_WINDOW], [32768]))
        self.assertFalse(self.decide([32768 + SWA_WINDOW - 1], [32768]))

    def test_any_final_or_near_final_chunk_keeps_tail(self):
        # Final chunk, and the chunk before a final chunk shorter than SWA_WINDOW.
        self.assertFalse(self.decide([100000], [100000]))
        self.assertFalse(self.decide([100000], [100000 - 50]))
        self.assertFalse(self.decide([100000, 4096], [32768, 4096]))

    def test_disabled_or_unsupported(self):
        self.assertFalse(self.decide([100000], [32768], enabled=False))
        self.assertFalse(self.decide([100000], [32768], mode=_Mode.MIXED))
        self.assertFalse(self.decide([100000], [32768], mode=_Mode.DECODE))
        self.assertFalse(self.decide(None, [32768]))
        self.assertFalse(self.decide([100000], [32768], extend=[1, 1]))


class TestEquivalence(CustomTestCase):
    """Every decoder row a later step reads is written as without the skip."""

    gate = staticmethod(make_gate())
    # main split the layout helpers; the tail-lens helper must be in scope.
    _layout_ns = dict(torch=torch)
    load(
        SOURCE / "kernels/ops/attention/dsv4_attn_metadata_kernels.py",
        "late_layer_tail_lens",
        _layout_ns,
    )
    layout = staticmethod(
        load(
            SOURCE / "kernels/ops/attention/dsv4_attn_metadata_kernels.py",
            "late_layer_tail_layout",
            _layout_ns,
        )
    )

    def replay(self, prompt_len, chunk, skip):
        """Chunked prefill of one request: decoder row position -> window floor it
        was computed under, for every row the late layers write."""
        written = {}
        start = 0
        while start < prompt_len:
            end = min(prompt_len, start + chunk)
            if skip and decide(self.gate, [prompt_len], [end], extend=[end - start]):
                start = end
                continue
            idx, _, floor = self.layout(
                extend_lens_cpu=[end - start],
                seq_lens_cpu=[end],
                tail_len=SWA_WINDOW,
                device=torch.device("cpu"),
            )
            for i, f in zip(idx.tolist(), floor.tolist()):
                written[start + i] = f
            start = end
        return written

    @staticmethod
    def live_rows(written, prompt_len):
        """Rows read after prefill: decode's window and the DSpark draft's context
        (both SWA_WINDOW back), plus what those rows read in their own window."""
        todo = list(range(max(0, prompt_len - SWA_WINDOW), prompt_len))
        live = set()
        while todo:
            pos = todo.pop()
            if pos in live:
                continue
            live.add(pos)
            if pos in written:
                todo.extend(range(max(written[pos], pos - SWA_WINDOW + 1), pos))
        return live

    def test_live_rows_unchanged(self):
        lens = (1, 127, 128, 129, 255, 4096, 32768 + 50, 32768 + 127, 32768 + 128)
        for prompt_len in lens + (32768 + 129, 100000, 131072):
            for chunk in (128, 1000, 4096, 32768):
                case = (prompt_len, chunk)
                full = self.replay(prompt_len, chunk, skip=False)
                skipped = self.replay(prompt_len, chunk, skip=True)
                live = self.live_rows(full, prompt_len)
                self.assertEqual(live, self.live_rows(skipped, prompt_len), case)
                for pos in live:
                    self.assertIn(pos, skipped, (*case, pos))
                    # Same chunk and floor: the same computation produced the row.
                    self.assertEqual(skipped[pos], full[pos], (*case, pos))

    def test_long_prompts_skip_mid_chunks(self):
        full = self.replay(100000, 32768, skip=False)
        skipped = self.replay(100000, 32768, skip=True)
        # Only the final chunk's tail runs.
        self.assertEqual(len(full), 4 * SWA_WINDOW)
        self.assertEqual(sorted(skipped), list(range(100000 - SWA_WINDOW, 100000)))


if __name__ == "__main__":
    unittest.main()
