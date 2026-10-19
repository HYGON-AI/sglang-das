"""Unit tests for DetokenizerManager incremental-decoding offsets.

Pure CPU: builds the manager without __init__ (no IPC, no real tokenizer) and
drives _decode_batch_token_id_output directly with a byte-level stub tokenizer.

Covers the U+FFFD commit gate: "decoded tail ends in U+FFFD" is only a proxy
for "the trailing bytes are an incomplete character". A byte-fallback
vocabulary has tokens that decode to U+FFFD on their own, so the old gate
could stay shut for a whole request (no streamed text, O(n^2) re-decode).

The fix commits (a) at the latest clean token boundary within the last
MAX_UTF8_TAIL_TOKENS tokens, and (b) after MAX_STALLED_DECODE_STEPS frozen
steps when no clean boundary exists at all. Incomplete characters are still
held back until their bytes arrive.
"""

import unittest

import pytest

from sglang.srt.managers.detokenizer_manager import (
    MAX_STALLED_DECODE_STEPS,
    DetokenizerManager,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Lone UTF-8 continuation byte: never decodes to a character on its own, so a
# byte-fallback vocabulary renders it as U+FFFD no matter what follows.
STRAY_BYTE = 0x80
# Three-byte UTF-8 sequence for U+4E2D, one byte per token.
CJK_BYTES = (0xE4, 0xB8, 0xAD)
TEXT_TOKEN = 256


class _ByteFallbackTokenizer:
    """Ids 0-255 are raw bytes, 256+ are ASCII letters.

    Mirrors the part of a byte-fallback vocabulary these tests depend on:
    undecodable bytes render as U+FFFD, and a multi-byte character only
    appears once all of its bytes have arrived.
    """

    is_fast = True

    def batch_decode(self, ids_list, skip_special_tokens=True, **kwargs):
        return [self._decode(ids) for ids in ids_list]

    @staticmethod
    def _decode(ids):
        buf = bytearray()
        for i in ids:
            if i < 256:
                buf.append(i)
            else:
                buf.append(ord("a") + (i - TEXT_TOKEN) % 26)
        return buf.decode("utf-8", errors="replace")


def _make_manager():
    m = DetokenizerManager.__new__(DetokenizerManager)
    m.tokenizer = _ByteFallbackTokenizer()
    m.vocab_size = None
    m.decode_status = {}
    m.disable_tokenizer_batch_decode = False
    m.is_tool_call_parser_gpt_oss = False
    return m


class _Recv:
    """The subset of BatchTokenIDOutput that the decode path reads."""

    def __init__(self, rid, decode_ids, read_offset):
        self.rids = [rid]
        self.finished_reasons = [None]
        self.decoded_texts = [""]
        self.decode_ids = [list(decode_ids)]
        self.read_offsets = [read_offset]
        self.skip_special_tokens = [True]
        self.spaces_between_special_tokens = [True]
        self.no_stop_trim = [False]


def _stream(manager, rid, prompt, steps):
    """Drive one streaming request; return (decode status, emitted text).

    The scheduler sends the surrogate-context prompt tail together with the
    first output chunk, then one event per later chunk, so the prompt is
    context only and never part of the decoded output.
    """
    out = []

    def feed(decode_ids, read_offset):
        out.append(
            manager._decode_batch_token_id_output(_Recv(rid, decode_ids, read_offset))[
                0
            ]
        )

    feed(list(prompt) + list(steps[0]), len(prompt))
    for chunk in steps[1:]:
        feed(list(chunk), 0)
    return manager.decode_status[rid], "".join(out)


class TestIncrementalDecodeOffsets(unittest.TestCase):
    def test_incomplete_character_is_still_held_back(self):
        """A multi-byte character split across tokens must not be corrupted.

        Committing on every step would emit U+FFFD for the leading bytes and
        lose the character; the gate must hold until its last byte arrives.
        """
        manager = _make_manager()
        s, text = _stream(manager, "rid", [TEXT_TOKEN], [list(CJK_BYTES)])
        self.assertEqual(text, "中")
        self.assertEqual(s.get_decoded_text(), "中")

    def test_degenerate_run_emits_at_the_clean_boundary(self):
        """[STRAY, text] repeated: text flows from the first step.

        The old gate held everything because every step ended with U+FFFD.
        The look-back commit finds the clean boundary one token back and
        streams right away, without waiting for a stall cap.
        """
        manager = _make_manager()
        steps = [[STRAY_BYTE, TEXT_TOKEN]] * 12
        s, text = _stream(manager, "rid", [TEXT_TOKEN], steps)
        self.assertIn("a", text)
        # Each chunk is [STRAY, text]: its decode ends clean, so every chunk
        # streams through immediately - no stall cap needed, nothing held.
        self.assertEqual(text, "\ufffda" * 12)

    def test_pure_replacement_run_is_bounded(self):
        """A pure run of U+FFFD tokens must not freeze the offsets forever.

        No clean boundary exists, so after MAX_STALLED_DECODE_STEPS frozen
        steps the tail is committed: the client sees text periodically and
        the re-decoded window stays bounded instead of growing with output.
        """
        manager = _make_manager()
        steps = [[STRAY_BYTE]] * (3 * MAX_STALLED_DECODE_STEPS)
        s, text = _stream(manager, "rid", [TEXT_TOKEN], steps)
        self.assertGreater(len(text), 0, "no text ever reached the client")
        self.assertLessEqual(
            len(s.decode_ids) - s.surr_offset,
            MAX_STALLED_DECODE_STEPS + MAX_STALLED_DECODE_STEPS,
            "re-decoded window grew past one stall cap",
        )

    def test_stall_counter_resets_on_a_clean_step(self):
        """Only *consecutive* stalls may force a commit.

        Normal CJK traffic stalls briefly while each character's bytes arrive;
        if stalls accumulated across characters, a long response would be
        force-committed mid-character and corrupted.
        """
        manager = _make_manager()
        steps = [list(CJK_BYTES) + [TEXT_TOKEN]] * 10
        s, text = _stream(manager, "rid", [TEXT_TOKEN], steps)
        self.assertEqual(text.count("\ufffd"), 0, f"corrupted output: {text!r}")
        self.assertEqual(text, "中a" * 10)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
