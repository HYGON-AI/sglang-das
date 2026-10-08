"""Special token escape global state.

When the user passes ``--glm-special-token-escape-seed <int>`` at launch, every
``added_tokens_decoder`` entry of the loaded tokenizer is rewritten to
``f"{original}<{sha256(seed:original)[:8]}>"`` while keeping the token id
unchanged. Angle brackets are used (rather than square brackets) so the escaped
strings can be embedded as-is in EBNF / xgrammar string literals without
risking confusion with character-class syntax. This module exposes the
resulting ``original -> escaped`` mapping as a process-global singleton so that
reasoning parsers, tool-call parsers and the GLM decoding constraint can read
the escaped strings without each having to re-derive them from the seed.

When the feature is not enabled, ``mapping`` is empty and ``get(original)``
falls back to the original string, so all callers can use the same code path
regardless of whether escape is active.
"""

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EscapedSpecialTokens:
    mapping: dict = field(default_factory=dict)
    seed: Optional[int] = None
    enabled: bool = False

    def get(self, original: str) -> str:
        return self.mapping.get(original, original)


_GLOBAL: Optional[EscapedSpecialTokens] = None
_LOCK = threading.Lock()


def get_global_escaped_special_tokens() -> EscapedSpecialTokens:
    if _GLOBAL is None:
        return EscapedSpecialTokens()
    return _GLOBAL


def set_global_escaped_special_tokens(obj: EscapedSpecialTokens) -> None:
    global _GLOBAL
    with _LOCK:
        if (
            _GLOBAL is not None
            and _GLOBAL.seed == obj.seed
            and _GLOBAL.mapping == obj.mapping
        ):
            return
        _GLOBAL = obj


def reset_global_escaped_special_tokens() -> None:
    global _GLOBAL
    with _LOCK:
        _GLOBAL = None


def hash_suffix(seed: int, token: str) -> str:
    return hashlib.sha256(f"{seed}:{token}".encode("utf-8")).hexdigest()[:8]


def escape_token(seed: int, token: str) -> str:
    return f"{token}<{hash_suffix(seed, token)}>"


def escape_text(text):
    """Apply the process-global escape mapping to ``text`` (e.g. a chat template).

    No-op when escape is disabled, ``text`` is not a str, or it already contains
    escaped forms (idempotence). Longer originals first to avoid partial overlaps.
    Used to escape secondary template copies (e.g. processor.chat_template) that
    escape_tokenizer_special_tokens does not touch.
    """
    sp = get_global_escaped_special_tokens()
    if not sp.enabled or not sp.mapping or not isinstance(text, str):
        return text
    already = any(
        escaped != original and escaped in text
        for original, escaped in sp.mapping.items()
    )
    if already:
        return text
    for original in sorted(sp.mapping, key=len, reverse=True):
        escaped = sp.mapping[original]
        if escaped != original:
            text = text.replace(original, escaped)
    return text
