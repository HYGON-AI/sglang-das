"""In-place rewrite of every ``added_tokens_decoder`` entry on a HuggingFace
fast tokenizer so that ``content`` becomes ``f"{original}<{8-hex}>"`` while the
token id is preserved. Angle brackets are used (not square brackets) to keep
the escaped strings safe to embed verbatim in EBNF / xgrammar string literals.

The mutation is applied directly to ``tokenizer._tokenizer`` via the backend
``tokenizers.Tokenizer.to_str``/``from_str`` round-trip; this has been verified
to be equivalent to ``save_pretrained`` + edit + ``from_pretrained``.

Also rewrites the seven Python-level special-token attributes (eos/pad/bos/sep/
cls/unk/mask) and the chat_template, then installs a process-global
``EscapedSpecialTokens`` so downstream parsers and constraints pick up the
escaped strings.
"""

import json
import logging
import re
from typing import Optional

from sglang.srt.constrained.glm.escape import (
    EscapedSpecialTokens,
    escape_token,
    get_global_escaped_special_tokens,
    set_global_escaped_special_tokens,
)

logger = logging.getLogger(__name__)

_ESCAPE_FLAG_ATTR = "_sglang_special_token_escape_seed"


def escape_chat_template(chat_template: Optional[str], mapping=None) -> Optional[str]:
    """Rewrite every special token in ``chat_template`` to its escaped form.

    ``mapping`` defaults to the process-global mapping, so this is a no-op when
    escape is disabled. Replacement is longest-first to keep tokens that are
    prefixes of other tokens from being partially rewritten.

    Callers that overwrite ``tokenizer.chat_template`` after
    ``escape_tokenizer_special_tokens`` has run must pipe the new template
    through here. Assigning a bare template to an escaped tokenizer makes the
    template's special tokens miss ``added_tokens`` and get split into ordinary
    BPE pieces, which corrupts the prompt.

    Idempotent: an already-escaped template is returned unchanged. That matters
    because ``TemplateManager._resolve_hf_chat_template`` falls back to
    ``tokenizer.chat_template`` when there is no processor, and the tokenizer's
    own template *has* been escaped. Since ``original`` is a prefix of
    ``escaped``, a plain ``str.replace`` would append a second suffix
    (``<|user|><hash><hash>``) and break the ``added_tokens`` match just as
    badly as leaving it bare.
    """
    if not chat_template:
        return chat_template

    if mapping is None:
        sp = get_global_escaped_special_tokens()
        if not sp.enabled:
            return chat_template
        mapping = sp.mapping

    for original in sorted(mapping.keys(), key=len, reverse=True):
        escaped = mapping[original]
        if escaped == original:
            continue
        # Skip occurrences that already carry their suffix. A lambda is used as
        # the replacement so backslashes in the token are not read as regex
        # group references.
        suffix = escaped[len(original) :]
        chat_template = re.sub(
            re.escape(original) + r"(?!" + re.escape(suffix) + r")",
            lambda _m, _e=escaped: _e,
            chat_template,
        )
    return chat_template


def escape_tokenizer_special_tokens(tokenizer, seed: int) -> EscapedSpecialTokens:
    """Mutate ``tokenizer`` in place and install the process-global mapping.

    Callers must only invoke this when escape is enabled (seed is an int from
    user CLI). To disable escape, simply do not call this function — the global
    singleton then returns an identity ``EscapedSpecialTokens()``.
    """
    assert isinstance(seed, int), (
        f"seed must be an int when escape is enabled, got {type(seed).__name__}"
    )

    prev = getattr(tokenizer, _ESCAPE_FLAG_ATTR, None)
    if prev is not None:
        assert prev == seed, (
            f"escape_tokenizer_special_tokens called twice with different seeds "
            f"({prev} vs {seed}); refusing to remutate."
        )
        return get_global_escaped_special_tokens()

    if not hasattr(tokenizer, "backend_tokenizer"):
        raise RuntimeError(
            "--glm-special-token-escape-seed requires a fast tokenizer with a "
            "tokenizers backend; got "
            f"{type(tokenizer).__name__} which has no backend_tokenizer."
        )

    from tokenizers import Tokenizer

    backend = tokenizer.backend_tokenizer
    backend_json = json.loads(backend.to_str())
    added = backend_json.get("added_tokens", [])
    if not added:
        raise RuntimeError(
            "Tokenizer has no added_tokens; --glm-special-token-escape-seed has "
            "nothing to escape. Use a fast tokenizer that exposes special tokens "
            "via added_tokens_decoder."
        )

    originals = [t["content"] for t in added]
    mapping = {o: escape_token(seed, o) for o in originals}

    for entry in backend_json["added_tokens"]:
        entry["content"] = mapping[entry["content"]]
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(backend_json))

    for attr in (
        "eos_token",
        "pad_token",
        "bos_token",
        "sep_token",
        "cls_token",
        "unk_token",
        "mask_token",
        # Multimodal placeholder strings, if present. The HF processor reads
        # tokenizer.image_token/video_token at init; the already-built processor
        # instance is additionally patched in get_processor.
        "image_token",
        "video_token",
        "audio_token",
    ):
        val = getattr(tokenizer, attr, None)
        if val and val in mapping and mapping[val] != val:
            try:
                setattr(tokenizer, attr, mapping[val])
            except Exception as e:
                logger.warning("Failed to setattr %s on tokenizer: %s", attr, e)

    ct = getattr(tokenizer, "chat_template", None)
    if ct:
        tokenizer.chat_template = escape_chat_template(ct, mapping=mapping)

    try:
        addl = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        if addl and any(t in mapping for t in addl):
            tokenizer.additional_special_tokens = [mapping.get(t, t) for t in addl]
    except Exception:
        # Some tokenizer implementations expose this list as read-only.
        pass

    obj = EscapedSpecialTokens(mapping=mapping, seed=seed, enabled=True)
    set_global_escaped_special_tokens(obj)
    setattr(tokenizer, _ESCAPE_FLAG_ATTR, seed)

    return obj
