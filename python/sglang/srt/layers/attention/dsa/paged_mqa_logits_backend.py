from __future__ import annotations

from enum import Enum

from sglang.srt.utils import is_hcu, is_hip, is_sm100_supported


class DSAPagedMQALogitsBackend(Enum):
    DEEPGEMM = "deepgemm"
    CUTEDSL = "cutedsl"
    AITER = "aiter"
    LIGHTOP = "lightop"

    def is_deepgemm(self) -> bool:
        return self == DSAPagedMQALogitsBackend.DEEPGEMM

    def is_cutedsl(self) -> bool:
        return self == DSAPagedMQALogitsBackend.CUTEDSL

    def is_aiter(self) -> bool:
        return self == DSAPagedMQALogitsBackend.AITER

    def is_lightop(self) -> bool:
        return self == DSAPagedMQALogitsBackend.LIGHTOP

    @staticmethod
    def resolve(value: str) -> DSAPagedMQALogitsBackend:
        # HCU reports itself as a HIP platform, but its DSA indexer uses the
        # LightOp page-64 layout rather than AITER's ROCm layout.
        if is_hcu():
            if value not in ("auto", "lightop"):
                raise ValueError(
                    f"dsa_paged_mqa_logits_backend={value!r} is not supported on "
                    "HCU; only 'lightop' is implemented."
                )
            return DSAPagedMQALogitsBackend.LIGHTOP

        if is_hip():
            if value not in ("auto", "aiter"):
                raise ValueError(
                    f"dsa_paged_mqa_logits_backend={value!r} is not supported on "
                    "ROCm; only 'aiter' is implemented."
                )
            return DSAPagedMQALogitsBackend.AITER

        if value == "auto" or value == "deepgemm":
            return DSAPagedMQALogitsBackend.DEEPGEMM
        if value == "aiter":
            raise ValueError("dsa_paged_mqa_logits_backend='aiter' requires ROCm.")
        if value == "lightop":
            raise ValueError("dsa_paged_mqa_logits_backend='lightop' requires HCU.")
        if value == "cutedsl":
            if not is_sm100_supported():
                raise ValueError(
                    "dsa_paged_mqa_logits_backend='cutedsl' requires SM100 (Blackwell)."
                )
            return DSAPagedMQALogitsBackend.CUTEDSL
        raise ValueError(f"Unknown dsa_paged_mqa_logits_backend: {value!r}")
