"""The HTTP JSON contract gate: `proto/sglang/api/v1` is the schema for both
the Rust and the Python server, so a `/generate` body the Python dataclass
parse admitted but the schema rejects is refused here with the schema's own
error text. The generated decoder is the single source of that judgment.
"""

from typing import Any, Optional

from sglang.api.v1.api_types import GenerateRequest, JsonContractError


def generate_contract_error(body: Any) -> Optional[str]:
    """The contract violation in a parsed `/generate` JSON body, or None."""
    # GenerateReqInput's PlainValidator already walked input_ids in a C loop
    # (array("q")), so the generated decoder's per-token loop only adds the
    # bool-element case. Re-walking 128K tokens here costs ~25ms per request.
    # Precondition: the schema must not grow input_ids length/range or
    # cross-field constraints, which only this decoder would enforce.
    if isinstance(body, dict) and "input_ids" in body:
        body = {k: v for k, v in body.items() if k != "input_ids"}
    try:
        GenerateRequest.from_json_value(body)
    except JsonContractError as e:
        return str(e)
    return None
