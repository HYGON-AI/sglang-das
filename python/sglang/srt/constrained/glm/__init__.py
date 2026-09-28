import traceback

from .schema import generation_constraint as _generation_constraint
from .schema import get_special_token_config

__all__ = ["get_special_token_config", "generation_constraint"]


def generation_constraint(*args, **kwargs):
    return _generation_constraint(
        *args,
        **kwargs,
        accommodate_chat_template=True,
        allow_multiple_assistant_turns=False,
    )
