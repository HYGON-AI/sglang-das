"""Read the resolved v0.5.18 configuration for the migrated HCU components."""

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.runtime_context import get_server_args


def get_glm5_next_runtime_args():
    return resolved_view(get_server_args())
