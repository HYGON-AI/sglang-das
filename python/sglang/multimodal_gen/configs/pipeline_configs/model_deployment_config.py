"""
ModelDeploymentConfig provides model-specific config on how to deploy a model optimally

"""

from dataclasses import dataclass
from typing import Literal

OffloadComponentName = Literal["dit", "text_encoder", "image_encoder", "vae"]


@dataclass(frozen=True)
class ModelDeploymentConfig:
    auto_dit_layerwise_offload: bool = False
    # if the available memory is bigger than this value, keep dit resident instead of apply layerwise-offload
    auto_dit_layerwise_offload_high_memory_disable_gb: float | None = None
    auto_disable_component_offload_min_available_memory_gb: float | None = None
    # keep this explicit because large encoders can OOM even when DiT fits resident
    auto_disable_component_offload_components: tuple[OffloadComponentName, ...] = (
        "dit",
        "text_encoder",
        "image_encoder",
    )
    fsdp_auto_min_available_memory_gb: float | None = None
    fsdp_auto_requires_cfg: bool = True
    fsdp_auto_requires_default_parallelism: bool = True
    # Models with strict numerical contracts can keep speed mode from
    # implicitly enabling torch.compile.
    speed_mode_enable_torch_compile_by_default: bool = True
    keep_resident_min_available_gb: float | None = None
    keep_resident_components: tuple[OffloadComponentName, ...] = ()
    auto_enable_cfg_parallel: bool = True
    supports_cfg_parallel: bool = True
