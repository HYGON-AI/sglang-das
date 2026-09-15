"""Normalize compact DSPARK exports without changing their checkpoint files."""

from copy import deepcopy


def normalize_dspark_config(config: dict):
    """Return a flat HF network config for a nested DSPARK export, else None."""
    network = config.get("transformer_layer_config")
    architectures = config.get("architectures") or []
    is_dspark = config.get("speculators_model_type") == "dspark" or any(
        name in ("DSparkDraftModel", "Qwen3DSparkModel") for name in architectures
    )
    if not is_dspark or network is None:
        return None
    if not isinstance(network, dict):
        raise ValueError("DSPARK transformer_layer_config must be an object.")
    model_type = network.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("DSPARK transformer_layer_config requires model_type.")
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size"):
        value = network.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"DSPARK transformer_layer_config requires positive {key}.")
    result = deepcopy(network)
    for key, value in config.items():
        if key in ("transformer_layer_config", "model_type"):
            continue
        if key in network and value is None:
            continue
        if key in network and value != network[key]:
            raise ValueError(f"Conflicting DSPARK network config field: {key}.")
        result[key] = deepcopy(value)
    result["model_type"] = model_type
    return result
