from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_bool_env_var, is_hcu

_is_hcu = is_hcu()
_use_fused_bailing_rms_quant = _is_hcu and get_bool_env_var(
    "SGLANG_USE_FUSED_BAILING_RMS_QUANT"
)
if _use_fused_bailing_rms_quant:
    from lightop.norm import rms_norm_per_token_fp8_quant


class HcuSkipNorm:
    """Leave normalization/residual addition to the HCU fused projection."""

    def __call__(self, hidden_states, residual=None, **kwargs):
        return hidden_states if residual is None else (hidden_states, residual)


class _HcuBailingRmsQuantNorm:
    """HCU bailing: stands in for a stage's RMSNorm and fuses the (residual add
    +) RMSNorm + per-token FP8 quant into one lightop kernel. Called like the
    norm, it returns the quantized ``(fp8, scale)`` pair where the norm returned
    its output; with a residual the kernel adds into it in place. A
    post-residual addition is not applied on this path."""

    def __init__(self, norm, forward_batch, *, first_layer_plain: bool):
        self._norm = norm
        self._forward_batch = forward_batch
        # The attention input of a stack's first layer keeps the plain norm.
        self._first_layer_plain = first_layer_plain
        self.weight = norm.weight
        self.variance_epsilon = norm.variance_epsilon

    def __call__(self, hidden_states, residual=None, post_residual_addition=None):
        if residual is None and self._first_layer_plain:
            return self._norm(hidden_states)
        sparse = getattr(self._forward_batch, "bailing_sparse_rms_quant_fusion", False)
        out_fp8, out_bs = rms_norm_per_token_fp8_quant(
            hidden_states,
            self._norm.weight,
            epsilon=self._norm.variance_epsilon,
            fp8type=0,
            residual=residual,
            update_input=residual is not None or sparse,
        )
        if sparse:
            # The sparse layer's gate reads the normalized hidden states the
            # kernel wrote back in place.
            self._forward_batch.bailing_sparse_norm_hidden_states = hidden_states
        if residual is None:
            return out_fp8, out_bs
        return (out_fp8, out_bs), residual


def hcu_bailing_stage_norm(stage_name: str, norm, forward_batch):
    """The norm a stage's steps run with under HCU bailing's fused quant: the
    attention input takes it without EP and DP, the FFN input when the bailing
    layer asks for it through ``forward_batch.rms_quant_flag``."""
    if stage_name == "attention":
        if get_parallel().ep_size != 1 or get_parallel().dp_size != 1:
            return norm
        return _HcuBailingRmsQuantNorm(norm, forward_batch, first_layer_plain=True)
    if getattr(forward_batch, "rms_quant_flag", False):
        return _HcuBailingRmsQuantNorm(norm, forward_batch, first_layer_plain=False)
    return norm
