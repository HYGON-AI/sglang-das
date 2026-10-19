from transformers import CONFIG_MAPPING, PretrainedConfig
from transformers.models.glm4v.configuration_glm4v import Glm4vVisionConfig

from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    mamba2_state_dtype,
)
from sglang.srt.runtime_context import get_parallel


class Glm5NextVisionConfig(Glm4vVisionConfig):
    model_type = "glm5next_vision"


class Glm5VNextVisionConfig(Glm5NextVisionConfig):
    model_type = "glm5v_next_vision"


def _default_linear_attn_config():
    kda_layers = [
        0,
        1,
        2,
        4,
        5,
        6,
        8,
        9,
        10,
        12,
        13,
        14,
        16,
        17,
        18,
        20,
        21,
        22,
        24,
        25,
        26,
        28,
        29,
        30,
        32,
        33,
        34,
        36,
        37,
        38,
        40,
        41,
        42,
        44,
    ]
    return {
        "num_heads": 64,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "kda_layers": kda_layers,
        "full_attn_layers": [i for i in range(45) if i not in kda_layers],
        "safe_gate": True,
    }


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    sub_configs = {"vision_config": Glm5NextVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    _NO_PROMOTE = frozenset(
        {
            "model_type",
            "architectures",
            "text_config",
            "vision_config",
            "image_token_id",
            "video_token_id",
            "image_start_token_id",
            "image_end_token_id",
            "video_start_token_id",
            "video_end_token_id",
        }
    )

    def __init__(
        self,
        model_type="glm5_next",
        vocab_size=154880,
        hidden_size=4096,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=45,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=256,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        rope_scaling=None,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        mla_nope=False,
        n_group=1,
        topk_group=1,
        n_routed_experts=288,
        n_shared_experts=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        num_experts_per_tok=8,
        first_k_dense_replace=3,
        moe_layer_freq=1,
        num_nextn_predict_layers=1,
        linear_attn_config=None,
        linear_allow_neg_eigval=False,
        mhc=True,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        hc_post_mult_value=2.0,
        swiglu_limit=10.0,
        index_head_dim=128,
        index_topk=2048,
        index_n_heads=32,
        index_dsa_use_layernorm=True,
        disable_nsa=False,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=154820,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        architectures=None,
        text_config=None,
        vision_config=None,
        image_token_id=151363,
        video_token_id=151364,
        image_start_token_id=151339,
        image_end_token_id=151340,
        video_start_token_id=151341,
        video_end_token_id=151342,
        **kwargs,
    ):
        if architectures is not None:
            arch_aliases = {
                "Glm5vNextForCausalLM": "Glm5NextForCausalLM",
                "Glm5vNextForConditionalGeneration": "Glm5NextForConditionalGeneration",
            }
            architectures = [arch_aliases.get(arch, arch) for arch in architectures]

        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_nope = mla_nope
        self.mla = True

        self.n_group = n_group
        self.topk_group = topk_group
        self.n_routed_experts = n_routed_experts
        self.num_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_shared_experts = n_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.linear_attn_config = dict(
            linear_attn_config or _default_linear_attn_config()
        )
        kda_layers = list(self.linear_attn_config.get("kda_layers", []))
        full_attn_layers = self.linear_attn_config.get("full_attn_layers")
        if full_attn_layers is None:
            full_attn_layers = [
                i for i in range(num_hidden_layers) if i not in set(kda_layers)
            ]
            self.linear_attn_config["full_attn_layers"] = full_attn_layers

        kda_set = set(kda_layers)
        full_attn_set = set(full_attn_layers)
        all_layers = set(range(num_hidden_layers))
        if len(kda_set) != len(kda_layers) or len(full_attn_set) != len(
            full_attn_layers
        ):
            raise ValueError("Attention layer lists must not contain duplicates.")
        if kda_set & full_attn_set:
            raise ValueError("KDA layers and full-attention layers must not overlap.")
        if kda_set | full_attn_set != all_layers:
            missing = sorted(all_layers - (kda_set | full_attn_set))
            extra = sorted((kda_set | full_attn_set) - all_layers)
            raise ValueError(
                "KDA and full-attention layers must cover the model exactly; "
                f"missing={missing}, out_of_range={extra}."
            )
        self.linear_allow_neg_eigval = linear_allow_neg_eigval

        self.mhc = mhc
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.hc_post_mult_value = hc_post_mult_value
        self.swiglu_limit = swiglu_limit

        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_dsa_use_layernorm = index_dsa_use_layernorm
        self.disable_nsa = disable_nsa

        is_vlm = text_config is not None or vision_config is not None
        if is_vlm:
            if text_config is not None:
                text_conf_cls = self.sub_configs["text_config"]
                if isinstance(text_config, text_conf_cls):
                    self.text_config = text_config
                elif isinstance(text_config, PretrainedConfig):
                    self.text_config = text_conf_cls(**text_config.to_dict())
                else:
                    self.text_config = text_conf_cls(**text_config)

                for key, value in self.text_config.__dict__.items():
                    if key.startswith("_") or key in self._NO_PROMOTE:
                        continue
                    setattr(self, key, value)

            if isinstance(vision_config, dict):
                self.vision_config = self.sub_configs["vision_config"](**vision_config)
            else:
                self.vision_config = vision_config

            self.image_token_id = image_token_id
            self.video_token_id = video_token_id
            self.image_start_token_id = image_start_token_id
            self.image_end_token_id = image_end_token_id
            self.video_start_token_id = video_start_token_id
            self.video_end_token_id = video_end_token_id

        self.initializer_range = initializer_range
        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        if not hasattr(self, "rope_scaling"):
            self.rope_scaling = rope_scaling
        rope_parameters = getattr(self, "rope_parameters", None)
        if rope_parameters is None:
            if self.rope_scaling is None:
                rope_parameters = {
                    "rope_theta": self.rope_theta,
                    "rope_type": "default",
                }
            else:
                rope_parameters = dict(self.rope_scaling)
                rope_parameters.setdefault("rope_theta", self.rope_theta)
                rope_parameters.setdefault(
                    "rope_type", rope_parameters.get("type", "default")
                )
            self.rope_parameters = rope_parameters
        if self.rope_scaling is None:
            self.rope_scaling = self.rope_parameters
        self.architectures = architectures or ["Glm5NextForCausalLM"]

    @property
    def is_mla(self):
        return True

    @property
    def is_moe(self):
        return self.n_routed_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return len(self.linear_layer_ids) > 0

    def is_kda_layer(self, layer_idx: int):
        if self.linear_attn_config is None:
            return False
        return layer_idx in set(self.linear_attn_config.get("kda_layers", []))

    @property
    def linear_layer_ids(self):
        if self.linear_attn_config is None:
            return []
        layers = self.linear_attn_config.get("kda_layers")
        if layers is not None:
            return list(layers)
        return []

    @property
    def full_attention_layer_ids(self):
        if self.linear_attn_config is None:
            return list(range(self.num_hidden_layers))
        layers = self.linear_attn_config.get("full_attn_layers")
        if layers is not None:
            return list(layers)
        return [i for i in range(self.num_hidden_layers) if not self.is_kda_layer(i)]

    @property
    def mamba2_cache_params(self) -> KimiLinearCacheParams:
        from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp

        head_shard_size = (
            get_parallel().attn_cp_size
            if is_dsa_enable_prefill_cp()
            else get_parallel().attn_tp_size
        )

        shape = KimiLinearStateShape.create(
            tp_world_size=head_shard_size,
            num_heads=self.linear_attn_config["num_heads"],
            head_dim=self.linear_attn_config["head_dim"],
            num_k_heads=self.linear_attn_config.get(
                "num_k_heads", self.linear_attn_config["num_heads"]
            ),
            head_k_dim=self.linear_attn_config.get(
                "head_k_dim", self.linear_attn_config["head_dim"]
            ),
            conv_kernel_size=self.linear_attn_config.get("short_conv_kernel_size", 4),
        )
        return KimiLinearCacheParams(
            shape=shape,
            layers=self.linear_layer_ids,
            dtype=mamba2_state_dtype(self),
        )


class Glm5NextTextConfig(Glm5NextConfig):
    model_type = "glm5next_text"

    def __init__(self, model_type="glm5next_text", **kwargs):
        super().__init__(model_type=model_type, **kwargs)


Glm5NextConfig.sub_configs["text_config"] = Glm5NextTextConfig


class Glm5NextTextUnderscoreConfig(Glm5NextTextConfig):
    model_type = "glm5_next_text"

    def __init__(self, model_type="glm5_next_text", **kwargs):
        super().__init__(model_type=model_type, **kwargs)


class Glm5VNextConfig(Glm5NextConfig):
    model_type = "glm5v_next"
    sub_configs = {
        "vision_config": Glm5VNextVisionConfig,
        "text_config": Glm5NextTextUnderscoreConfig,
    }

    def __init__(self, model_type="glm5v_next", **kwargs):
        super().__init__(model_type=model_type, **kwargs)


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=Glm5NextConfig,
        backend_class_name="sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend",
        arch_names=["Glm5NextForCausalLM", "Glm5NextForConditionalGeneration"],
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
        support_mamba_cache_extra_buffer=True,
    )
)

try:
    CONFIG_MAPPING.register("glm5_next", Glm5NextConfig)
except Exception:
    CONFIG_MAPPING._extra_content["glm5_next"] = Glm5NextConfig

for _model_type, _config_cls in (
    ("glm5next_text", Glm5NextTextConfig),
    ("glm5_next_text", Glm5NextTextUnderscoreConfig),
    ("glm5next_vision", Glm5NextVisionConfig),
    ("glm5v_next", Glm5VNextConfig),
    ("glm5v_next_vision", Glm5VNextVisionConfig),
):
    try:
        CONFIG_MAPPING.register(_model_type, _config_cls)
    except Exception:
        CONFIG_MAPPING._extra_content[_model_type] = _config_cls
