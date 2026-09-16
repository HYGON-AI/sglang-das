"""Constraints for the V4 CUDA/HCU prefill LayerSplit implementation."""


def validate_cp_cache_layer_split(args, hf_config):
    from sglang.srt.environ import envs
    from sglang.srt.utils import is_cuda, is_hcu

    def require(condition, message):
        if not condition:
            raise ValueError("--enable-cp-cache-layer-split " + message)

    require(
        hf_config.architectures[0] == "DeepseekV4ForCausalLM", "requires DeepSeek V4"
    )
    require(
        not args.enable_dsa_cache_layer_split,
        "must not be combined with --enable-dsa-cache-layer-split",
    )
    require(
        args.disaggregation_mode == "prefill", "requires --disaggregation-mode prefill"
    )
    require(
        args.enable_prefill_cp
        and args.cp_strategy == "interleave"
        and args.attn_cp_size > 1,
        "requires --enable-prefill-cp --cp-strategy interleave --attn-cp-size > 1",
    )
    require(args.pp_size == 1, "requires --pp-size 1")
    require(is_cuda() or is_hcu(), "requires CUDA or HCU")
    require(
        args.disaggregation_transfer_backend == "mooncake",
        "requires the Mooncake transfer backend",
    )
    require(
        not args.enable_hisparse and not args.enable_hierarchical_cache,
        "does not yet support HiSparse or HiCache in this HCU port",
    )
    require(not args.prefill_only_disable_kv_cache, "requires the prefill KV cache")
    require(envs.SGLANG_OPT_USE_COMPRESSOR_V2.get(), "requires Compressor V2")
    require(
        not envs.SGLANG_DSV4_COMPRESS_RLC.get(),
        "does not support SGLANG_DSV4_COMPRESS_RLC",
    )
    require(
        not envs.SGLANG_ENABLE_CP_V2.get(),
        "requires the legacy interleave CP path in this HCU port",
    )
    require(
        not envs.SGLANG_DISAGG_STAGING_BUFFER.get(),
        "does not support the separate disaggregation staging transport",
    )
    require(
        args.speculative_algorithm in (None, "EAGLE", "DSPARK"),
        "supports EAGLE or DSPARK speculation only",
    )
    if args.speculative_algorithm == "EAGLE":
        require(args.speculative_eagle_topk == 1, "requires EAGLE topk=1")
    if args.speculative_algorithm == "DSPARK":
        require(
            bool(getattr(hf_config, "dspark_target_layer_ids", None)),
            "requires a DSPARK model configuration",
        )
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )

    require(not is_unified_kv_triton(), "does not support unified_kv_triton")
    from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
        use_dsv4_q8kv8_sparse_prefill,
    )

    require(
        not use_dsv4_q8kv8_sparse_prefill(args.dsv4_prefill_backend),
        "does not yet support the Q8 sparse-prefill backend in this port",
    )
    # Compacted pages have dynamic shapes and the staging state belongs to one
    # forward at a time. The normal eager loop preserves CP collective order.
    from sglang.srt.model_executor.cuda_graph_config import Backend

    require(
        args.cuda_graph_config.prefill.backend == Backend.DISABLED,
        "requires --cuda-graph-backend-prefill disabled",
    )
