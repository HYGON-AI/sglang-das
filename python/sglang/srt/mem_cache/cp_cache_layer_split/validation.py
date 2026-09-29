"""Constraints for the V4 CUDA/HCU prefill LayerSplit implementation."""


def validate_cp_cache_layer_split(args, hf_config):
    from sglang.srt.environ import envs
    from sglang.srt.utils import is_cuda, is_hcu

    def require(condition, message):
        if not condition:
            raise ValueError("--enable-cp-cache-layer-split " + message)

    # The V4.1 config normalizer maps this architecture to the V4 runtime
    # class; model_type remains the reliable V4.1 discriminator.
    require(getattr(hf_config, "model_type", None) == "deepseek_v41", "requires DeepSeek V4.1")
    text_config = getattr(hf_config, "text_config", hf_config)
    ratios = set(text_config.compress_ratios)
    require(ratios <= {0, 1, 2}, f"supports compression ratios 0/1/2, got {ratios}")
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
    require(
        not args.enable_two_batch_overlap,
        "does not support two-batch overlap",
    )
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
    require(
        not args.enable_encoder_swa_bounded_replay,
        "does not support encoder SWA bounded replay",
    )
    require(envs.SGLANG_OPT_USE_COMPRESSOR_V2.get(), "requires Compressor V2")
    require(
        not envs.SGLANG_DISAGG_STAGING_BUFFER.get(),
        "does not support the separate disaggregation staging transport",
    )
    require(
        args.speculative_algorithm is None,
        "does not yet support speculative decoding",
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
