import os

import pytest
import torch
from torch import nn

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    update_environment_variables,
)
from sglang.srt.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPinnedHostEmbedding,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.network import get_open_port
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="4-gpu-b200")

_TP_SIZE = 4


def _run_tp4_parity(local_rank: int, world_size: int, master_port: int) -> None:
    enable_symm_mem = (
        os.environ.get("SGLANG_TEST_QWEN4_PLE_SYMM_MEM", "1") != "0"
    )
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
        }
    )
    torch.cuda.set_device(local_rank)
    torch.set_default_device(f"cuda:{local_rank}")
    set_global_server_args_for_scheduler(
        ServerArgs(
            model_path="dummy",
            tp_size=world_size,
            disable_custom_all_reduce=True,
            enable_symm_mem=enable_symm_mem,
        )
    )
    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        backend="nccl",
        enable_symm_mem=enable_symm_mem,
    )

    try:
        for embedding_dim in (7, 257):
            baseline = VocabParallelEmbedding(
                17,
                embedding_dim,
                params_dtype=torch.bfloat16,
            )
            source = VocabParallelEmbedding(
                17,
                embedding_dim,
                params_dtype=torch.bfloat16,
            )
            source.register_buffer(
                "weight_scale",
                torch.ones((1,), dtype=torch.bfloat16, device="cuda"),
                persistent=True,
            )
            offloaded = Qwen4ExpPinnedHostEmbedding(source)

            full_weight = (
                torch.arange(17 * embedding_dim, dtype=torch.int64, device="cpu")
                .remainder(127)
                .reshape(17, embedding_dim)
                .to(torch.bfloat16)
            )
            baseline.weight_loader(baseline.weight, full_weight)
            offloaded.weight_loader(offloaded.weight, full_weight)

            ids = torch.tensor(
                [[0, 1, 4, 5, 8], [11, 12, 15, 16, 7]],
                dtype=torch.int64,
                device=f"cuda:{local_rank}",
            )
            expected = baseline(ids)
            actual = offloaded(ids)

            assert offloaded.weight.is_pinned()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        embedding_dim = 7
        ids = torch.tensor(
            [[0, 1, 4, 5, 8], [11, 12, 15, 16, 7]],
            dtype=torch.int64,
            device=f"cuda:{local_rank}",
        )
        scales = (
            torch.arange(1, 18, dtype=torch.bfloat16, device="cuda").reshape(
                17, 1
            )
            / 8
        )
        for weight_dtype in (torch.int8, torch.float8_e4m3fn):
            ple = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
            nn.Module.__init__(ple)
            ple.ngram_embedding = VocabParallelEmbedding(
                17,
                embedding_dim,
                params_dtype=weight_dtype,
                output_dtype=torch.bfloat16,
            )
            local_scales = torch.ones(
                (ple.ngram_embedding.num_embeddings_per_partition, 1),
                dtype=torch.bfloat16,
                device="cuda",
            )
            start = ple.ngram_embedding.shard_indices.org_vocab_start_index
            end = ple.ngram_embedding.shard_indices.org_vocab_end_index
            local_scales[: end - start].copy_(scales[start:end])
            ple.ngram_embedding.register_buffer(
                "weight_scale", local_scales, persistent=True
            )

            full_weight = (
                torch.arange(17 * embedding_dim, dtype=torch.int64, device="cuda")
                .remainder(31)
                .sub(15)
                .reshape(17, embedding_dim)
                .to(weight_dtype)
            )
            ple.ngram_embedding.weight_loader(
                ple.ngram_embedding.weight, full_weight
            )

            dequantized = full_weight.to(torch.bfloat16) * scales
            expected = dequantized.index_select(0, ids.flatten()).reshape(
                *ids.shape, embedding_dim
            )
            actual = ple._lookup_ngram_embeddings(ids)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.cuda.synchronize()
    finally:
        destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_qwen4_ple_pinned_embedding_tp4_bitwise_parity():
    if not torch.cuda.is_available() or torch.cuda.device_count() < _TP_SIZE:
        pytest.skip("This test requires four CUDA devices.")

    torch.multiprocessing.spawn(
        _run_tp4_parity,
        args=(_TP_SIZE, get_open_port()),
        nprocs=_TP_SIZE,
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
