"""Engram must hash global predecessors before HCU CP shards the queries."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.engram import EngramHasher
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4 as model_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def _hasher():
    obj = EngramHasher.__new__(EngramHasher)
    torch.nn.Module.__init__(obj)
    obj.max_ngram_size = 4
    obj.history = torch.zeros((4, 3), dtype=torch.int32)
    obj.pad_row = 3
    obj.pad_id = 0
    obj.image_token_id = None
    obj.token_map = torch.arange(128)
    obj.multipliers = torch.tensor([[1, 3, 5, 7]])
    obj.primes = torch.tensor([[[17], [19], [23]]])
    obj.offsets = torch.tensor([[0, 17, 36]])
    return obj


class _State:
    def __init__(self, residual):
        self.residual, self.pre = residual, None

    def with_residual(self, value):
        return _State(value)

    def materialized(self, config):
        return self


@pytest.mark.parametrize("hcu,cp", [(True, True), (False, True), (True, False)])
def test_engram_global_history_then_rank_local_hashes(hcu, cp):
    # Unequal request lengths and an execution-padding suffix, like a real P batch.
    lengths = [13, 8]
    live, padded = sum(lengths), 24
    ids = torch.cat((torch.arange(1, live + 1), torch.zeros(padded-live, dtype=torch.long)))
    batch = SimpleNamespace(
        input_ids=ids,
        positions=torch.tensor(list(range(13)) + list(range(8)) + [0]*3),
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.tensor([0, 2]),
        extend_seq_lens=torch.tensor(lengths),
        extend_seq_lens_cpu=lengths,
        extend_start_loc=torch.tensor([0, 13]),
        out_cache_loc=torch.cat((torch.arange(1, live+1), torch.zeros(3, dtype=torch.long))),
        engram_history=None,
        # Legacy HCU round-robin metadata deliberately has no total token count.
        attn_cp_metadata=SimpleNamespace(total_seq_lens=0 if hcu else padded),
    )
    reference = _hasher()
    expected = reference(ids, batch)
    cp_size = 8 if cp else 1
    for rank in range(cp_size):
        hasher = _hasher()
        captured = {}

        def embed(x, hashes, fb, *, cp_all_tokens):
            captured.update(hashes=hashes, cp_all_tokens=cp_all_tokens)
            return x

        embed.layer_hash_index = 0
        layer = SimpleNamespace(
            engram=embed, hc_cfg=None,
            forward_hc_pre_from_prev=lambda **kwargs: kwargs["state"],
        )
        model = SimpleNamespace(
            decoder_replay_tail=None,
            pp_group=SimpleNamespace(world_size=1), engram_hasher=hasher,
            late_layer_start=None, start_layer=0, end_layer=1, layers=[layer],
            config=SimpleNamespace(model_type="deepseek_v41", vision_n_layers=0),
        )
        local_ids = ids[rank::cp_size]
        with ExitStack() as stack:
            for name, value in {
                "_is_hcu": hcu, "_is_hip": False,
            }.items():
                stack.enter_context(patch.object(model_module, name, value))
            for name, value in {
                "is_cp_active": cp and not hcu,
                "dsa_use_prefill_cp": cp and hcu,
                "is_in_breakable_cuda_graph": False,
                "check_cuda_graph_backend": True,
                "get_parallel": SimpleNamespace(attn_cp_rank=rank, attn_cp_size=cp_size),
                "get_forward": SimpleNamespace(sp_active=False),
            }.items():
                stack.enter_context(patch.object(model_module, name, return_value=value))
            stack.enter_context(patch.object(model_module.mhc, "HcState", _State))
            model_module.DeepseekV4Model._forward_layers_hc_pre_from_prev(
                model, batch.positions[rank::cp_size],
                torch.zeros((len(local_ids), 2, 4)), batch,
                local_ids, local_ids, False, [], None,
            )
        torch.testing.assert_close(captured["hashes"], expected[rank::cp_size, 0])
        torch.testing.assert_close(hasher.history, reference.history)
        assert captured["cp_all_tokens"] == cp
