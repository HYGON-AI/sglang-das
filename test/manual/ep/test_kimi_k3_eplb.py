"""CPU regression tests for Kimi K3 and DSpark EPLB integration."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK3LinearForCausalLM,
    KimiK3MoE,
)
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2


class TestKimiK3EPLB(unittest.TestCase):
    def test_expert_location_config(self):
        for config in (
            SimpleNamespace(num_hidden_layers=60, num_experts=896, num_expert_group=8),
            SimpleNamespace(num_hidden_layers=60, n_routed_experts=896, n_group=8),
        ):
            with self.subTest(config=config):
                result = KimiK3LinearForCausalLM.get_model_config_for_expert_location(
                    config
                )
                self.assertEqual(result.num_layers, 60)
                self.assertEqual(result.num_logical_experts, 896)
                self.assertEqual(result.num_groups, 8)
                wrapped = (
                    KimiK3ForConditionalGeneration.get_model_config_for_expert_location(
                        SimpleNamespace(text_config=config)
                    )
                )
                self.assertEqual(wrapped, result)

    def test_live_expert_weights_and_scales(self):
        experts = torch.nn.Module()
        experts.num_local_experts = 2
        names = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
        for name in (*names, "correction_bias", "global_weight"):
            experts.register_parameter(
                name, torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
            )
        experts.register_parameter(
            "scalar", torch.nn.Parameter(torch.ones(()), requires_grad=False)
        )
        experts.global_weight._sglang_require_global_experts = True
        moe = SimpleNamespace(experts=experts)
        weights = KimiK3MoE.get_moe_weights(moe)
        self.assertEqual(
            [weight.data_ptr() for weight in weights],
            [getattr(experts, name).data_ptr() for name in names],
        )
        # Quantization may replace Parameters after module construction.
        experts.w13_weight = torch.nn.Parameter(torch.zeros(2, 3), requires_grad=False)
        self.assertEqual(
            KimiK3MoE.get_moe_weights(moe)[0].data_ptr(),
            experts.w13_weight.data_ptr(),
        )

    def test_dense_layers_are_excluded(self):
        moe = KimiK3MoE.__new__(KimiK3MoE)
        torch.nn.Module.__init__(moe)
        moe.experts = torch.nn.Module()
        moe.experts.num_local_experts = 2
        moe.experts.register_parameter(
            "w13_weight", torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
        )
        # Only the pipeline-local layer range is exposed.
        model = SimpleNamespace(
            layers=[
                SimpleNamespace(mlp=torch.nn.Linear(3, 3)),
                SimpleNamespace(mlp=moe),
                SimpleNamespace(mlp=torch.nn.Linear(3, 3)),
            ],
            start_layer=1,
            end_layer=3,
        )
        result = KimiK3LinearForCausalLM.routed_experts_weights_of_layer.fget(
            SimpleNamespace(model=model)
        )
        self.assertEqual(set(result), {1})
        self.assertEqual(result[1][0].data_ptr(), moe.experts.w13_weight.data_ptr())

    def test_verify_metrics_are_preserved(self):
        metrics = {"balancedness": 0.8}
        target = SimpleNamespace(
            logits_output=None,
            can_run_cuda_graph=True,
            expert_distribution_metrics=metrics,
        )
        executor = SimpleNamespace(
            target_worker=SimpleNamespace(
                forward_batch_generation=lambda **kwargs: target
            )
        )
        result = TargetVerifyExecutor._forward_prepared_verify(
            executor,
            batch=SimpleNamespace(),
            verify_input=SimpleNamespace(
                prepare_for_verify=lambda *args: (object(), None)
            ),
            seq_lens_cpu_backup=None,
            seq_lens_sum_backup=0,
        )
        self.assertIs(result.expert_distribution_metrics, metrics)

    def test_idle_metrics_are_preserved(self):
        worker = SimpleNamespace(device="cpu", verify_num_draft_tokens=8)
        for metrics in (None, {"balancedness": 0.8}):
            with self.subTest(metrics=metrics):
                result = DSparkWorkerV2._decode_idle_result(
                    worker, on_publish=None, expert_distribution_metrics=metrics
                )
                self.assertIs(result.expert_distribution_metrics, metrics)


if __name__ == "__main__":
    unittest.main()
