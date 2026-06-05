import os
import unittest

import sglang as sgl
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=45, suite="stage-b-test-1-gpu-small-amd")


# DCU BW1000 validated on 10.16.1.66/dxl-sglang: local Qwen2-VL external model path passed three runs.
register_dcu_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-small-dcu",
    disabled='DCU Full Enabled run 26941698027 failed; keep disabled until BW1000 failure is fixed or revalidated.',
)

class TestExternalModels(CustomTestCase):
    def test_external_model(self):
        envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set("sglang.test.external_models")
        envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.set("sglang.test.external_models")
        prompt = "Today is a sunny day and I like"
        model_path = os.environ.get(
            "SGLANG_TEST_EXTERNAL_MODEL_NAME",
            "Qwen/Qwen2-VL-2B-Instruct",
        )

        engine = sgl.Engine(
            model_path=model_path,
            cuda_graph_max_bs=1,
            max_total_tokens=64,
            enable_multimodal=True,
        )
        out = engine.generate(prompt)["text"]
        engine.shutdown()

        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
