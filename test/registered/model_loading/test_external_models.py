# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import sglang as sgl
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
    register_hcu_ci,
)
from sglang.test.test_utils import CustomTestCase

# HCU BW1100 validated on 10.16.1.66/dxl-sglang: local Qwen2-VL external model path passed three runs.
register_hcu_ci(
    est_time=120,
    suite="stage-b-test-1-hcu-small",
    disabled="HCU Full Enabled run 26941698027 failed; keep disabled until BW1100 failure is fixed or revalidated.",
)
register_cuda_ci(est_time=29, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=45, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=203, suite="base-c-test-cpu")


class TestExternalModels(CustomTestCase):
    def test_external_model(self):
        envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set("sglang.test.external_models")
        envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.set("sglang.test.external_models")
        prompt = "Today is a sunny day and I like"
        model_path = "Qwen/Qwen2-VL-2B-Instruct"

        engine = sgl.Engine(
            model_path=model_path,
            cuda_graph_max_bs_decode=1,
            max_total_tokens=64,
            enable_multimodal=True,
        )
        out = engine.generate(prompt)["text"]
        engine.shutdown()

        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
