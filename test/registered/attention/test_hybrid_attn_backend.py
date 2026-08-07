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

from sglang.test.ci.ci_register import register_cuda_ci, register_hcu_ci

# HCU_CSV_CI_UNVERIFIED: Registered from sglang.csv CI coverage; not re-tested in this framework pass.
register_hcu_ci(
    est_time=120,
    suite="stage-b-test-1-hcu-small",
    nightly=False,
    disabled="HCU CSV CI placeholder: hybrid attention backend needs BW1100 validation before enabling.",
)

from sglang.test.server_fixtures.hybrid_attn_backend_fixture import (
    TestHybridAttnBackendBase,
)
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
)

# Hybrid attention backend tests (FA3 prefill + FlashInfer decode, requires SM 90+ / H100)
# Multiple test classes: base, MLA, TorchCompile, SpecDecode variants
register_cuda_ci(est_time=407, stage="extra-a", runner_config="1-gpu-large")


class TestHybridAttnBackendMLA(TestHybridAttnBackendBase):
    accuracy_threshold = 0.60
    model = DEFAULT_MODEL_NAME_FOR_TEST_MLA


class TestHybridAttnBackendTorchCompile(TestHybridAttnBackendBase):
    accuracy_threshold = 0.65
    extra_args = ["--enable-torch-compile"]


class TestHybridAttnBackendSpeculativeDecodingPrefillBackend(TestHybridAttnBackendBase):
    speculative_decode = True
    # This eagle test uses a very small model, so the accuracy is low.
    accuracy_threshold = 0.2
    extra_args = [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_EAGLE,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "2",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-attention-mode",
        "prefill",
    ]


class TestHybridAttnBackendSpeculativeDecodingDecodeBackend(TestHybridAttnBackendBase):
    speculative_decode = True
    # This eagle test uses a very small model, so the accuracy is low.
    accuracy_threshold = 0.2
    extra_args = [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_EAGLE,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "2",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-attention-mode",
        "decode",
    ]


if __name__ == "__main__":
    unittest.main()
