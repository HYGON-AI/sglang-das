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

from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    MoeA2ABackend,
    MoeRunnerBackend,
    get_deepep_config,
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
    get_tbo_token_distribution_threshold,
    initialize_moe_config,
    is_moe_input_scattered_across_dp_ranks,
    is_tbo_enabled,
    should_skip_mlp_all_reduce,
    should_skip_post_experts_all_reduce,
    should_use_dp_reduce_scatterv,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
    should_use_flashinfer_trtllm_moe,
)

__all__ = [
    "DeepEPMode",
    "MoeA2ABackend",
    "MoeRunner",
    "MoeRunnerConfig",
    "MoeRunnerBackend",
    "initialize_moe_config",
    "get_moe_a2a_backend",
    "get_moe_runner_backend",
    "get_deepep_mode",
    "should_skip_mlp_all_reduce",
    "should_skip_post_experts_all_reduce",
    "should_use_dp_reduce_scatterv",
    "should_use_flashinfer_trtllm_moe",
    "should_use_flashinfer_cutlass_moe_fp4_allgather",
    "is_moe_input_scattered_across_dp_ranks",
    "is_tbo_enabled",
    "get_tbo_token_distribution_threshold",
    "get_deepep_config",
]
