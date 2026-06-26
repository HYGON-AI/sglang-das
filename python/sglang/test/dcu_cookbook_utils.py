"""Helpers for DCU cookbook-style SGLang nightly tests."""

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.dcu_utils import (
    RED_DOT_IMAGE_DATA_URL,
    get_model_path,
    openai_base_url,
)
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import popen_launch_server


DCU_COOKBOOK_API_KEY = "sk-123456"

QWEN3_COOKBOOK_ENV = {
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_USE_FUSED_TOPK_SOFTMAX": "1",
    "SGLANG_USE_LIGHTOP": "1",
    "SGLANG_USE_CAUSAL_CONV1D": "1",
    "SGLANG_USE_AITER_LINEAR_ATTN": "1",
}

GLM51_COOKBOOK_ENV = {
    "NCCL_MIN_NCHANNELS": "16",
    "NCCL_MAX_NCHANNELS": "16",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "HSA_ENABLE_COREDUMP": "1",
    "USE_DCU_CUSTOM_ALLREDUCE": "1",
    "ALLREDUCE_STREAM_WITH_COMPUTE": "1",
    "HIP_KERNEL_EVENT_SYSTENFENCE": "1",
    "SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD": "0",
    "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=0x40000",
    "HIP_KERNEL_BATCH_CEILING": "100",
    "GPU_FORCE_BLIT_COPY_SIZE": "16",
    "HSA_KERNARG_POOL_SIZE": "8388608",
    "ROC_AQL_QUEUE_SIZE": "131072",
    "SGLANG_USE_LIGHTOP": "1",
    "SGLANG_ROCM_USE_AITER_MOE": "0",
    "W8A8_SUPPORT_METHODS": "3",
    "SGLANG_KVALLOC_KERNEL": "1",
    "SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO": "1",
    "SGLANG_ASSIGN_EXTEND_CACHE_LOCS": "1",
    "SGLANG_ASSIGN_REQ_TO_TOKEN_POOL": "1",
    "SGLANG_GET_LAST_LOC": "1",
    "SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON": "1",
    "SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES": "1",
    "HIP_GRAPH_ACCUMULATE_DISPATCH": "1",
    "HIP_GRAPH_USE_CMD_CACHE": "0",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "NCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
}

DEEPSEEK_V32_COOKBOOK_ENV = {
    "USE_DCU_CUSTOM_ALLREDUCE": "1",
    "SGL_CHUNKED_PREFIX_CACHE_THRESHOLD": "0",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=0x40000",
    "SGLANG_KVALLOC_KERNEL": "1",
    "SGLANG_USE_LIGHTOP": "1",
    "SGLANG_USE_OPT_CAT": "1",
    "SGLANG_USE_FP8_W8A8_MOE": "1",
    "SGLANG_USE_RMS_QUANT_PATH": "1",
    "USE_FUSED_RMS_QUANT_PATH": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HIP_KERNEL_BATCH_CEILING": "100",
    "GPU_MAX_HW_QUEUES": "4",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO": "1",
    "SGLANG_ASSIGN_EXTEND_CACHE_LOCS": "1",
    "SGLANG_ASSIGN_REQ_TO_TOKEN_POOL": "1",
    "SGLANG_GET_LAST_LOC": "1",
    "SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON": "1",
    "SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES": "1",
    "HIP_H2D_DISABLE_COPY_BUFFER": "0",
    "HIP_D2H_DISABLE_COPY_BUFFER": "0",
    "HIP_H2D_DIRECT_COPY_THRESHOLD": "32768",
    "HIP_H2D_HSAAPI_COPY_THRESHOLD": "32768",
    "HIP_D2H_DIRECT_COPY_THRESHOLD": "512",
    "HIP_D2H_HSAAPI_COPY_THRESHOLD": "512",
    "HSA_KERNARG_POOL_SIZE": "8388608",
    "ROC_AQL_QUEUE_SIZE": "131072",
    "NCCL_MAX_NCHANNELS": "16",
    "NCCL_MIN_NCHANNELS": "16",
    "ALLREDUCE_STREAM_WITH_COMPUTE": "1",
    "USE_SPE_MQP": "1",
}

MINIMAX_M25_COOKBOOK_ENV = {
    "SGLANG_USE_MODELSCOPE": "1",
    "USE_DCU_CUSTOM_ALLREDUCE": "1",
    "SGL_CHUNKED_PREFIX_CACHE_THRESHOLD": "0",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=0x40000",
    "SGLANG_USE_LIGHTOP": "1",
    "VLLM_USE_LIGHTOP_MOE_ALIGN": "1",
    "LMSLIM_USE_LIGHTOP": "1",
    "SGLANG_KVALLOC_KERNEL": "1",
    "SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO": "1",
    "SGLANG_ASSIGN_EXTEND_CACHE_LOCS": "1",
    "SGLANG_ASSIGN_REQ_TO_TOKEN_POOL": "1",
    "SGLANG_GET_LAST_LOC": "1",
    "SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON": "1",
    "SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES": "1",
    "NCCL_MAX_NCHANNELS": "16",
    "NCCL_MIN_NCHANNELS": "16",
    "ALLREDUCE_STREAM_WITH_COMPUTE": "1",
}

KIMI_K26_COOKBOOK_ENV = {
    "SGLANG_USE_LIGHTOP": "1",
    "SGLANG_USE_OPT_CAT": "1",
    "USE_DCU_CUSTOM_ALLREDUCE": "1",
    "SGL_CHUNKED_PREFIX_CACHE_THRESHOLD": "0",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=0x40000",
    "HIP_GRAPH_ACCUMULATE_DISPATCH": "0",
    "SGLANG_KVALLOC_KERNEL": "1",
    "SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO": "1",
    "SGLANG_ASSIGN_EXTEND_CACHE_LOCS": "1",
    "SGLANG_ASSIGN_REQ_TO_TOKEN_POOL": "1",
    "SGLANG_GET_LAST_LOC": "1",
    "SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON": "1",
    "SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES": "1",
    "NCCL_MAX_NCHANNELS": "16",
    "NCCL_MIN_NCHANNELS": "16",
    "ALLREDUCE_STREAM_WITH_COMPUTE": "1",
}

MIMO_V2_FLASH_COOKBOOK_ENV = {
    "SGLANG_USE_LIGHTOP": "1",
    "SGLANG_KV_LAYOUT_DCU_FA": "0",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_USE_AITER_FP8_ASM_MOE": "1",
    "SGLANG_USE_TRITON_EXTEND_FROM_AITER": "1",
    "SGLANG_USE_MODELSCOPE": "1",
}

VLM_COOKBOOK_ENV = {
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_USE_LIGHTOP": "1",
}


@dataclass(frozen=True)
class DcuCookbookModelConfig:
    name: str
    env_name: str
    default_path: str
    tp_size: int
    server_args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 3600
    dtype_or_quant: str = "bf16"

    def resolve_model_path(self) -> str:
        return get_model_path(self.env_name, self.default_path)

    def merged_env(self) -> dict[str, str]:
        merged = os.environ.copy()
        merged.update(self.env)
        return merged


def _common_text_args(tp_size: int) -> list[str]:
    return [
        "--attention-backend",
        "fa3",
        "--tp-size",
        str(tp_size),
        "--page-size",
        "64",
        "--trust-remote-code",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]


def _qwen3_next_args(tp_size: int) -> list[str]:
    return _common_text_args(tp_size) + [
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
        "--max-running-requests",
        "4",
        "--disable-custom-all-reduce",
    ]


def _glm51_args(quantization: str) -> list[str]:
    return [
        "--trust-remote-code",
        "--tp-size",
        "8",
        "--nsa-prefill-backend",
        "flashmla_auto",
        "--nsa-decode-backend",
        "flashmla_sparse",
        "--quantization",
        quantization,
        "--dtype",
        "bfloat16",
        "--dist-timeout",
        "10000",
        "--watchdog-timeout",
        "3600",
        "--page-size",
        "64",
        "--kv-cache-dtype",
        "bf16",
        "--mem-fraction-static",
        "0.8",
        "--chunked-prefill-size",
        "8192",
        "--reasoning-parser",
        "glm45",
        "--tool-call-parser",
        "glm47",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]

def _deepseek_v32_args(quantization: str) -> list[str]:
    return [
        "--numa-node",
        "0",
        "0",
        "0",
        "0",
        "1",
        "1",
        "1",
        "1",
        "--disable-radix-cache",
        "--page-size",
        "64",
        "--context-length",
        "65536",
        "--quantization",
        quantization,
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--trust-remote-code",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--dtype",
        "bfloat16",
        "--tp-size",
        "8",
        "--pp-size",
        "1",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "nsa",
        "--nsa-prefill-backend",
        "flashmla_auto",
        "--nsa-decode-backend",
        "flashmla_kv",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]


def _minimax_m25_args(quantization: str | None = None) -> list[str]:
    args = [
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--trust-remote-code",
        "--page-size",
        "64",
        "--dtype",
        "bfloat16",
        "--tp-size",
        "4",
        "--pp-size",
        "1",
        "--dp-size",
        "2",
        "--tool-call-parser",
        "minimax-m2",
        "--reasoning-parser",
        "minimax-append-think",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "fa3",
        "--numa-node",
        "0",
        "0",
        "0",
        "0",
        "1",
        "1",
        "1",
        "1",
        "--chunked-prefill-size",
        "16384",
        "--max-running-requests",
        "512",
        "--context-length",
        "131072",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]
    if quantization is not None:
        args = ["--quantization", quantization] + args
    return args


def _kimi_k26_args() -> list[str]:
    return [
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--trust-remote-code",
        "--page-size",
        "64",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--dtype",
        "bfloat16",
        "--tp-size",
        "8",
        "--pp-size",
        "1",
        "--mem-fraction-static",
        "0.9",
        "--attention-backend",
        "dcu_mla",
        "--cuda-graph-max-bs",
        "16",
        "--numa-node",
        "0",
        "0",
        "0",
        "0",
        "1",
        "1",
        "1",
        "1",
        "--chunked-prefill-size",
        "-1",
        "--max-running-requests",
        "512",
        "--context-length",
        "65536",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]


def _mimo_v2_flash_args() -> list[str]:
    return [
        "--pp-size",
        "1",
        "--dp-size",
        "2",
        "--tp-size",
        "8",
        "--page-size",
        "64",
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.85",
        "--max-running-requests",
        "128",
        "--tool-call-parser",
        "mimo",
        "--disable-radix-cache",
        "--context-length",
        "262144",
        "--attention-backend",
        "triton",
        "--chunked-prefill-size",
        "-1",
        "--enable-dp-attention",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]


def _vlm_args(tp_size: int, quantization: str | None = None) -> list[str]:
    args = [
        "--attention-backend",
        "fa3",
        "--mm-attention-backend",
        "fa3",
        "--tp-size",
        str(tp_size),
        "--page-size",
        "64",
        "--enable-multimodal",
        "--trust-remote-code",
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]
    if quantization is not None:
        args += ["--quantization", quantization]
    return args


QWEN3_NEXT_80B_4GPU = DcuCookbookModelConfig(
    name="Qwen3-Next-80B-A3B-Instruct",
    env_name="SGLANG_DCU_QWEN3_NEXT_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-Next-80B-A3B-Instruct",
    tp_size=4,
    timeout=3600,
    dtype_or_quant="bf16",
    env=QWEN3_COOKBOOK_ENV,
    server_args=_qwen3_next_args(4),
)

QWEN3_30B_A3B_4GPU = DcuCookbookModelConfig(
    name="Qwen3-30B-A3B",
    env_name="SGLANG_DCU_QWEN3_30B_A3B_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-30B-A3B",
    tp_size=4,
    timeout=3600,
    dtype_or_quant="bf16",
    env=QWEN3_COOKBOOK_ENV,
    server_args=_common_text_args(4),
)

QWEN3_32B_4GPU = DcuCookbookModelConfig(
    name="Qwen3-32B",
    env_name="SGLANG_DCU_QWEN3_32B_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-32B",
    tp_size=4,
    timeout=3600,
    dtype_or_quant="bf16",
    env=QWEN3_COOKBOOK_ENV,
    server_args=_common_text_args(4),
)

QWEN3_30B_A3B_W8A8_4GPU = DcuCookbookModelConfig(
    name="Qwen3-30B-A3B-w8a8",
    env_name="SGLANG_DCU_QWEN3_30B_A3B_W8A8_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/vllm-w8a8-models/Qwen3-30B-A3B-w8a8",
    tp_size=4,
    timeout=3600,
    dtype_or_quant="w8a8",
    env=QWEN3_COOKBOOK_ENV,
    server_args=_common_text_args(4),
)

GLM51_CHANNEL_FP8_8GPU = DcuCookbookModelConfig(
    name="GLM-5.1-Channel-FP8",
    env_name="SGLANG_DCU_GLM51_CHANNEL_FP8_MODEL",
    default_path="/public4/share/GLM-5.1-Channel-fp8",
    tp_size=8,
    timeout=7200,
    dtype_or_quant="w8a8_fp8",
    env=GLM51_COOKBOOK_ENV,
    server_args=_glm51_args("w8a8_fp8"),
)

GLM51_CHANNEL_INT8_8GPU = DcuCookbookModelConfig(
    name="GLM-5.1-Channel-INT8",
    env_name="SGLANG_DCU_GLM51_CHANNEL_INT8_MODEL",
    default_path="/public4/share/GLM-5.1-Channel-INT8",
    tp_size=8,
    timeout=7200,
    dtype_or_quant="w8a8_int8",
    env=GLM51_COOKBOOK_ENV,
    server_args=_glm51_args("w8a8_int8"),
)

DEEPSEEK_V32_CHANNEL_FP8_8GPU = DcuCookbookModelConfig(
    name="DeepSeek-V3.2-Channel-FP8",
    env_name="SGLANG_DCU_DEEPSEEK_V32_CHANNEL_FP8_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/deepseek-v3.2/DeepSeek-V3.2-channel-fp8",
    tp_size=8,
    timeout=7200,
    dtype_or_quant="w8a8_fp8",
    env=DEEPSEEK_V32_COOKBOOK_ENV,
    server_args=_deepseek_v32_args("w8a8_fp8"),
)

DEEPSEEK_V32_CHANNEL_INT8_8GPU = DcuCookbookModelConfig(
    name="DeepSeek-V3.2-Channel-INT8",
    env_name="SGLANG_DCU_DEEPSEEK_V32_CHANNEL_INT8_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/deepseek-v3.2/vllm-w8a8-models/DeepSeek-V3.2-Channel-INT8",
    tp_size=8,
    timeout=7200,
    dtype_or_quant="w8a8_int8",
    env=DEEPSEEK_V32_COOKBOOK_ENV,
    server_args=_deepseek_v32_args("w8a8_int8"),
)

MINIMAX_M25_FP8_8GPU = DcuCookbookModelConfig(
    name="MiniMax-M2.5",
    env_name="SGLANG_DCU_MINIMAX_M25_FP8_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/MiniMax-M2.5",
    tp_size=8,
    timeout=7200,
    dtype_or_quant="fp8",
    env=MINIMAX_M25_COOKBOOK_ENV,
    server_args=_minimax_m25_args(),
)

MINIMAX_M25_W8A8_8GPU = DcuCookbookModelConfig(
    name="MiniMax-M2.5-W8A8",
    env_name="SGLANG_DCU_MINIMAX_M25_W8A8_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/MiniMax-M2.5-Channel-INT8-w8a8",
    tp_size=8,
    timeout=7200,
    dtype_or_quant="slimquant_marlin",
    env=MINIMAX_M25_COOKBOOK_ENV,
    server_args=_minimax_m25_args("slimquant_marlin"),
)

KIMI_K26_8GPU = DcuCookbookModelConfig(
    name="Kimi-K2.6",
    env_name="SGLANG_DCU_KIMI_K26_MODEL",
    default_path="/public4/opendas/DL_DATA/llm-models/Kimi-K2.6",
    tp_size=8,
    timeout=7200,
    dtype_or_quant="w4a16",
    env=KIMI_K26_COOKBOOK_ENV,
    server_args=_kimi_k26_args(),
)

MIMO_V2_FLASH_8GPU = DcuCookbookModelConfig(
    name="MiMo-V2-Flash",
    env_name="SGLANG_DCU_MIMO_V2_FLASH_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/Xiaomi/MiMo-V2-Flash",
    tp_size=8,
    timeout=7200,
    dtype_or_quant="fp8",
    env=MIMO_V2_FLASH_COOKBOOK_ENV,
    server_args=_mimo_v2_flash_args(),
)

QWEN3_VL_4B_INSTRUCT = DcuCookbookModelConfig(
    name="Qwen3-VL-4B-Instruct",
    env_name="SGLANG_DCU_QWEN3_VL_4B_INSTRUCT_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-VL-4B-Instruct",
    tp_size=1,
    timeout=3600,
    dtype_or_quant="bf16",
    env=VLM_COOKBOOK_ENV,
    server_args=_vlm_args(1),
)

GLM41V_9B_THINKING = DcuCookbookModelConfig(
    name="GLM-4.1V-9B-Thinking",
    env_name="SGLANG_DCU_GLM41V_9B_THINKING_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/glm4/GLM-4.1V-9B-Thinking",
    tp_size=1,
    timeout=3600,
    dtype_or_quant="bf16",
    env=VLM_COOKBOOK_ENV,
    server_args=_vlm_args(1),
)

QWEN3_VL_32B_INSTRUCT = DcuCookbookModelConfig(
    name="Qwen3-VL-32B-Instruct",
    env_name="SGLANG_DCU_QWEN3_VL_32B_INSTRUCT_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-VL-32B-Instruct",
    tp_size=4,
    timeout=5400,
    dtype_or_quant="bf16",
    env=VLM_COOKBOOK_ENV,
    server_args=_vlm_args(4),
)

QWEN25_VL_72B_W8A8 = DcuCookbookModelConfig(
    name="Qwen2.5-VL-72B-Instruct-W8A8",
    env_name="SGLANG_DCU_QWEN25_VL_72B_W8A8_MODEL",
    default_path="/public/opendas/DL_DATA/llm-models/vllm-w8a8-models/Qwen2.5-VL-72B-Instruct-quantized.w8a8",
    tp_size=8,
    timeout=7200,
    dtype_or_quant="w8a8",
    env=VLM_COOKBOOK_ENV,
    server_args=_vlm_args(8, "w8a8_int8"),
)

QWEN3_4GPU_MODELS = [QWEN3_NEXT_80B_4GPU, QWEN3_30B_A3B_4GPU, QWEN3_32B_4GPU]
QWEN3_4GPU_PERF_MODELS = [QWEN3_30B_A3B_4GPU, QWEN3_32B_4GPU]
QWEN3_4GPU_QUANT_MODELS = [QWEN3_30B_A3B_W8A8_4GPU]
GLM51_8GPU_MODELS = [GLM51_CHANNEL_FP8_8GPU, GLM51_CHANNEL_INT8_8GPU]
GLM51_8GPU_PERF_MODELS = [GLM51_CHANNEL_FP8_8GPU]
DEEPSEEK_V32_8GPU_MODELS = [
    DEEPSEEK_V32_CHANNEL_FP8_8GPU,
    DEEPSEEK_V32_CHANNEL_INT8_8GPU,
]
DEEPSEEK_V32_8GPU_PERF_MODELS = [DEEPSEEK_V32_CHANNEL_FP8_8GPU]
DEEPSEEK_V32_8GPU_QUANT_MODELS = [DEEPSEEK_V32_CHANNEL_INT8_8GPU]
MINIMAX_M25_8GPU_MODELS = [MINIMAX_M25_FP8_8GPU]
MINIMAX_M25_8GPU_PERF_MODELS = [MINIMAX_M25_FP8_8GPU]
MINIMAX_M25_8GPU_QUANT_MODELS = [MINIMAX_M25_W8A8_8GPU]
KIMI_MIMO_8GPU_MODELS = [KIMI_K26_8GPU, MIMO_V2_FLASH_8GPU]
KIMI_MIMO_8GPU_PERF_MODELS = [KIMI_K26_8GPU, MIMO_V2_FLASH_8GPU]
VLM_COOKBOOK_MODELS = [
    QWEN3_VL_4B_INSTRUCT,
    GLM41V_9B_THINKING,
    QWEN3_VL_32B_INSTRUCT,
    QWEN25_VL_72B_W8A8,
]
VLM_COOKBOOK_QUANT_MODELS = [QWEN25_VL_72B_W8A8]
COOKBOOK_GSM8K_EVAL_MODELS = [QWEN3_32B_4GPU, QWEN3_30B_A3B_4GPU]
COOKBOOK_MMLU_EVAL_MODELS = [QWEN3_32B_4GPU]
COOKBOOK_MMMU_EVAL_MODELS = [QWEN3_VL_4B_INSTRUCT, QWEN3_VL_32B_INSTRUCT]


def selected_configs(
    configs: list[DcuCookbookModelConfig], env_name: str
) -> list[DcuCookbookModelConfig]:
    pattern = os.environ.get(env_name, "").strip()
    if not pattern:
        return list(configs)
    regex = re.compile(pattern, re.IGNORECASE)
    selected = [
        config
        for config in configs
        if regex.search(config.name) or regex.search(config.default_path)
    ]
    if not selected:
        raise AssertionError(f"{env_name}={pattern!r} did not match any model configs")
    return selected


def _threshold_suffix(model_name: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").upper()
    if not suffix:
        raise AssertionError(f"cannot derive threshold env suffix from {model_name!r}")
    return suffix


def _get_threshold(
    config: DcuCookbookModelConfig, defaults: dict[str, float], env_prefix: str
) -> float | None:
    suffix = _threshold_suffix(config.name)
    for key in (f"{env_prefix}_{suffix}", f"{env_prefix}_DEFAULT", env_prefix):
        value = os.environ.get(key)
        if value not in (None, ""):
            return float(value)
    return defaults.get(config.name)


def assert_cookbook_min_score(
    config: DcuCookbookModelConfig,
    metrics: dict,
    defaults: dict[str, float],
    env_prefix: str,
) -> None:
    threshold = _get_threshold(config, defaults, env_prefix)
    if threshold is None:
        return
    score = float(metrics["score"])
    print(f"DCU cookbook threshold {config.name}: score={score}, min_score={threshold}")
    if score < threshold:
        raise AssertionError(f"{config.name} score={score} < min_score={threshold}")


def assert_cookbook_min_output_throughput(
    config: DcuCookbookModelConfig,
    result: dict,
    defaults: dict[str, float],
    env_prefix: str,
) -> None:
    threshold = _get_threshold(config, defaults, env_prefix)
    if threshold is None:
        return
    output_tps = float(result["output_throughput"])
    print(
        f"DCU cookbook threshold {config.name}: "
        f"output_throughput={output_tps}, min_output_tps={threshold}"
    )
    if output_tps < threshold:
        raise AssertionError(
            f"{config.name} output_throughput={output_tps} < min_output_tps={threshold}"
        )


def assert_server_info_ready(base_url: str, api_key: str) -> dict:
    response = requests.get(
        base_url.rstrip("/") + "/server_info",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not payload:
        raise AssertionError(f"invalid /server_info response: {payload}")
    return payload


class CookbookServer:
    def __init__(self, config: DcuCookbookModelConfig, base_url: str):
        self.config = config
        self.base_url = base_url
        self.model_path = config.resolve_model_path()
        self.process = None

    def __enter__(self):
        try:
            self.process = popen_launch_server(
                self.model_path,
                self.base_url,
                timeout=self.config.timeout,
                api_key=DCU_COOKBOOK_API_KEY,
                other_args=list(self.config.server_args),
                env=self.config.merged_env(),
            )
            assert_server_info_ready(self.base_url, DCU_COOKBOOK_API_KEY)
        except Exception:
            if self.process is not None:
                kill_process_tree(self.process.pid)
                self.process = None
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.process is not None:
            kill_process_tree(self.process.pid)
            self.process = None

    def assert_chat_non_empty(self) -> str:
        response = requests.post(
            openai_base_url(self.base_url) + "/chat/completions",
            headers={
                "Authorization": f"Bearer {DCU_COOKBOOK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_path,
                "messages": [{"role": "user", "content": "中国的首都是哪里？"}],
                "temperature": 0,
                "max_tokens": 32,
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("choices", [{}])[0].get("message", {})
        content = message.get("content") or message.get("reasoning_content") or ""
        if not content.strip():
            raise AssertionError(f"chat completion returned empty content: {payload}")
        return content

    def assert_vlm_chat_non_empty(self) -> str:
        response = requests.post(
            openai_base_url(self.base_url) + "/chat/completions",
            headers={
                "Authorization": f"Bearer {DCU_COOKBOOK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_path,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请用一句话描述这张图片。"},
                            {
                                "type": "image_url",
                                "image_url": {"url": RED_DOT_IMAGE_DATA_URL},
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 64,
            },
            timeout=240,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("choices", [{}])[0].get("message", {})
        content = message.get("content") or message.get("reasoning_content") or ""
        if not content.strip():
            raise AssertionError(f"VLM chat completion returned empty content: {payload}")
        return content


def run_random_serving_perf(
    config: DcuCookbookModelConfig,
    base_url: str,
    output_dir: str,
    num_prompts: int = 64,
    input_len: int = 2048,
    output_len: int = 256,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = config.resolve_model_path()
    result_file = output_path / (
        f"{config.name.replace('/', '_')}_{num_prompts}_{input_len}_{output_len}.jsonl"
    )
    env = config.merged_env()
    env["OPENAI_API_KEY"] = DCU_COOKBOOK_API_KEY
    cmd = [
        "python3",
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang-oai-chat",
        "--base-url",
        base_url.rstrip("/"),
        "--model",
        model_path,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--request-rate",
        "inf",
        "--output-file",
        str(result_file),
    ]

    started_at = time.time()
    completed = subprocess.run(cmd, env=env, text=True, capture_output=True)
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr)
        raise AssertionError(
            f"bench_serving failed for {config.name} after {time.time() - started_at:.1f}s"
        )
    if not result_file.exists():
        raise AssertionError(f"bench_serving did not create {result_file}")
    with result_file.open() as f:
        payload = json.loads(f.readline())
    required = [
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_itl_ms",
    ]
    missing = [key for key in required if key not in payload or payload[key] is None]
    if missing:
        raise AssertionError(f"benchmark output missing required metrics {missing}")
    print(
        "DCU cookbook perf "
        f"{config.name}: request_throughput={payload['request_throughput']}, "
        f"input_throughput={payload['input_throughput']}, "
        f"output_throughput={payload['output_throughput']}, "
        f"mean_ttft_ms={payload['mean_ttft_ms']}, "
        f"mean_tpot_ms={payload['mean_tpot_ms']}, "
        f"mean_itl_ms={payload['mean_itl_ms']}"
    )
    return payload


def run_cookbook_accuracy_eval(
    config: DcuCookbookModelConfig,
    base_url: str,
    eval_name: str,
    num_examples: int,
    num_threads: int,
    num_shots: int = 5,
    max_tokens: int = 2048,
) -> dict:
    if eval_name not in {"gsm8k", "mmlu", "mmmu"}:
        raise AssertionError(f"unsupported cookbook eval_name={eval_name!r}")

    dataset_path = None
    gsm8k_data_path = None
    if eval_name == "mmlu":
        dataset_path = (
            os.environ.get("SGLANG_DCU_COOKBOOK_MMLU_DATASET_PATH")
            or os.environ.get("SGLANG_DCU_MMLU_DATASET_PATH")
        )
    elif eval_name == "mmmu":
        dataset_path = (
            os.environ.get("SGLANG_DCU_COOKBOOK_MMMU_DATASET_PATH")
            or os.environ.get("SGLANG_DCU_MMMU_DATASET_PATH")
        )
    elif eval_name == "gsm8k":
        gsm8k_data_path = (
            os.environ.get("SGLANG_DCU_COOKBOOK_GSM8K_DATA_PATH")
            or os.environ.get("SGLANG_DCU_GSM8K_DATA_PATH")
        )

    os.environ["OPENAI_API_KEY"] = DCU_COOKBOOK_API_KEY
    with CookbookServer(config, base_url) as server:
        args = SimpleNamespace(
            base_url=base_url,
            model=server.model_path,
            eval_name=eval_name,
            api="chat",
            num_examples=num_examples,
            num_threads=num_threads,
            num_shots=num_shots,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            repeat=1,
            gsm8k_data_path=gsm8k_data_path,
            dataset_path=dataset_path,
            response_answer_regex=None,
            return_latency=False,
        )
        metrics = run_eval(args)

    score = metrics.get("score", metrics.get("mean_score"))
    if score is None:
        raise AssertionError(f"{eval_name} did not report score: {metrics}")
    score = float(score)
    if not 0.0 <= score <= 1.0:
        raise AssertionError(f"{eval_name} score out of range [0, 1]: {score}")

    latency = metrics.get("latency")
    print(
        "DCU cookbook accuracy "
        f"{eval_name} {config.name}: num_examples={num_examples}, "
        f"num_threads={num_threads}, score={score}, latency={latency}"
    )
    return metrics
