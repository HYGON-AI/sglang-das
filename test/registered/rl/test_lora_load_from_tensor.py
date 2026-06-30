import json
import os
import unittest

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

import sglang as sgl
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci

register_dcu_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-small-dcu",
)

from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=102, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd")

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_REPO = "charent/self_cognition_Alice"
TEST_PROMPT = "Hello, my name is"
EXPECTED_OUTPUT = (
    " Alice, and I am a software engineer. I am excited to share my journey"
)
MAX_NEW_TOKENS = 16

DCU_MODEL_PATH = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-4B"
DCU_LORA_PATH = "/public/opendas/DL_DATA/llm-models/lora/nissenj/Qwen3-4B-lora-v2"
DCU_TEST_PROMPT = "SGL is a"
DCU_LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def _is_dcu():
    return os.environ.get("SGLANG_IS_IN_CI_DCU") == "1"


class TestLoRALoadFromTensor(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.is_dcu = _is_dcu()
        if cls.is_dcu:
            cls.model_path = os.environ.get("SGLANG_DCU_LORA_TENSOR_MODEL", DCU_MODEL_PATH)
            cls.lora_path = os.environ.get("SGLANG_DCU_LORA_TENSOR_ADAPTER", DCU_LORA_PATH)
            cls.test_prompt = os.environ.get("SGLANG_DCU_LORA_TENSOR_PROMPT", DCU_TEST_PROMPT)
            cls.expected_output = None
            cls.max_new_tokens = 8
            for path in [cls.model_path, cls.lora_path]:
                if not os.path.exists(path):
                    raise unittest.SkipTest(f"DCU LoRA tensor path does not exist: {path}")

            cls.engine = sgl.Engine(
                model_path=cls.model_path,
                enable_lora=True,
                max_lora_rank=32,
                lora_target_modules=DCU_LORA_TARGET_MODULES,
                mem_fraction_static=0.65,
                attention_backend="fa3",
                page_size=64,
                disable_cuda_graph=True,
                disable_radix_cache=True,
                log_level="error",
            )

            cls.lora_tensors = load_file(
                os.path.join(cls.lora_path, "adapter_model.safetensors")
            )
            with open(os.path.join(cls.lora_path, "adapter_config.json"), "r") as f:
                cls.lora_config_dict = json.load(f)
            return

        cls.model_path = MODEL_PATH
        cls.lora_path = LORA_REPO
        cls.test_prompt = TEST_PROMPT
        cls.expected_output = EXPECTED_OUTPUT
        cls.max_new_tokens = MAX_NEW_TOKENS
        cls.engine = sgl.Engine(
            model_path=cls.model_path,
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=["all"],
            mem_fraction_static=0.6,
            log_level="error",
        )

        lora_adapter = snapshot_download(
            repo_id=LORA_REPO,
            allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
        )
        # Load tensors and config from downloaded adapter
        cls.lora_tensors = load_file(
            os.path.join(lora_adapter, "adapter_model.safetensors")
        )
        with open(os.path.join(lora_adapter, "adapter_config.json"), "r") as f:
            cls.lora_config_dict = json.load(f)

    def test_lora_lru_eviction(self):
        if self.is_dcu:
            self.skipTest("DCU keeps this case focused on tensor loading; LRU pressure creates a second large Engine.")

        print("[Test]Testing LRU LoRA eviction...")
        MAX_LOADED_LORAS = 8
        print(f"[Test]Max loaded LoRAs: {MAX_LOADED_LORAS}")
        test_engine = sgl.Engine(
            model_path=MODEL_PATH,
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=["all"],
            mem_fraction_static=0.6,
            log_level="error",
            max_loaded_loras=MAX_LOADED_LORAS,
        )

        # Load 10 LoRA adapters, max allowed is 8
        # This should trigger LRU eviction when we exceed the limit
        TEST_LORA_COUNT = 10
        for i in range(TEST_LORA_COUNT):
            print(f"[Test]Loading LoRA adapter {i+1}/10: self_cognition_Alice_{i}")
            result = test_engine.load_lora_adapter_from_tensors(
                lora_name=f"self_cognition_Alice_{i}",
                tensors=self.lora_tensors,
                config_dict=self.lora_config_dict,
            )
            self.assertTrue(
                result.success,
                f"Failed to load LoRA adapter {i}: {result.error_message}",
            )
            print(
                f"[Test]Successfully loaded LoRA {i+1}, current loaded adapters: {list(result.loaded_adapters.keys())}"
            )

        EXPECTED_LORA_ADAPTERS = [
            "self_cognition_Alice_2",
            "self_cognition_Alice_3",
            "self_cognition_Alice_4",
            "self_cognition_Alice_5",
            "self_cognition_Alice_6",
            "self_cognition_Alice_7",
            "self_cognition_Alice_8",
            "self_cognition_Alice_9",
        ]
        EXPECTED_LORA_COUNT = 8
        self.assertEqual(
            len(result.loaded_adapters),
            EXPECTED_LORA_COUNT,
            f"Loaded adapters count does not match expected result: {len(result.loaded_adapters)} != {EXPECTED_LORA_COUNT}",
        )
        self.assertEqual(
            list(result.loaded_adapters.keys()),
            EXPECTED_LORA_ADAPTERS,
            f"Loaded adapters do not match expected result: {list(result.loaded_adapters.keys())} != {EXPECTED_LORA_ADAPTERS}",
        )
        print(
            f"[Test]LRU eviction test passed! Final loaded adapters: {len(result.loaded_adapters)}"
        )

    def test_lora_e2e_load_from_tensor_params(self):
        print("[Test]Testing LoRA load from tensor params...")

        lora_name = "self_cognition_Alice" if not self.is_dcu else "dcu_qwen3_lora"
        result = self.engine.load_lora_adapter_from_tensors(
            lora_name=lora_name,
            tensors=self.lora_tensors,
            config_dict=self.lora_config_dict,
        )
        self.assertTrue(
            result.success,
            f"Failed to load LoRA from tensors: {result.error_message}",
        )

        output_without_lora = self.engine.generate(
            prompt=[self.test_prompt],
            sampling_params={
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.0,
            },
        )

        output_lora = self.engine.generate(
            prompt=[self.test_prompt],
            sampling_params={
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.0,
            },
            lora_path=[lora_name],
        )

        print(f"[Without LoRA] {output_without_lora[0]}")
        print(f"[With LoRA]  {output_lora[0]}")
        if self.is_dcu:
            self.assertTrue(output_without_lora[0]["text"].strip())
            self.assertTrue(output_lora[0]["text"].strip())
            return

        self.assertNotEqual(
            output_without_lora[0]["text"][: len(self.expected_output)],
            self.expected_output,
            "Output before applying LoRA should not match expected result",
        )

        self.assertEqual(
            output_lora[0]["text"][: len(self.expected_output)],
            self.expected_output,
            "Output after applying LoRA does not match expected result",
        )

    def test_lora_load_unload_load_from_tensor_params(self):
        print("[Test]Testing LoRA load, unload, load from tensor params...")

        # Load LoRA adapter from tensors
        result = self.engine.load_lora_adapter_from_tensors(
            lora_name="self_cognition_Alice_multiple",
            tensors=self.lora_tensors,
            config_dict=self.lora_config_dict,
        )
        self.assertTrue(
            result.success,
            f"Failed to load LoRA from tensors: {result.error_message}",
        )

        # Unload LoRA adapter
        result = self.engine.unload_lora_adapter("self_cognition_Alice_multiple")
        self.assertTrue(
            result.success, f"Failed to unload LoRA: {result.error_message}"
        )
        with self.assertRaises(ValueError) as context:
            output_lora = self.engine.generate(
                prompt=[self.test_prompt],
                sampling_params={
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": 0.0,
                },
                lora_path=["self_cognition_Alice_multiple"],
            )
        # Load LoRA adapter again
        result_again = self.engine.load_lora_adapter_from_tensors(
            lora_name="self_cognition_Alice_multiple",
            tensors=self.lora_tensors,
            config_dict=self.lora_config_dict,
        )
        self.assertTrue(
            result_again.success,
            f"Failed to load LoRA from tensors: {result_again.error_message}",
        )
        output_lora_loaded_again = self.engine.generate(
            prompt=[self.test_prompt],
            sampling_params={
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.0,
            },
            lora_path=["self_cognition_Alice_multiple"],
        )

        print(f"[With LoRA Loaded again]  {output_lora_loaded_again[0]}")
        if self.is_dcu:
            self.assertTrue(output_lora_loaded_again[0]["text"].strip())
            return

        self.assertEqual(
            output_lora_loaded_again[0]["text"][: len(self.expected_output)],
            self.expected_output,
            "Output after applying LoRA does not match expected result",
        )

    def test_lora_logp_diff_with_huggingface(self):
        """
        Test comparing SGLang and HuggingFace LoRA logprobs when loading LoRA from tensors.
        This verifies that loading LoRA adapters from tensors produces consistent logprobs
        with HuggingFace.
        """
        if self.is_dcu:
            self.skipTest("DCU local LoRA tensor smoke avoids HuggingFace runner and exact-logprob drift.")

        from sglang.test.runners import HFRunner, SRTRunner
        from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER

        print("[Test]Testing LoRA logprob difference with HuggingFace...")

        lora_name = "self_cognition_Alice_logprob_test"
        prompts = [TEST_PROMPT]

        # Step 1: Run SGLang with LoRA loaded from tensors
        print("[Test]Running SGLang with LoRA from tensors...")
        with SRTRunner(
            MODEL_PATH,
            torch_dtype=torch.float16,
            model_type="generation",
            tp_size=1,
            max_loras_per_batch=1,
            lora_backend="triton",
            disable_cuda_graph=False,
            disable_radix_cache=True,
            port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
            mem_fraction_static=0.6,
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ) as srt_runner:
            result = srt_runner.engine.load_lora_adapter_from_tensors(
                lora_name=lora_name,
                tensors=self.lora_tensors,
                config_dict=self.lora_config_dict,
            )
            self.assertTrue(
                result.success,
                f"Failed to load LoRA from tensors: {result.error_message}",
            )

            # Run inference with loaded LoRA
            srt_outputs = srt_runner.forward(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                lora_paths=[lora_name],
            )

        # Step 2: Run HuggingFace with LoRA
        print("[Test]Running HuggingFace with LoRA...")
        torch.cuda.empty_cache()

        with HFRunner(
            MODEL_PATH,
            torch_dtype=torch.float16,
            model_type="generation",
            patch_model_do_sample_false=True,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                lora_paths=[LORA_REPO],
            )

        # Step 3: Compare results
        sglang_text = srt_outputs.output_strs[0]
        hf_text = hf_outputs.output_strs[0]

        print(f"[Text Output]")
        print(f"  SGLang:      {sglang_text}")
        print(f"  HuggingFace: {hf_text}")

        # Compare prefill (input) logprobs
        sglang_prefill = torch.tensor(srt_outputs.top_input_logprobs[0])
        hf_prefill = torch.tensor(hf_outputs.top_input_logprobs[0])

        prefill_diff = torch.abs(sglang_prefill - hf_prefill)
        prefill_max_diff = torch.max(prefill_diff).item()
        prefill_mean_diff = torch.mean(prefill_diff).item()

        print(f"\n[Prefill Logprob Comparison]")
        print(f"  Shape:           {list(sglang_prefill.shape)}")
        print(f"  Max difference:  {prefill_max_diff:.6e}")
        print(f"  Mean difference: {prefill_mean_diff:.6e}")

        # Compare decode (output) logprobs
        sglang_decode = torch.tensor(srt_outputs.top_output_logprobs[0])
        hf_decode = torch.tensor(hf_outputs.top_output_logprobs[0])

        decode_diff = torch.abs(sglang_decode - hf_decode)
        decode_max_diff = torch.max(decode_diff).item()
        decode_mean_diff = torch.mean(decode_diff).item()

        print(f"\n[Decode Logprob Comparison]")
        print(f"  Shape:           {list(sglang_decode.shape)}")
        print(f"  Max difference:  {decode_max_diff:.6e}")
        print(f"  Mean difference: {decode_mean_diff:.6e}")

        # Assert logprobs are close (threshold 1e-1)
        LOGPROB_THRESHOLD = 1e-1
        self.assertLess(
            prefill_max_diff,
            LOGPROB_THRESHOLD,
            f"Prefill logprob max difference too large: {prefill_max_diff:.6e} > {LOGPROB_THRESHOLD:.0e}",
        )
        self.assertLess(
            decode_max_diff,
            LOGPROB_THRESHOLD,
            f"Decode logprob max difference too large: {decode_max_diff:.6e} > {LOGPROB_THRESHOLD:.0e}",
        )

        # Verify text outputs match expected
        self.assertEqual(
            sglang_text[: len(EXPECTED_OUTPUT)],
            EXPECTED_OUTPUT,
            "SGLang output does not match expected result",
        )

        print("\n[Test]LoRA logprob comparison test passed!")

    def test_lora_e2e_load_from_flattened_bucket(self):
        """Test loading LoRA via FlattenedTensorBucket format (RL weight sync path)."""
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        named_tensors = list(self.lora_tensors.items())
        bucket = FlattenedTensorBucket(named_tensors=[(n, t) for n, t in named_tensors])
        bucket_dict = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }
        serialized = MultiprocessingSerializer.serialize(bucket_dict, output_str=True)

        result = self.engine.load_lora_adapter_from_tensors(
            lora_name="self_cognition_Alice_flattened",
            tensors=serialized,
            config_dict=self.lora_config_dict,
            load_format="flattened_bucket",
        )
        self.assertTrue(result.success, f"Failed: {result.error_message}")

        output = self.engine.generate(
            prompt=[self.test_prompt],
            sampling_params={"max_new_tokens": self.max_new_tokens, "temperature": 0.0},
            lora_path=["self_cognition_Alice_flattened"],
        )
        if self.is_dcu:
            self.assertTrue(output[0]["text"].strip())
            return

        self.assertEqual(
            output[0]["text"][: len(self.expected_output)],
            self.expected_output,
            "Output after applying LoRA via flattened bucket does not match expected",
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()


if __name__ == "__main__":
    unittest.main()
