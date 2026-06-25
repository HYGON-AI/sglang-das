import unittest

import requests

from sglang.test.ci.ci_register import register_dcu_ci

register_dcu_ci(
    est_time=900,
    suite="stage-b-test-1-gpu-small-dcu",
    disabled="DCU disabled retest: torchao can be installed, but current package lacks float8_dynamic_activation_float8_weight required by sglang torchao_utils; needs container dependency alignment.",
)
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestTorchAODCU(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--torchao-config", "int4wo-128", "--attention-backend", "fa3", "--page-size", "64", "--trust-remote-code", "--disable-cuda-graph"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "ignore_eos": True,
            },
        )
        return response.json()

    def test_torchao_generate(self):
        res = self.run_decode(16)
        self.assertIn("text", res)
        self.assertTrue(res["text"])


@unittest.skip("DCU torchao smoke keeps text-only int4wo path; VLM fp8wo needs separate backend validation.")
class TestTorchAOForVLM(CustomTestCase):
    def test_vlm_generate(self):
        model_path = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
        chat_template = get_chat_template_by_model_path(model_path)
        text = f"{chat_template.image_token}What is in this picture? Answer: "

        engine = Engine(
            model_path=model_path,
            max_total_tokens=512,
            enable_multimodal=True,
            torchao_config="fp8wo",
        )
        out = engine.generate([text], image_data=[DEFAULT_IMAGE_URL])
        engine.shutdown()
        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
