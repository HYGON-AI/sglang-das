import unittest

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_cookbook_utils import (
    CookbookServer,
    KIMI_MIMO_8GPU_MODELS,
    selected_configs,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_dcu_ci(
    est_time=14400,
    suite="nightly-dcu-large-model-8gpu",
    nightly=True,
)


class TestKimiMimoCookbook8GpuServerDCU(unittest.TestCase):
    def test_kimi_mimo_cookbook_8gpu_chat(self):
        configs = selected_configs(
            KIMI_MIMO_8GPU_MODELS, "SGLANG_DCU_KIMI_MIMO_8GPU_MODEL_FILTER"
        )
        for config in configs:
            with self.subTest(model=config.name):
                with CookbookServer(config, DEFAULT_URL_FOR_TEST) as server:
                    content = server.assert_chat_non_empty()
                    self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
