import unittest

from sglang.test.ci.ci_register import register_dcu_ci
from sglang.test.dcu_cookbook_utils import (
    CookbookServer,
    VLM_COOKBOOK_MODELS,
    selected_configs,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_dcu_ci(
    est_time=14400,
    suite="nightly-dcu-vlm",
    nightly=True,
)


class TestCookbookVlmServerDCU(unittest.TestCase):
    def test_cookbook_vlm_chat(self):
        configs = selected_configs(
            VLM_COOKBOOK_MODELS, "SGLANG_DCU_VLM_COOKBOOK_MODEL_FILTER"
        )
        for config in configs:
            with self.subTest(model=config.name):
                with CookbookServer(config, DEFAULT_URL_FOR_TEST) as server:
                    content = server.assert_vlm_chat_non_empty()
                    self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()
