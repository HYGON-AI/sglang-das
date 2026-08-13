import unittest
from unittest.mock import patch

import sglang.srt.speculative.eagle_worker_v2 as eagle_worker_v2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEagleWorkerV2GraphSupport(CustomTestCase):
    def test_hcu_uses_cuda_style_graph_runner(self):
        with (
            patch.object(eagle_worker_v2, "_is_cuda", False),
            patch.object(eagle_worker_v2, "_is_hcu", True),
            patch.object(eagle_worker_v2, "_is_musa", False),
        ):
            self.assertTrue(eagle_worker_v2._supports_cuda_graph_runner())

    def test_plain_hip_does_not_use_cuda_style_graph_runner(self):
        with (
            patch.object(eagle_worker_v2, "_is_cuda", False),
            patch.object(eagle_worker_v2, "_is_hcu", False),
            patch.object(eagle_worker_v2, "_is_musa", False),
        ):
            self.assertFalse(eagle_worker_v2._supports_cuda_graph_runner())


if __name__ == "__main__":
    unittest.main()
