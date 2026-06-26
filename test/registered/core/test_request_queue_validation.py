import asyncio
import os
import re
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci

register_dcu_ci(est_time=120, suite="stage-b-test-1-gpu-small-dcu")

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    send_concurrent_generate_requests,
    send_generate_requests,
)

register_cuda_ci(est_time=53, stage="stage-b", runner_config="1-gpu-small")
register_amd_ci(est_time=70, suite="stage-b-test-1-gpu-small-amd")


DCU_SMALL_MODEL = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-0.6B"


def _is_dcu() -> bool:
    return os.environ.get("SGLANG_IS_IN_CI_DCU", "0") == "1"


def _dcu_url() -> str:
    return f"http://127.0.0.1:{find_available_port(11001)}"


def _dcu_generate_status(base_url: str) -> int:
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        },
        timeout=60,
    )
    return response.status_code


def _server_args() -> tuple[str, ...]:
    args = (
        "--max-running-requests",  # Enforce max request concurrency is 1
        "1",
        "--max-queued-requests",  # Enforce max queued request number is 1
        "1",
        "--attention-backend",
        "fa3" if _is_dcu() else "triton",
    )
    if not _is_dcu():
        return args
    return args + (
        "--page-size",
        "64",
        "--max-total-tokens",
        "512",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--trust-remote-code",
    )


class TestMaxQueuedRequests(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DCU_SMALL_MODEL if _is_dcu() else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = _dcu_url() if _is_dcu() else DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_server_args(),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()
        os.remove(STDOUT_FILENAME)
        os.remove(STDERR_FILENAME)

    def test_max_queued_requests_validation_with_serial_requests(self):
        """Verify request is not throttled when the max concurrency is 1."""
        if _is_dcu():
            status_codes = [_dcu_generate_status(self.base_url) for _ in range(4)]
        else:
            status_codes = send_generate_requests(
                self.base_url,
                num_requests=10,
            )

        for status_code in status_codes:
            assert status_code == 200  # request shouldn't be throttled

    def test_max_queued_requests_validation_with_concurrent_requests(self):
        """Verify request throttling with concurrent requests."""
        if _is_dcu():
            with ThreadPoolExecutor(max_workers=6) as executor:
                status_codes = list(
                    executor.map(lambda _: _dcu_generate_status(self.base_url), range(6))
                )
        else:
            status_codes = asyncio.run(
                send_concurrent_generate_requests(self.base_url, num_requests=10)
            )
        self.assertLessEqual(status_codes.count(200), 2)

        # expected_status_codes = [200, 200, 503, 503, 503, 503, 503, 503, 503, 503]
        # self.assertEqual(status_codes, expected_status_codes)

    def test_max_running_requests_and_max_queued_request_validation(self):
        """Verify running request and queued request numbers based on server logs."""
        rr_pattern = re.compile(r"#running-req:\s*(\d+)")
        qr_pattern = re.compile(r"#queue-req:\s*(\d+)")

        with open(STDERR_FILENAME) as lines:
            for line in lines:
                rr_match, qr_match = rr_pattern.search(line), qr_pattern.search(line)
                if rr_match:
                    assert int(rr_match.group(1)) <= 1
                if qr_match:
                    assert int(qr_match.group(1)) <= 1


if __name__ == "__main__":
    unittest.main()
