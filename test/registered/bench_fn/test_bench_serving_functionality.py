import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from sglang.bench_serving import run_benchmark
from sglang.benchmark.utils import parse_custom_headers
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_dcu_ci

register_dcu_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-small-dcu",
)

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    get_benchmark_args,
    popen_launch_server,
)

register_cuda_ci(est_time=300, suite="nightly-1-gpu", nightly=True)
register_amd_ci(est_time=300, suite="nightly-amd-1-gpu", nightly=True)

MODEL = "Qwen/Qwen3-0.6B"
DCU_MODEL = "/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-0.6B"
NUM_CONVERSATIONS, NUM_TURNS = 4, 3


def _is_dcu() -> bool:
    return os.environ.get("SGLANG_IS_IN_CI_DCU", "0") == "1"


def _dcu_url() -> str:
    return f"http://127.0.0.1:{find_available_port(11001)}"


class TestBenchServingFunctionality(CustomTestCase):
    def test_gsp_multi_turn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_url = _dcu_url() if _is_dcu() else DEFAULT_URL_FOR_TEST
            server_args = [
                "--log-requests",
                "--log-requests-level",
                "3",
                "--log-requests-format",
                "json",
                "--log-requests-target",
                "stdout",
                temp_dir,
            ]
            if _is_dcu():
                server_args += [
                    "--attention-backend",
                    "fa3",
                    "--page-size",
                    "64",
                    "--max-total-tokens",
                    "512",
                    "--disable-cuda-graph",
                    "--disable-radix-cache",
                    "--trust-remote-code",
                ]
            else:
                server_args = ["--mem-fraction-static", "0.7"] + server_args
            process = popen_launch_server(
                DCU_MODEL if _is_dcu() else MODEL,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=server_args,
            )
            try:
                args = get_benchmark_args(
                    base_url=base_url,
                    backend="sglang-oai-chat",
                    tokenizer=DCU_MODEL if _is_dcu() else MODEL,
                    dataset_name="generated-shared-prefix",
                    num_prompts=2 if _is_dcu() else NUM_CONVERSATIONS,
                    request_rate=float("inf"),
                    gsp_num_groups=2,
                    gsp_prompts_per_group=1 if _is_dcu() else 2,
                    gsp_system_prompt_len=16 if _is_dcu() else 64,
                    gsp_question_len=8 if _is_dcu() else 16,
                    gsp_output_len=8 if _is_dcu() else 16,
                    gsp_num_turns=2 if _is_dcu() else NUM_TURNS,
                )
                args.warmup_requests = 0
                res = run_benchmark(args)
                expected_conversations = 2 if _is_dcu() else NUM_CONVERSATIONS
                expected_turns = 2 if _is_dcu() else NUM_TURNS
                expected_completed = expected_conversations * expected_turns
                self.assertEqual(res["completed"], expected_completed)

                time.sleep(1)
                logs = "".join(f.read_text() for f in Path(temp_dir).glob("*.log"))
                self._verify_multi_turn_logs(logs, expected_conversations, expected_turns)
            finally:
                kill_process_tree(process.pid)

    def _verify_multi_turn_logs(
        self, content: str, num_conversations: int, num_turns: int
    ):
        reqs = []
        for line in content.splitlines():
            idx = line.find("{")
            if idx == -1:
                continue
            try:
                obj = json.loads(line[idx:])
            except json.JSONDecodeError:
                continue
            if obj.get("event") != "request.finished":
                continue
            text = obj.get("obj", {}).get("text")
            rid = obj.get("rid", "")
            if text and not rid.startswith(HEALTH_CHECK_RID_PREFIX):
                reqs.append(text)

        self.assertGreaterEqual(len(reqs), num_conversations * num_turns)

        # Verify prefix relationships
        reqs_sorted = sorted(reqs, key=len)
        prefix_count = 0
        for i, text in enumerate(reqs_sorted):
            for j in range(i + 1, len(reqs_sorted)):
                if reqs_sorted[j].startswith(text):
                    prefix_count += 1
                    break

        expected = num_conversations * (num_turns - 1)
        self.assertGreaterEqual(
            prefix_count, expected, f"Expected at least {expected} prefix pairs"
        )


class TestBenchServingCustomHeaders(CustomTestCase):
    def test_parse_custom_headers(self):
        headers = parse_custom_headers(["MyHeader=MY_VALUE", "Another=value=hello"])
        self.assertEqual(headers, {"MyHeader": "MY_VALUE", "Another": "value=hello"})

        headers = parse_custom_headers(["InvalidNoEquals"])
        self.assertEqual(headers, {})

        headers = parse_custom_headers(["=NoKey"])
        self.assertEqual(headers, {})

    # TODO: Using well-implemented mock server, e.g. the on in sgl-router
    def test_custom_headers_sent_to_server(self):
        import queue

        received_requests = queue.Queue()

        class HeaderEchoHandler(BaseHTTPRequestHandler):
            def _handle(self):
                received_requests.put(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": dict(self.headers),
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if self.path == "/v1/models":
                    self.wfile.write(json.dumps({"data": [{"id": "gpt2"}]}).encode())
                elif self.path == "/generate":
                    self.wfile.write(
                        json.dumps(
                            {"text": "ok", "meta_info": {"completion_tokens": 1}}
                        ).encode()
                    )
                else:
                    self.wfile.write(json.dumps({}).encode())

            do_GET = do_POST = _handle

        server = HTTPServer(("127.0.0.1", 0), HeaderEchoHandler)
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        try:
            args = get_benchmark_args(
                base_url=f"http://127.0.0.1:{port}",
                backend="sglang",
                dataset_name="random",
                tokenizer="gpt2",
                num_prompts=1,
                random_input_len=8,
                random_output_len=8,
                header=["X-Custom-Test=TestValue123", "X-Another=AnotherVal"],
            )
            args.warmup_requests = 0
            args.disable_tqdm = True
            run_benchmark(args)
        except Exception:
            pass
        finally:
            server.shutdown()

        all_reqs = []
        while not received_requests.empty():
            all_reqs.append(received_requests.get_nowait())

        generate_reqs = [r for r in all_reqs if r["path"] == "/generate"]
        self.assertGreater(
            len(generate_reqs),
            0,
            f"No /generate request. All: {[r['path'] for r in all_reqs]}",
        )
        headers = generate_reqs[0]["headers"]
        self.assertEqual(headers.get("X-Custom-Test"), "TestValue123")
        self.assertEqual(headers.get("X-Another"), "AnotherVal")


if __name__ == "__main__":
    unittest.main()
