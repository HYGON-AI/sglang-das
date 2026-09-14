"""Run real TCP content checks with isolated masters, for both reuse modes."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for reuse in [0, 1]:
        port, metrics, client = free_port(), free_port(), free_port()
        while len({port, metrics, client}) != 3:
            port, metrics, client = free_port(), free_port(), free_port()
        master_log = args.output_dir / f"master_reuse{reuse}.log"
        test_log = args.output_dir / f"test_reuse{reuse}.log"
        with master_log.open("w") as log:
            master = subprocess.Popen(
                [
                    os.environ.get("MOONCAKE_MASTER_BINARY", "mooncake_master"),
                    f"--port={port}",
                    f"--metrics_port={metrics}",
                    "--enable_http_metadata_server=false",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 30
                while True:
                    if master.poll() is not None:
                        raise RuntimeError(f"Master exited; see {master_log}")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                            break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise RuntimeError(
                                f"Master startup timed out; see {master_log}"
                            )
                        time.sleep(0.1)
                env = dict(
                    os.environ,
                    MOONCAKE_MASTER=f"127.0.0.1:{port}",
                    MOONCAKE_LOCAL_HOSTNAME=f"127.0.0.1:{client}",
                    MOONCAKE_PROTOCOL="tcp",
                    MOONCAKE_DEVICE="",
                    MOONCAKE_TE_META_DATA_SERVER="P2PHANDSHAKE",
                    E2E_REUSE_RANGES=str(reuse),
                )
                env.pop("MC_USE_TENT", None)
                env.pop("MC_USE_TEV1", None)
                with test_log.open("w") as output:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(
                                Path(__file__).with_name(
                                    "mooncake_read_plan_tcp_smoke.py"
                                )
                            ),
                        ],
                        env=env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        timeout=60,
                    )
                content = test_log.read_text()
                assert completed.returncode == 0, content[-4000:]
                assert "session_ranges_tcp_e2e PASSED" in content
                assert "read_plan_failure_cleanup PASSED" in content
                results.append(
                    dict(
                        reuse=reuse,
                        bytes_checked=49152,
                        status="passed",
                        log=str(test_log),
                    )
                )
                print(
                    f"TCP reuse={reuse}: content and failure cleanup passed", flush=True
                )
            finally:
                master.terminate()
                try:
                    master.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    master.kill()
                    master.wait()
    (args.output_dir / "tcp_result.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
