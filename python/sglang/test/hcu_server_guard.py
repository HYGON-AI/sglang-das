# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Reliable lifecycle management for HCU test servers."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import Optional
from urllib.parse import urlparse

import psutil
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import popen_launch_server

DEFAULT_SERVER_SHUTDOWN_TIMEOUT = 60.0


def _endpoint(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
        raise ValueError(f"invalid server base URL: {base_url!r}")
    return parsed.hostname, parsed.port


def _listener_command(pid: Optional[int]) -> str:
    if pid is None:
        return "<pid unavailable>"
    try:
        command = " ".join(psutil.Process(pid).cmdline())
        return command or psutil.Process(pid).name()
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return "<command unavailable>"


def _tcp_listeners(base_url: str) -> list[dict]:
    host, port = _endpoint(base_url)
    listeners = []
    try:
        for connection in psutil.net_connections(kind="tcp"):
            if connection.status != psutil.CONN_LISTEN or not connection.laddr:
                continue
            if connection.laddr.port != port:
                continue
            listeners.append(
                {
                    "pid": connection.pid,
                    "address": f"{connection.laddr.ip}:{connection.laddr.port}",
                    "command": _listener_command(connection.pid),
                }
            )
    except (psutil.AccessDenied, OSError) as exc:
        try:
            with socket.create_connection((host, port), timeout=1):
                listeners.append(
                    {
                        "pid": None,
                        "address": f"{host}:{port}",
                        "command": f"<psutil unavailable: {exc}>",
                    }
                )
        except OSError:
            pass
    return listeners


def _request_headers(api_key: Optional[str]) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _read_model_info(
    base_url: str, api_key: Optional[str], timeout: float = 60.0
) -> dict:
    response = requests.get(
        base_url.rstrip("/") + "/model_info",
        headers=_request_headers(api_key),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not payload.get("model_path"):
        raise AssertionError(f"invalid /model_info response: {payload}")
    return payload


def _try_read_model_info(base_url: str, api_key: Optional[str]) -> str:
    try:
        payload = _read_model_info(base_url, api_key, timeout=3.0)
        return str(payload.get("model_path"))
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def _normalized_model_path(model_path: str) -> str:
    value = str(model_path).strip().rstrip("/")
    if value.startswith(("/", ".")):
        return os.path.realpath(value)
    return value


def _process_tree_pids(parent_pid: int) -> set[int]:
    try:
        parent = psutil.Process(parent_pid)
        return {parent_pid, *(child.pid for child in parent.children(recursive=True))}
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return {parent_pid}


def _format_listeners(listeners: list[dict]) -> str:
    return "; ".join(
        f"pid={item['pid']} address={item['address']} command={item['command']}"
        for item in listeners
    )


def _assert_port_free(base_url: str, api_key: Optional[str]) -> None:
    listeners = _tcp_listeners(base_url)
    if not listeners:
        return
    model_info = _try_read_model_info(base_url, api_key)
    raise RuntimeError(
        "HCU server port is already occupied before launch: "
        f"base_url={base_url} model={model_info} listeners=[{_format_listeners(listeners)}]"
    )


def _assert_server_identity(
    process: subprocess.Popen,
    base_url: str,
    expected_model_path: str,
    api_key: Optional[str],
) -> None:
    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(
            f"HCU server launcher exited before identity check: pid={process.pid} "
            f"return_code={return_code}"
        )

    payload = _read_model_info(base_url, api_key)
    actual_model_path = str(payload["model_path"])
    if _normalized_model_path(actual_model_path) != _normalized_model_path(
        expected_model_path
    ):
        raise RuntimeError(
            "HCU server model identity mismatch: "
            f"expected={expected_model_path!r} actual={actual_model_path!r} "
            f"launcher_pid={process.pid}"
        )

    return_code = process.poll()
    if return_code is not None:
        raise RuntimeError(
            f"HCU server launcher exited during identity check: pid={process.pid} "
            f"return_code={return_code}"
        )

    listeners = _tcp_listeners(base_url)
    if not listeners:
        raise RuntimeError(
            f"HCU server has no TCP listener after startup: base_url={base_url}"
        )
    listener_pids = {item["pid"] for item in listeners if item["pid"] is not None}
    process_pids = _process_tree_pids(process.pid)
    foreign_pids = listener_pids - process_pids
    if foreign_pids:
        raise RuntimeError(
            "HCU server listener does not belong to the launched process tree: "
            f"launcher_pid={process.pid} process_tree={sorted(process_pids)} "
            f"listeners=[{_format_listeners(listeners)}]"
        )
    if not listener_pids:
        print(
            "HCU server listener PID is unavailable; model identity was verified: "
            f"model={actual_model_path} base_url={base_url}"
        )
    else:
        print(
            "HCU server identity verified: "
            f"expected={expected_model_path} actual={actual_model_path} "
            f"launcher_pid={process.pid} listener_pids={sorted(listener_pids)}"
        )


def _wait_for_port_release(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        listeners = _tcp_listeners(base_url)
        if not listeners:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "HCU server port remained occupied after shutdown: "
                f"base_url={base_url} listeners=[{_format_listeners(listeners)}]"
            )
        time.sleep(min(0.5, remaining))


class HcuServerGuard:
    """Launch an HCU test server and prove requests reach that exact process/model."""

    def __init__(
        self,
        model_path: str,
        base_url: str,
        *,
        timeout: float,
        other_args: Optional[list[str]] = None,
        env: Optional[dict] = None,
        api_key: Optional[str] = None,
        shutdown_timeout: float = DEFAULT_SERVER_SHUTDOWN_TIMEOUT,
    ):
        self.model_path = model_path
        self.base_url = base_url
        self.timeout = timeout
        self.other_args = list(other_args or ())
        self.env = env
        self.api_key = api_key
        self.shutdown_timeout = shutdown_timeout
        self.process: Optional[subprocess.Popen] = None

    def __enter__(self):
        _assert_port_free(self.base_url, self.api_key)
        try:
            self.process = popen_launch_server(
                self.model_path,
                self.base_url,
                timeout=self.timeout,
                api_key=self.api_key,
                other_args=self.other_args,
                env=self.env,
            )
            _assert_server_identity(
                self.process,
                self.base_url,
                self.model_path,
                self.api_key,
            )
        except Exception as launch_error:
            if self.process is not None:
                try:
                    self.stop()
                except Exception as cleanup_error:
                    raise RuntimeError(
                        f"{launch_error}; HCU server cleanup also failed: {cleanup_error}"
                    ) from launch_error
            raise
        return self

    def stop(self) -> None:
        process = self.process
        self.process = None
        deadline = time.monotonic() + self.shutdown_timeout
        errors = []

        if process is not None:
            try:
                remaining = max(0.1, deadline - time.monotonic())
                kill_process_tree(process.pid, wait_timeout=remaining)
            except Exception as exc:
                errors.append(f"kill process tree failed: {exc}")
            try:
                remaining = max(0.1, deadline - time.monotonic())
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    remaining = max(0.1, deadline - time.monotonic())
                    process.wait(timeout=remaining)
                except Exception as exc:
                    errors.append(f"reap launcher pid={process.pid} failed: {exc}")
            except ChildProcessError:
                # psutil may already have reaped the launcher.
                pass
            except Exception as exc:
                errors.append(f"wait launcher pid={process.pid} failed: {exc}")

        try:
            _wait_for_port_release(self.base_url, max(0.0, deadline - time.monotonic()))
        except Exception as exc:
            errors.append(str(exc))

        if errors:
            raise RuntimeError("; ".join(errors))

    def __exit__(self, exc_type, exc, tb):
        try:
            self.stop()
        except Exception as cleanup_error:
            if exc_type is not None:
                print(
                    "HCU server cleanup failed while handling another test error: "
                    f"{cleanup_error}"
                )
                return False
            raise
        return False
