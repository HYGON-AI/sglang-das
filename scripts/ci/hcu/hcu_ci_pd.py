#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HCU_PD_UTILS_PATH = REPO_ROOT / "python/sglang/test/hcu_pd_utils.py"
HCU_PD_UTILS_SPEC = importlib.util.spec_from_file_location(
    "_sglang_hcu_pd_utils", HCU_PD_UTILS_PATH
)
if HCU_PD_UTILS_SPEC is None or HCU_PD_UTILS_SPEC.loader is None:
    raise RuntimeError(f"cannot load HCU PD utilities from {HCU_PD_UTILS_PATH}")
HCU_PD_UTILS = importlib.util.module_from_spec(HCU_PD_UTILS_SPEC)
sys.modules[HCU_PD_UTILS_SPEC.name] = HCU_PD_UTILS
HCU_PD_UTILS_SPEC.loader.exec_module(HCU_PD_UTILS)

BOOTSTRAP_PORT = HCU_PD_UTILS.BOOTSTRAP_PORT
DECODE_PORT = HCU_PD_UTILS.DECODE_PORT
MINIMAX_M27_MODEL_ENV = HCU_PD_UTILS.MINIMAX_M27_MODEL_ENV
PREFILL_PORT = HCU_PD_UTILS.PREFILL_PORT
ROUTER_PORT = HCU_PD_UTILS.ROUTER_PORT
HcuPDRoleConfig = HCU_PD_UTILS.HcuPDRoleConfig
minimax_m27_pd_env = HCU_PD_UTILS.minimax_m27_pd_env
minimax_m27_router_command = HCU_PD_UTILS.minimax_m27_router_command
minimax_m27_server_args = HCU_PD_UTILS.minimax_m27_server_args
minimax_m27_server_command = HCU_PD_UTILS.minimax_m27_server_command
resolve_minimax_m27_model_path = HCU_PD_UTILS.resolve_minimax_m27_model_path

ROLE_PREFILL = "prefill"
ROLE_DECODE = "decode"
VALID_ROLES = {ROLE_PREFILL, ROLE_DECODE}
COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_DIR_RE = re.compile(r"^run-[0-9]+$")


class PDInfrastructureError(RuntimeError):
    pass


class PDPeerAbort(PDInfrastructureError):
    pass


def _now_payload() -> dict[str, Any]:
    return {
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise PDInfrastructureError(f"required environment variable is empty: {name}")
    return value


def _safe_component(name: str, value: str) -> str:
    if not COMPONENT_RE.fullmatch(value):
        raise PDInfrastructureError(f"unsafe {name}: {value!r}")
    return value


def _set_shared_permissions(path: Path, *, is_dir: bool, gid: int) -> None:
    desired_mode = 0o2775 if is_dir else 0o664
    required_group_mode = 0o070 if is_dir else 0o060

    try:
        path.chmod(desired_mode)
        if (
            hasattr(os, "chown")
            and hasattr(os, "getgid")
            and hasattr(os, "getgroups")
            and gid in {os.getgid(), *os.getgroups()}
        ):
            os.chown(path, -1, gid)
    except PermissionError:
        pass

    stat_result = path.stat()
    if os.name == "posix" and (
        stat_result.st_gid != gid
        or stat_result.st_mode & required_group_mode != required_group_mode
    ):
        raise PDInfrastructureError(
            f"shared path has incompatible ownership or mode: {path} "
            f"gid={stat_result.st_gid}, mode={oct(stat_result.st_mode & 0o7777)}, "
            f"expected gid={gid} with group "
            f"{'rwx' if is_dir else 'rw'} permission"
        )


def ensure_shared_dir(path: Path, gid: int) -> None:
    missing_paths = []
    cursor = path
    while not cursor.exists():
        missing_paths.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    if not missing_paths:
        _set_shared_permissions(path, is_dir=True, gid=gid)
        return
    for created_path in reversed(missing_paths):
        _set_shared_permissions(created_path, is_dir=True, gid=gid)


def atomic_write_json(path: Path, payload: dict[str, Any], gid: int) -> None:
    ensure_shared_dir(path.parent, gid)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        _set_shared_permissions(temporary, is_dir=False, gid=gid)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_once(path: Path, payload: dict[str, Any], gid: int) -> bool:
    ensure_shared_dir(path.parent, gid)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    _set_shared_permissions(path, is_dir=False, gid=gid)
    return True


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PDInfrastructureError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PDInfrastructureError(f"JSON object expected in {path}")
    return payload


def run_checked(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> str:
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise PDInfrastructureError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-8000:]}"
        )
    return completed.stdout.strip()


@dataclass(frozen=True)
class PDRunContext:
    role: str
    run_id: str
    attempt: str
    sha: str
    target_ref: str
    runner_name: str
    hostname: str
    image: str
    image_id: str
    model_path: str
    local_ip: str
    peer_ip: str
    prefill_ip: str
    decode_ip: str
    ifname: str
    ib_device: str
    gid_index: str
    checkout: Path
    shared_root: Path
    wheel_root: Path
    shared_gid: int
    peer_timeout: int
    service_timeout: int
    heartbeat_timeout: int
    completion_timeout: int

    @classmethod
    def from_environment(cls, role: str) -> "PDRunContext":
        if role not in VALID_ROLES:
            raise PDInfrastructureError(f"unsupported role: {role}")
        run_id = _safe_component("run ID", _require_env("GITHUB_RUN_ID"))
        attempt = _safe_component(
            "run attempt", os.environ.get("GITHUB_RUN_ATTEMPT", "1")
        )
        sha = _require_env("HCU_PD_SHA").lower()
        if not SHA_RE.fullmatch(sha):
            raise PDInfrastructureError(f"HCU_PD_SHA must be a full Git SHA: {sha}")

        prefill_ip = _require_env("HCU_PD_PREFILL_IP")
        decode_ip = _require_env("HCU_PD_DECODE_IP")
        expected_local_ip = prefill_ip if role == ROLE_PREFILL else decode_ip
        expected_peer_ip = decode_ip if role == ROLE_PREFILL else prefill_ip
        local_ip = _require_env("HCU_PD_LOCAL_IP")
        peer_ip = _require_env("HCU_PD_PEER_IP")
        if local_ip != expected_local_ip or peer_ip != expected_peer_ip:
            raise PDInfrastructureError(
                f"role IP mismatch for {role}: local={local_ip}, peer={peer_ip}, "
                f"expected local={expected_local_ip}, peer={expected_peer_ip}"
            )
        gid_index = os.environ.get("HCU_PD_GID_INDEX", "3").strip()
        if not gid_index.isdigit():
            raise PDInfrastructureError(
                f"HCU_PD_GID_INDEX must be a non-negative integer: {gid_index!r}"
            )

        checkout = Path(
            os.environ.get("HCU_PD_CHECKOUT", os.environ.get("GITHUB_WORKSPACE", ""))
        ).resolve()
        if not checkout.is_dir():
            raise PDInfrastructureError(f"checkout does not exist: {checkout}")

        return cls(
            role=role,
            run_id=run_id,
            attempt=attempt,
            sha=sha,
            target_ref=os.environ.get("HCU_PD_TARGET_REF", sha),
            runner_name=_require_env("RUNNER_NAME"),
            hostname=socket.gethostname(),
            image=_require_env("HCU_PD_IMAGE"),
            image_id=_require_env("HCU_PD_IMAGE_ID"),
            model_path=resolve_minimax_m27_model_path(),
            local_ip=local_ip,
            peer_ip=peer_ip,
            prefill_ip=prefill_ip,
            decode_ip=decode_ip,
            ifname=_require_env("HCU_PD_LOCAL_IFNAME"),
            ib_device=_require_env("HCU_PD_LOCAL_IB_DEVICE"),
            gid_index=gid_index,
            checkout=checkout,
            shared_root=Path(
                os.environ.get("HCU_PD_SHARED_ROOT", "/ci_public/sglang-das/hcu-pd")
            ).resolve(),
            wheel_root=Path(
                os.environ.get("HCU_PD_WHEEL_ROOT", "/ci_public/sglang-das/hcu-wheels")
            ).resolve(),
            shared_gid=int(os.environ.get("HCU_PD_SHARED_GID", "1002")),
            peer_timeout=int(os.environ.get("HCU_PD_PEER_TIMEOUT", "1800")),
            service_timeout=int(os.environ.get("HCU_PD_SERVICE_TIMEOUT", "7200")),
            heartbeat_timeout=int(os.environ.get("HCU_PD_HEARTBEAT_TIMEOUT", "180")),
            completion_timeout=int(
                os.environ.get("HCU_PD_COMPLETION_TIMEOUT", "10800")
            ),
        )

    @property
    def peer_role(self) -> str:
        return ROLE_DECODE if self.role == ROLE_PREFILL else ROLE_PREFILL

    @property
    def run_dir(self) -> Path:
        return self.shared_root / f"run-{self.run_id}" / f"attempt-{self.attempt}"

    @property
    def role_dir(self) -> Path:
        return self.run_dir / self.role

    @property
    def peer_dir(self) -> Path:
        return self.run_dir / self.peer_role

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def results_dir(self) -> Path:
        return self.run_dir / "results"

    @property
    def abort_path(self) -> Path:
        return self.run_dir / "abort.json"

    @property
    def done_path(self) -> Path:
        return self.run_dir / "done.json"

    @property
    def container_name(self) -> str:
        return f"ci_sglang_hcu_pd_{self.run_id}_{self.attempt}_{self.role}"

    @property
    def role_port(self) -> int:
        return PREFILL_PORT if self.role == ROLE_PREFILL else DECODE_PORT

    @property
    def wheel_bundle(self) -> Path:
        return self.wheel_root / self.sha

    def role_config(self) -> HcuPDRoleConfig:
        return HcuPDRoleConfig(
            role=self.role,
            host_ip=self.local_ip,
            ifname=self.ifname,
            ib_device=self.ib_device,
        )

    def claim_payload(self) -> dict[str, Any]:
        return {
            **_now_payload(),
            "role": self.role,
            "run_id": self.run_id,
            "run_attempt": self.attempt,
            "commit_sha": self.sha,
            "target_ref": self.target_ref,
            "runner_name": self.runner_name,
            "hostname": self.hostname,
            "image": self.image,
            "image_id": self.image_id,
            "model_path": self.model_path,
            "ip": self.local_ip,
            "ifname": self.ifname,
            "ib_device": self.ib_device,
            "gid_index": self.gid_index,
        }


class Heartbeat:
    def __init__(self, context: PDRunContext, interval: float = 10.0):
        self.context = context
        self.interval = interval
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run, name=f"hcu-pd-{context.role}-heartbeat", daemon=True
        )

    def start(self) -> None:
        self._write()
        self.thread.start()

    def _write(self) -> None:
        atomic_write_json(
            self.context.role_dir / "heartbeat.json",
            {
                **_now_payload(),
                "role": self.context.role,
                "runner_name": self.context.runner_name,
                "hostname": self.context.hostname,
            },
            self.context.shared_gid,
        )

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self._write()
            except BaseException as exc:
                self.error = exc
                return

    def check(self) -> None:
        if self.error is not None:
            raise PDInfrastructureError(
                f"heartbeat writer failed: {self.error}"
            ) from self.error

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=self.interval + 2)


class PDOrchestrator:
    def __init__(self, context: PDRunContext):
        self.context = context
        self.heartbeat = Heartbeat(context)
        self.processes: dict[str, subprocess.Popen] = {}
        self.log_streams: list[Any] = []

    def prepare_layout(self) -> None:
        for path in (
            self.context.shared_root,
            self.context.run_dir,
            self.context.role_dir,
            self.context.logs_dir,
            self.context.results_dir,
            self.context.wheel_root,
        ):
            ensure_shared_dir(path, self.context.shared_gid)

    def prune_old_runs(self) -> None:
        if self.context.role != ROLE_PREFILL:
            return
        cutoff = time.time() - 14 * 24 * 60 * 60
        for candidate in self.context.shared_root.iterdir():
            if not candidate.is_dir() or not RUN_DIR_RE.fullmatch(candidate.name):
                continue
            if candidate == self.context.run_dir.parent:
                continue
            if candidate.stat().st_mtime >= cutoff:
                continue
            if candidate.parent.resolve() != self.context.shared_root:
                continue
            shutil.rmtree(candidate)

    def _check_abort(self) -> None:
        if self.context.abort_path.exists():
            payload = read_json(self.context.abort_path)
            raise PDPeerAbort(
                f"PD run aborted by {payload.get('role', 'unknown')}: "
                f"{payload.get('error', payload)}"
            )

    def _peer_heartbeat_age(self) -> float | None:
        path = self.context.peer_dir / "heartbeat.json"
        if not path.exists():
            return None
        payload = read_json(path)
        timestamp = float(payload.get("timestamp", 0))
        return time.time() - timestamp

    def _check_peer_alive(self) -> None:
        age = self._peer_heartbeat_age()
        if age is not None and age > self.context.heartbeat_timeout:
            raise PDInfrastructureError(
                f"{self.context.peer_role} heartbeat is stale: {age:.1f}s"
            )
        self.heartbeat.check()
        self._check_abort()

    def wait_for_path(
        self, path: Path, *, timeout: float, require_peer_heartbeat: bool = True
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check_abort()
            self.heartbeat.check()
            if path.exists():
                return
            if require_peer_heartbeat:
                self._check_peer_alive()
            time.sleep(2)
        raise PDInfrastructureError(f"timed out waiting for {path}")

    def write_claim_and_wait(self) -> dict[str, Any]:
        atomic_write_json(
            self.context.role_dir / "claim.json",
            self.context.claim_payload(),
            self.context.shared_gid,
        )
        self.heartbeat.start()
        peer_claim_path = self.context.peer_dir / "claim.json"
        self.wait_for_path(
            peer_claim_path,
            timeout=self.context.peer_timeout,
            require_peer_heartbeat=False,
        )
        peer = read_json(peer_claim_path)
        self._validate_peer_claim(peer)
        return peer

    def _validate_peer_claim(self, peer: dict[str, Any]) -> None:
        expected = {
            "role": self.context.peer_role,
            "run_id": self.context.run_id,
            "run_attempt": self.context.attempt,
            "commit_sha": self.context.sha,
            "target_ref": self.context.target_ref,
            "image": self.context.image,
            "image_id": self.context.image_id,
            "model_path": self.context.model_path,
            "ip": self.context.peer_ip,
            "gid_index": self.context.gid_index,
        }
        mismatches = {
            key: {"expected": value, "actual": peer.get(key)}
            for key, value in expected.items()
            if peer.get(key) != value
        }
        if peer.get("runner_name") == self.context.runner_name:
            mismatches["runner_name"] = {
                "expected": "different runners",
                "actual": peer.get("runner_name"),
            }
        if peer.get("hostname") == self.context.hostname:
            mismatches["hostname"] = {
                "expected": "different physical hosts",
                "actual": peer.get("hostname"),
            }
        if mismatches:
            raise PDInfrastructureError(f"peer claim mismatch: {mismatches}")

    def _check_port_available(self, host: str, port: int) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError as exc:
                raise PDInfrastructureError(
                    f"port is unavailable on {host}:{port}: {exc}"
                ) from exc

    def _interface_ipv4(self, ifname: str) -> str:
        import fcntl
        import struct

        if len(ifname.encode()) > 15:
            raise PDInfrastructureError(
                f"network interface name is too long: {ifname!r}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            try:
                request = struct.pack("256s", ifname.encode())
                response = fcntl.ioctl(sock.fileno(), 0x8915, request)
            except OSError as exc:
                raise PDInfrastructureError(
                    f"cannot read IPv4 address for {ifname}: {exc}"
                ) from exc
        return socket.inet_ntoa(response[20:24])

    def preflight(self) -> dict[str, Any]:
        if not Path(self.context.model_path).is_dir():
            raise PDInfrastructureError(
                f"MiniMax-M2.7 model directory is missing: {self.context.model_path}"
            )
        if not (Path(self.context.model_path) / "config.json").is_file():
            raise PDInfrastructureError(
                f"model config.json is missing: {self.context.model_path}"
            )
        if not any(Path(self.context.model_path).glob("*.safetensors")) and not any(
            Path(self.context.model_path).glob("*.safetensors.index.json")
        ):
            raise PDInfrastructureError(
                f"model weights are missing: {self.context.model_path}"
            )

        interface_ip = self._interface_ipv4(self.context.ifname)
        if self.context.local_ip != interface_ip:
            raise PDInfrastructureError(
                f"{self.context.ifname} has IPv4 {interface_ip}, "
                f"expected {self.context.local_ip}"
            )

        rdma_root = Path("/sys/class/infiniband") / self.context.ib_device
        state_path = rdma_root / "ports" / "1" / "state"
        if not state_path.is_file() or "ACTIVE" not in state_path.read_text():
            raise PDInfrastructureError(
                f"RDMA device is not ACTIVE: {self.context.ib_device}"
            )
        net_path = rdma_root / "device" / "net" / self.context.ifname
        if not net_path.exists():
            mapped_interfaces = sorted(
                path.name for path in (rdma_root / "device" / "net").glob("*")
            )
            raise PDInfrastructureError(
                f"RDMA mapping mismatch: {self.context.ib_device} maps to "
                f"{mapped_interfaces}, expected {self.context.ifname}"
            )
        gid_path = rdma_root / "ports" / "1" / "gids" / self.context.gid_index
        gid_type_path = (
            rdma_root / "ports" / "1" / "gid_attrs" / "types" / self.context.gid_index
        )
        gid_ndev_path = (
            rdma_root / "ports" / "1" / "gid_attrs" / "ndevs" / self.context.gid_index
        )
        if not all(path.is_file() for path in (gid_path, gid_type_path, gid_ndev_path)):
            raise PDInfrastructureError(
                f"RDMA GID index {self.context.gid_index} is unavailable on "
                f"{self.context.ib_device}"
            )
        gid_value = gid_path.read_text().strip()
        gid_type = gid_type_path.read_text().strip()
        gid_ndev = gid_ndev_path.read_text().strip()
        try:
            mapped_ip = ipaddress.IPv6Address(gid_value).ipv4_mapped
        except ipaddress.AddressValueError as exc:
            raise PDInfrastructureError(
                f"invalid RDMA GID value at index {self.context.gid_index}: "
                f"{gid_value!r}"
            ) from exc
        if (
            mapped_ip != ipaddress.IPv4Address(self.context.local_ip)
            or gid_type != "RoCE v2"
            or gid_ndev != self.context.ifname
        ):
            raise PDInfrastructureError(
                f"RDMA GID mismatch at index {self.context.gid_index}: "
                f"gid={gid_value}, type={gid_type}, ndev={gid_ndev}; "
                f"expected IPv4={self.context.local_ip}, type=RoCE v2, "
                f"ndev={self.context.ifname}"
            )

        image_id = run_checked(
            ["docker", "image", "inspect", "--format", "{{.Id}}", self.context.image]
        )
        if image_id != self.context.image_id:
            raise PDInfrastructureError(
                f"local image ID changed: expected={self.context.image_id}, actual={image_id}"
            )

        hcu_output = run_checked(["hy-smi", "--showbus"])
        hcu_count = len(set(re.findall(r"HCU\[(\d+)\]", hcu_output)))
        if hcu_count != 8:
            raise PDInfrastructureError(f"expected 8 HCUs, found {hcu_count}")

        self._check_port_available(self.context.local_ip, self.context.role_port)
        if self.context.role == ROLE_PREFILL:
            self._check_port_available(self.context.local_ip, ROUTER_PORT)
            self._check_port_available(self.context.local_ip, BOOTSTRAP_PORT)

        probe_path = (
            self.context.role_dir
            / f".write-probe-{self.context.hostname}-{os.getpid()}"
        )
        probe_path.write_text("ok\n")
        _set_shared_permissions(probe_path, is_dir=False, gid=self.context.shared_gid)
        probe_path.unlink()

        payload = {
            **_now_payload(),
            "role": self.context.role,
            "hcu_count": hcu_count,
            "image_id": image_id,
            "model_path": self.context.model_path,
            "ip": self.context.local_ip,
            "ifname": self.context.ifname,
            "ib_device": self.context.ib_device,
            "gid_index": self.context.gid_index,
            "gid": gid_value,
            "gid_type": gid_type,
            "rdma_state": state_path.read_text().strip(),
        }
        atomic_write_json(
            self.context.role_dir / "preflight.json",
            payload,
            self.context.shared_gid,
        )
        return payload

    def _open_log(self, name: str):
        path = self.context.logs_dir / name
        ensure_shared_dir(path.parent, self.context.shared_gid)
        stream = path.open("ab", buffering=0)
        _set_shared_permissions(path, is_dir=False, gid=self.context.shared_gid)
        self.log_streams.append(stream)
        return stream

    def _run_monitored(
        self,
        name: str,
        command: list[str],
        *,
        env: dict[str, str] | None,
        timeout: float,
        log_name: str,
    ) -> None:
        stream = self._open_log(log_name)
        stream.write(f"$ {' '.join(command)}\n".encode())
        process = subprocess.Popen(
            command,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if time.monotonic() >= deadline:
                process.terminate()
                raise PDInfrastructureError(f"{name} timed out after {timeout}s")
            self._check_peer_alive()
            time.sleep(5)
        if process.returncode != 0:
            tail = Path(stream.name).read_text(errors="replace")[-12000:]
            raise PDInfrastructureError(
                f"{name} failed with exit code {process.returncode}\n{tail}"
            )

    def ensure_shared_wheels(self) -> dict[str, Any]:
        ready = self.context.wheel_bundle / "READY"
        if self.context.role == ROLE_PREFILL and not ready.exists():
            env = os.environ.copy()
            env.update(
                {
                    "HCU_PD_SHA": self.context.sha,
                    "HCU_PD_CHECKOUT": str(self.context.checkout),
                    "HCU_PD_IMAGE": self.context.image,
                    "HCU_PD_WHEEL_ROOT": str(self.context.wheel_root),
                    "HCU_PD_SHARED_GID": str(self.context.shared_gid),
                }
            )
            self._run_monitored(
                "shared wheel build",
                [
                    "bash",
                    str(
                        self.context.checkout
                        / "scripts/ci/hcu/hcu_ci_build_shared_wheels.sh"
                    ),
                    "build",
                ],
                env=env,
                timeout=self.context.completion_timeout,
                log_name="wheel-build.log",
            )
        else:
            self.wait_for_path(
                ready,
                timeout=self.context.completion_timeout,
                require_peer_heartbeat=True,
            )

        manifest = read_json(self.context.wheel_bundle / "manifest.json")
        if manifest.get("commit_sha") != self.context.sha:
            raise PDInfrastructureError(
                f"wheel manifest SHA mismatch: {manifest.get('commit_sha')}"
            )
        kinds = {item.get("kind") for item in manifest.get("wheels", [])}
        if kinds != {"sglang", "sglang-kernel", "sglang-router"}:
            raise PDInfrastructureError(f"wheel bundle is incomplete: {kinds}")
        for item in manifest["wheels"]:
            path = self.context.wheel_bundle / item["path"]
            if not path.is_file():
                raise PDInfrastructureError(f"wheel is missing: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item.get("sha256"):
                raise PDInfrastructureError(f"wheel checksum mismatch: {path}")
        return manifest

    def cleanup_container(self) -> None:
        cleanup_script = (
            self.context.checkout / "scripts/ci/hcu/hcu_ci_cleanup_container.sh"
        )
        env = os.environ.copy()
        env.update(
            {
                "HCU_CI_CONTAINER_NAME": self.context.container_name,
                "GITHUB_WORKSPACE": str(self.context.checkout),
            }
        )
        completed = subprocess.run(
            ["bash", str(cleanup_script)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            raise PDInfrastructureError(
                f"container cleanup failed: {completed.stdout[-4000:]}"
            )

    def start_container(self, manifest: dict[str, Any]) -> None:
        self.cleanup_container()
        env = os.environ.copy()
        env.update(
            {
                "HCU_CI_IMAGE": self.context.image,
                "HCU_CI_CONTAINER_NAME": self.context.container_name,
                "HCU_CI_NETWORK_MODE": "host",
                "HCU_CI_SHM_SIZE": "200g",
                "HCU_CI_ENABLE_RDMA": "1",
                "HCU_CI_SKIP_PULL": "1",
                "HCU_MODEL_EXTRA_HOST_PATHS": "/public4",
                "HCU_WHEEL_STAGING_ROOT": str(self.context.wheel_bundle),
                "HCU_WHEEL_STAGING_CONTAINER_ROOT": "/hcu-pd-wheels",
                "GITHUB_WORKSPACE": str(self.context.checkout),
            }
        )
        run_checked(
            [
                "bash",
                str(self.context.checkout / "scripts/ci/hcu/hcu_ci_start_container.sh"),
            ],
            env=env,
            timeout=300,
        )
        actual_image_id = run_checked(
            [
                "docker",
                "inspect",
                "--format",
                "{{.Image}}",
                self.context.container_name,
            ]
        )
        if actual_image_id != self.context.image_id:
            raise PDInfrastructureError(
                f"container image mismatch: expected={self.context.image_id}, "
                f"actual={actual_image_id}"
            )

        hcu_count = run_checked(
            [
                "docker",
                "exec",
                self.context.container_name,
                "python3",
                "-c",
                "import torch; print(torch.cuda.device_count())",
            ],
            timeout=120,
        )
        if hcu_count.splitlines()[-1].strip() != "8":
            raise PDInfrastructureError(f"container does not see 8 HCUs: {hcu_count}")
        memlock = run_checked(
            [
                "docker",
                "exec",
                self.context.container_name,
                "bash",
                "-lc",
                "ulimit -l",
            ]
        )
        if memlock.strip() != "unlimited":
            raise PDInfrastructureError(
                f"container memlock is not unlimited: {memlock}"
            )

        wheel_priority = {
            "sglang-kernel": 0,
            "sglang": 1,
            "sglang-router": 2,
        }
        wheel_paths = [
            f"/hcu-pd-wheels/{item['path']}"
            for item in sorted(
                manifest["wheels"], key=lambda item: wheel_priority[item["kind"]]
            )
        ]
        run_checked(
            [
                "docker",
                "exec",
                self.context.container_name,
                "python3",
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                *wheel_paths,
            ],
            timeout=900,
        )
        versions = run_checked(
            [
                "docker",
                "exec",
                self.context.container_name,
                "python3",
                "-c",
                (
                    "from importlib import metadata; "
                    "print('|'.join(metadata.version(n) for n in "
                    "('sglang','sglang-kernel','sglang-router')))"
                ),
            ]
        )
        atomic_write_json(
            self.context.role_dir / "container.json",
            {
                **_now_payload(),
                "container_name": self.context.container_name,
                "image_id": actual_image_id,
                "versions": versions,
            },
            self.context.shared_gid,
        )

    def _start_container_process(
        self,
        name: str,
        command: list[str],
        *,
        env: dict[str, str],
        log_name: str,
    ) -> subprocess.Popen:
        stream = self._open_log(log_name)
        docker_command = ["docker", "exec", "-w", "/sglang-checkout"]
        for key, value in sorted(env.items()):
            docker_command.extend(["-e", f"{key}={value}"])
        docker_command.extend([self.context.container_name, *command])
        stream.write(f"$ {' '.join(docker_command)}\n".encode())
        process = subprocess.Popen(
            docker_command,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        self.processes[name] = process
        return process

    def _http_json(self, url: str, timeout: float = 5.0) -> dict[str, Any]:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise PDInfrastructureError(
                    f"unexpected HTTP status from {url}: {response.status}"
                )
            body = response.read()
        if not body:
            return {}
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return {"text": body.decode(errors="replace")}
        return payload if isinstance(payload, dict) else {"payload": payload}

    def _wait_service(
        self,
        *,
        base_url: str,
        process: subprocess.Popen,
        expected_model: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            self._check_peer_alive()
            if process.poll() is not None:
                raise PDInfrastructureError(
                    f"service process exited early with code {process.returncode}"
                )
            try:
                self._http_json(f"{base_url}/health")
                if expected_model is None:
                    return {}
                model_info = self._http_json(f"{base_url}/model_info")
                actual = str(model_info.get("model_path", "")).rstrip("/")
                if actual != expected_model.rstrip("/"):
                    raise PDInfrastructureError(
                        f"model identity mismatch at {base_url}: "
                        f"expected={expected_model!r}, actual={actual!r}"
                    )
                return model_info
            except (
                OSError,
                urllib.error.URLError,
                json.JSONDecodeError,
                PDInfrastructureError,
            ) as exc:
                last_error = exc
                time.sleep(5)
        raise PDInfrastructureError(
            f"service did not become ready at {base_url}: {last_error}"
        )

    def launch_role_service(self) -> dict[str, Any]:
        role_config = self.context.role_config()
        service_env = minimax_m27_pd_env(role_config, gid_index=self.context.gid_index)
        service_env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONUNBUFFERED": "1",
                "NCCL_DEBUG": "INFO",
                "SGLANG_IS_IN_CI": "1",
                "SGLANG_IS_IN_CI_HCU": "1",
            }
        )
        process = self._start_container_process(
            self.context.role,
            minimax_m27_server_command(role_config, self.context.model_path),
            env=service_env,
            log_name=f"{self.context.role}.log",
        )
        base_url = f"http://{self.context.local_ip}:{self.context.role_port}"
        model_info = self._wait_service(
            base_url=base_url,
            process=process,
            expected_model=self.context.model_path,
            timeout=self.context.service_timeout,
        )
        payload = {
            **_now_payload(),
            "role": self.context.role,
            "base_url": base_url,
            "model_path": model_info.get("model_path"),
            "launcher_pid": process.pid,
        }
        atomic_write_json(
            self.context.role_dir / "ready.json",
            payload,
            self.context.shared_gid,
        )
        return payload

    def launch_router(self) -> subprocess.Popen:
        process = self._start_container_process(
            "router",
            minimax_m27_router_command(self.context.prefill_ip, self.context.decode_ip),
            env={"PYTHONUNBUFFERED": "1"},
            log_name="router.log",
        )
        self._wait_service(
            base_url=f"http://{self.context.prefill_ip}:{ROUTER_PORT}",
            process=process,
            expected_model=None,
            timeout=1800,
        )
        return process

    def run_smoke_test(self) -> dict[str, Any]:
        result_path = "/sglang-checkout/test-results/hcu-pd/smoke-result.json"
        command = [
            "docker",
            "exec",
            "-w",
            "/sglang-checkout/test",
            "-e",
            "SGLANG_IS_IN_CI=1",
            "-e",
            "SGLANG_IS_IN_CI_HCU=1",
            "-e",
            "HCU_CI_USE_INSTALLED_WHEELS=1",
            "-e",
            f"HCU_PD_PREFILL_IP={self.context.prefill_ip}",
            "-e",
            f"HCU_PD_DECODE_IP={self.context.decode_ip}",
            "-e",
            f"{MINIMAX_M27_MODEL_ENV}={self.context.model_path}",
            "-e",
            f"HCU_PD_SMOKE_RESULT_PATH={result_path}",
            self.context.container_name,
            "python3",
            "run_suite.py",
            "--hw",
            "hcu",
            "--suite",
            "nightly-hcu-disaggregation-16",
            "--nightly",
        ]
        self._run_monitored(
            "MiniMax-M2.7 PD smoke",
            command,
            env=None,
            timeout=1800,
            log_name="smoke.log",
        )
        raw_result = run_checked(
            [
                "docker",
                "exec",
                self.context.container_name,
                "cat",
                result_path,
            ]
        )
        try:
            result = json.loads(raw_result)
        except json.JSONDecodeError as exc:
            raise PDInfrastructureError(
                f"invalid smoke-result.json: {raw_result}"
            ) from exc
        atomic_write_json(
            self.context.results_dir / "smoke-result.json",
            result,
            self.context.shared_gid,
        )
        return result

    def wait_for_completion(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.context.completion_timeout
        service = self.processes[self.context.role]
        while time.monotonic() < deadline:
            self._check_peer_alive()
            if service.poll() is not None:
                raise PDInfrastructureError(
                    f"{self.context.role} service exited with code {service.returncode}"
                )
            if self.context.done_path.exists():
                payload = read_json(self.context.done_path)
                if not payload.get("success"):
                    raise PDInfrastructureError(f"PD run failed: {payload}")
                return payload
            time.sleep(5)
        raise PDInfrastructureError("timed out waiting for Prefill completion")

    def run_core(self) -> dict[str, Any]:
        self.prepare_layout()
        self.prune_old_runs()
        self.preflight()
        self.write_claim_and_wait()
        manifest = self.ensure_shared_wheels()
        self.start_container(manifest)
        role_ready = self.launch_role_service()

        if self.context.role == ROLE_DECODE:
            done = self.wait_for_completion()
            return {"role_ready": role_ready, "done": done}

        self.wait_for_path(
            self.context.peer_dir / "ready.json",
            timeout=self.context.service_timeout,
            require_peer_heartbeat=True,
        )
        peer_ready = read_json(self.context.peer_dir / "ready.json")
        if str(peer_ready.get("model_path", "")).rstrip("/") != self.context.model_path:
            raise PDInfrastructureError(f"Decode model identity mismatch: {peer_ready}")
        self.launch_router()
        smoke = self.run_smoke_test()
        return {"role_ready": role_ready, "peer_ready": peer_ready, "smoke": smoke}

    def write_abort(self, error: BaseException) -> None:
        write_json_once(
            self.context.abort_path,
            {
                **_now_payload(),
                "role": self.context.role,
                "runner_name": self.context.runner_name,
                "hostname": self.context.hostname,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            self.context.shared_gid,
        )

    def execute(self) -> dict[str, Any]:
        started_at = time.monotonic()
        result: dict[str, Any] = {}
        error: BaseException | None = None
        cleanup_error: BaseException | None = None
        try:
            result = self.run_core()
        except BaseException as exc:
            error = exc
            self.write_abort(exc)

        try:
            self.cleanup_container()
        except BaseException as exc:
            cleanup_error = exc
            if error is None:
                error = exc
                self.write_abort(exc)

        if self.context.role == ROLE_PREFILL:
            done_payload = {
                **_now_payload(),
                "success": error is None,
                "role": self.context.role,
                "commit_sha": self.context.sha,
                "image_id": self.context.image_id,
                "duration_seconds": round(time.monotonic() - started_at, 3),
                "error": None if error is None else str(error),
            }
            atomic_write_json(
                self.context.done_path, done_payload, self.context.shared_gid
            )

        role_result = {
            **_now_payload(),
            "success": error is None,
            "role": self.context.role,
            "runner_name": self.context.runner_name,
            "hostname": self.context.hostname,
            "duration_seconds": round(time.monotonic() - started_at, 3),
            "error": None if error is None else str(error),
            "cleanup_error": None if cleanup_error is None else str(cleanup_error),
            "result": result,
        }
        atomic_write_json(
            self.context.role_dir / "result.json",
            role_result,
            self.context.shared_gid,
        )
        self.heartbeat.stop()
        for stream in self.log_streams:
            stream.close()

        if error is not None:
            raise error
        return role_result


def cleanup_only(context: PDRunContext) -> None:
    orchestrator = PDOrchestrator(context)
    orchestrator.cleanup_container()


def _install_signal_handlers() -> None:
    def handle_signal(signum, _frame):
        raise PDInfrastructureError(f"received signal {signum}")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coordinate two-node HCU PD CI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "preflight", "cleanup"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--role", required=True, choices=sorted(VALID_ROLES)
        )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.umask(0o002)
    _install_signal_handlers()
    try:
        context = PDRunContext.from_environment(args.role)
        orchestrator = PDOrchestrator(context)
        if args.command == "cleanup":
            orchestrator.cleanup_container()
        elif args.command == "preflight":
            orchestrator.prepare_layout()
            print(json.dumps(orchestrator.preflight(), indent=2))
        else:
            print(json.dumps(orchestrator.execute(), ensure_ascii=False, indent=2))
    except BaseException as exc:
        print(f"[hcu-pd] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
