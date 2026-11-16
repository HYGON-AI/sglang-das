# SPDX-License-Identifier: Apache-2.0
"""Load-time model state that lives outside ``state_dict`` (weight cache).

The weight-cache client builds the model on the meta device and maps the
daemon's parameters and buffers over it. Loading a checkpoint can also leave
state in plain attributes: flags flipped by a one-shot weight fix-up, tensors
derived from the weights, repacked expert weights stashed on a layer. None of
that is in ``state_dict``, so a client that only maps parameters serves a model
whose weights are post-processed while its bookkeeping still says "fresh".

The daemon therefore walks every module after loading and exports that state
too; the client applies whatever differs from its own freshly-built model, lets
the model rebuild what is local to a process (``post_ipc_weight_load``), and
then checks that everything the daemon held has a counterpart.

Three opt-outs keep per-process state from crossing the process boundary:

* ``_weight_cache_skip = True`` on a class: its instances are never walked.
* ``_weight_cache_skip_attrs`` on a module class: those attributes are skipped.
* Objects from ``_RUNTIME_MODULE_PREFIXES`` (process groups, communicators).
"""

from __future__ import annotations

import io
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Attribute path: (module_name, ((kind, key), ...)) where kind is
# "a" (getattr), "i" (sequence index) or "k" (mapping key).
Step = Tuple[str, Any]
Path = Tuple[str, Tuple[Step, ...]]

_MODULE_INTERNAL_KEYS = frozenset(vars(nn.Module()).keys())
_SCALAR_TYPES = (bool, int, float, str, type(None))
_MAX_DEPTH = 5
_MAX_CONTAINER_LEN = 8192
# Communicators, process groups and their sockets: rebuilt by every process.
_RUNTIME_MODULE_PREFIXES = ("sglang.srt.distributed.",)
# Extra tensors at or below this size are copied into engine-owned memory so
# an engine can never observe (or leave behind) another engine's writes; larger
# ones (repacked expert weights) are mapped zero-copy like the parameters.
CLONE_MAX_BYTES = 16 << 20
# CPU tensors cannot use device IPC; small ones travel by value.
CPU_BY_VALUE_MAX_BYTES = 64 << 20


def format_path(path: Path) -> str:
    module_name, steps = path
    out = module_name or "<root>"
    for kind, key in steps:
        out += f".{key}" if kind == "a" else f"[{key!r}]"
    return out


def _walkable_object(value: Any) -> bool:
    cls = type(value)
    module = getattr(cls, "__module__", "") or ""
    if not module.startswith("sglang.") or not hasattr(value, "__dict__"):
        return False
    return not module.startswith(_RUNTIME_MODULE_PREFIXES)


def _skipped(value: Any) -> bool:
    if getattr(type(value), "_weight_cache_skip", False):
        return True
    try:
        from transformers import PretrainedConfig

        # Parsed from config.json by both processes; thousands of fields.
        return isinstance(value, PretrainedConfig)
    except ImportError:
        return False


def canonical_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Every module once, under its shortest name.

    A model may register an existing module a second time after loading (a
    layer keeping its successor as a submodule). ``named_modules()`` would then
    name most modules through that link, which a freshly built model lacks.
    The shortest name is the construction-time one on both sides.
    """
    best: Dict[int, Tuple[int, int, str, nn.Module]] = {}
    for order, (name, module) in enumerate(
        model.named_modules(remove_duplicate=False)
    ):
        depth = name.count(".") if name else -1
        current = best.get(id(module))
        if current is None or depth < current[0]:
            best[id(module)] = (depth, order, name, module)
    return [(name, m) for _, _, name, m in sorted(best.values(), key=lambda b: b[1])]


def collect_extra_state(model: nn.Module) -> Dict[str, Any]:
    """Walk plain (non parameter/buffer/submodule) attributes of every module.

    Returns ``tensors`` and ``scalars`` (path -> value), ``containers``
    (path -> ("list" | "tuple" | "dict", size) in parents-first order) and
    ``opaque`` (path -> type name of a value this walk cannot carry).
    """
    tensors: Dict[Path, torch.Tensor] = {}
    scalars: Dict[Path, Any] = {}
    containers: Dict[Path, Tuple[str, int]] = {}
    opaque: Dict[Path, str] = {}
    modules = canonical_modules(model)
    registered = {id(m) for _, m in modules}

    def visit(value: Any, module_name: str, steps: Tuple[Step, ...], seen: frozenset):
        path = (module_name, steps)
        if isinstance(value, torch.Tensor):
            tensors[path] = value
            return
        if type(value) in _SCALAR_TYPES:
            # Exact builtin types only: an enum or numpy scalar would pickle as
            # its class, which the weight-cache wire format refuses to load.
            scalars[path] = value
            return
        if isinstance(value, nn.Module):
            if id(value) not in registered:
                opaque[path] = f"unregistered module {type(value).__name__}"
            return
        if _skipped(value) or id(value) in seen or len(steps) >= _MAX_DEPTH:
            return
        seen = seen | {id(value)}
        if isinstance(value, tuple) and hasattr(type(value), "_fields"):
            # NamedTuple: immutable and typed, so the engine cannot be handed
            # one piecewise. Its fields are walked as attributes; they end up
            # checked against whatever the model's post-load hook rebuilds.
            for key in type(value)._fields:
                visit(getattr(value, key), module_name, steps + (("a", key),), seen)
            return
        if type(value) in (list, tuple):
            if len(value) > _MAX_CONTAINER_LEN:
                opaque[path] = f"{type(value).__name__}[{len(value)}]"
                return
            kind = "tuple" if isinstance(value, tuple) else "list"
            containers[path] = (kind, len(value))
            for i, item in enumerate(value):
                visit(item, module_name, steps + (("i", i),), seen)
            return
        if type(value) is dict:
            if len(value) > _MAX_CONTAINER_LEN:
                opaque[path] = f"dict[{len(value)}]"
                return
            containers[path] = ("dict", len(value))
            for key, item in value.items():
                if type(key) in (str, int):
                    visit(item, module_name, steps + (("k", key),), seen)
            return
        if _walkable_object(value):
            for key, item in vars(value).items():
                visit(item, module_name, steps + (("a", key),), seen)
            return
        if callable(value) or isinstance(value, (type, torch.dtype, torch.device)):
            return
        opaque[path] = type(value).__name__

    for module_name, module in modules:
        skip_attrs = getattr(type(module), "_weight_cache_skip_attrs", ())
        for key, value in vars(module).items():
            if key in _MODULE_INTERNAL_KEYS or key in skip_attrs:
                continue
            visit(value, module_name, (("a", key),), frozenset({id(module)}))
    return {
        "tensors": tensors,
        "scalars": scalars,
        "containers": containers,
        "opaque": opaque,
    }


def export_extra_state(
    model: nn.Module, serialize: Callable[[torch.Tensor], Any]
) -> Dict[str, Any]:
    """Daemon side: build the picklable extra-state payload."""
    state = collect_extra_state(model)
    entries: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for path, tensor in state["tensors"].items():
        tensor = tensor.detach()
        entry: Dict[str, Any] = {
            "path": path,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "device": tensor.device.type,
            "nbytes": tensor.numel() * tensor.element_size(),
        }
        try:
            if tensor.device.type == "cpu":
                if entry["nbytes"] > CPU_BY_VALUE_MAX_BYTES:
                    raise RuntimeError("CPU tensor too large to send by value")
                buf = io.BytesIO()
                torch.save(tensor.contiguous(), buf)
                entry["value"] = buf.getvalue()
            elif tensor.device.type == "meta":
                raise RuntimeError("tensor is on the meta device")
            else:
                entry["handle"] = serialize(tensor)
        except Exception as e:  # noqa: BLE001 - reported to the engine
            skipped.append(f"{format_path(path)}: {e}")
            entry["error"] = str(e)
        entries.append(entry)
    if skipped:
        logger.warning(
            "[weight_cache] %d extra-state tensor(s) could not be exported:\n  %s",
            len(skipped),
            "\n  ".join(skipped[:20]),
        )
    return {
        "tensors": entries,
        "scalars": list(state["scalars"].items()),
        "containers": list(state["containers"].items()),
        "opaque": list(state["opaque"].items()),
    }


def export_host_state(model: nn.Module) -> Tuple[Dict[str, Any], set]:
    """Daemon side: state modules keep in host memory they can share themselves.

    A module opts in with ``export_ipc_host_state()`` / ``attach_ipc_host_state``.
    Returns the per-module descriptions and the names (every alias) of the
    parameters and buffers those modules cover, which must not go through
    tensor export.
    """
    host_state: Dict[str, Any] = {}
    covered_ids: set = set()
    for module_name, module in canonical_modules(model):
        export = getattr(module, "export_ipc_host_state", None)
        if export is None:
            continue
        desc = export()
        if desc is None:
            continue
        host_state[module_name] = desc
        covered_ids.update(id(p) for p in module.parameters(recurse=False))
        covered_ids.update(id(b) for b in module.buffers(recurse=False))
    covered = {
        name
        for name, p in model.named_parameters(remove_duplicate=False)
        if id(p) in covered_ids
    } | {
        name
        for name, b in model.named_buffers(remove_duplicate=False)
        if id(b) in covered_ids
    }
    return host_state, covered


def attach_host_state(model: nn.Module, host_state: Optional[Dict[str, Any]]) -> int:
    """Engine side: every opted-in module attaches (or verifies it needs none)."""
    host_state = dict(host_state or {})
    attached = 0
    for module_name, module in canonical_modules(model):
        attach = getattr(module, "attach_ipc_host_state", None)
        if attach is None:
            continue
        desc = host_state.pop(module_name, None)
        attach(desc)
        attached += desc is not None
    if host_state:
        raise RuntimeError(
            f"[weight_cache] the daemon exported host state for modules this "
            f"engine does not have: {sorted(host_state)}"
        )
    return attached


_MISSING = object()


def _norm(path: Any) -> Path:
    return (path[0], tuple(tuple(s) for s in path[1]))


def _step(obj: Any, step: Step) -> Any:
    kind, key = step
    try:
        return getattr(obj, key) if kind == "a" else obj[key]
    except (AttributeError, KeyError, IndexError, TypeError):
        return _MISSING


def _lookup(model: nn.Module, path: Path) -> Any:
    module_name, steps = path
    try:
        obj: Any = model.get_submodule(module_name) if module_name else model
    except AttributeError:
        return _MISSING
    for step in steps:
        obj = _step(obj, step)
        if obj is _MISSING:
            return _MISSING
    return obj


def _resolve_parent(model: nn.Module, path: Path) -> Any:
    return _lookup(model, (path[0], path[1][:-1]))


def _set(parent: Any, step: Step, value: Any) -> None:
    kind, key = step
    if kind == "a":
        if isinstance(parent, nn.Module):
            # Bypass nn.Module.__setattr__: the daemon held this as a plain
            # attribute, so it must not become a registered parameter/buffer.
            object.__setattr__(parent, key, value)
        else:
            setattr(parent, key, value)
    else:
        parent[key] = value


def _same_tensor_layout(a: Any, shape: List[int], dtype: str) -> bool:
    return (
        isinstance(a, torch.Tensor)
        and a.device.type != "meta"
        and list(a.shape) == list(shape)
        and str(a.dtype).replace("torch.", "") == dtype
    )


class ExtraStateResult:
    def __init__(self):
        self.keep_alive: List[torch.Tensor] = []
        # Items whose parent object does not exist yet on the engine: the
        # model's post_ipc_weight_load is expected to rebuild them.
        self.deferred_tensors: List[Dict[str, Any]] = []
        self.deferred_scalars: List[Tuple[Path, Any]] = []


def apply_extra_state(
    model: nn.Module,
    payload: Optional[Dict[str, Any]],
    deserialize: Callable[[Any], torch.Tensor],
) -> ExtraStateResult:
    """Engine side: apply the daemon's extra state onto the mapped model."""
    result = ExtraStateResult()
    if not payload:
        return result
    stats = {"shared": 0, "cloned": 0, "kept_local": 0, "scalars": 0}
    changed_scalars: List[str] = []
    failures: List[str] = []

    # 1. Containers, parents first. Tuples become lists while they are filled.
    tuples: List[Path] = []
    for raw_path, (kind, size) in payload.get("containers", []):
        path = _norm(raw_path)
        parent = _resolve_parent(model, path)
        if parent is _MISSING or parent is None:
            continue  # Under an object rebuilt later; its leaves are deferred.
        current = _step(parent, path[1][-1])
        if kind == "dict":
            if not isinstance(current, dict):
                _set(parent, path[1][-1], {})
            continue
        if isinstance(current, list) and len(current) == size:
            # Keep the engine's own list object (it may be aliased).
            if kind == "tuple":
                tuples.append(path)
            continue
        if isinstance(current, tuple) and len(current) == size:
            filled = list(current)
        else:
            filled = [None] * size
        try:
            _set(parent, path[1][-1], filled)
        except Exception:  # noqa: BLE001
            # Immutable parent (a NamedTuple): its leaves are deferred and
            # checked after the model's post-load hook rebuilt the object.
            continue
        if kind == "tuple":
            tuples.append(path)

    # 2. Scalars: flags that guard one-shot weight rewrites must be in place
    #    before the model's own post-load hook runs.
    for raw_path, value in payload.get("scalars", []):
        path = _norm(raw_path)
        parent = _resolve_parent(model, path)
        if parent is _MISSING or parent is None:
            result.deferred_scalars.append((path, value))
            continue
        current = _step(parent, path[1][-1])
        if current is not _MISSING and type(current) is type(value) and current == value:
            continue
        try:
            _set(parent, path[1][-1], value)
        except Exception:  # noqa: BLE001 - immutable parent: checked later
            result.deferred_scalars.append((path, value))
            continue
        stats["scalars"] += 1
        shown = "<missing>" if current is _MISSING else repr(current)
        changed_scalars.append(f"{format_path(path)}: {shown} -> {value!r}")

    # 3. Tensors.
    for entry in payload.get("tensors", []):
        path = _norm(entry["path"])
        entry = {**entry, "path": path}
        name = format_path(path)
        parent = _resolve_parent(model, path)
        if parent is _MISSING or parent is None:
            result.deferred_tensors.append(entry)
            continue
        current = _step(parent, path[1][-1])
        if "error" in entry:
            if isinstance(current, torch.Tensor) and current.device.type != "meta":
                stats["kept_local"] += 1  # The engine built its own at init.
            else:
                result.deferred_tensors.append(entry)
            continue
        try:
            if "value" in entry:
                tensor = torch.load(io.BytesIO(entry["value"]), weights_only=True)
                shared = False
            else:
                tensor = deserialize(entry["handle"])
                shared = True
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: import failed ({e})")
            continue
        if (
            isinstance(current, torch.Tensor)
            and current.device.type != "meta"
            and current.shape == tensor.shape
            and current.dtype == tensor.dtype
            and current.device == tensor.device
            and torch.equal(current, tensor)
        ):
            stats["kept_local"] += 1  # Built identically at init time.
            continue
        if shared and entry["nbytes"] <= CLONE_MAX_BYTES:
            tensor = tensor.clone()
            shared = False
        try:
            _set(parent, path[1][-1], tensor)
        except Exception:  # noqa: BLE001 - immutable parent: checked later
            result.deferred_tensors.append(entry)
            continue
        if shared:
            result.keep_alive.append(tensor)
            stats["shared"] += 1
        else:
            stats["cloned"] += 1

    # 4. Restore tuples, children first.
    for path in reversed(tuples):
        parent = _resolve_parent(model, path)
        current = _step(parent, path[1][-1])
        if isinstance(current, list):
            _set(parent, path[1][-1], tuple(current))

    logger.info(
        "[weight_cache] extra state applied: %d tensors zero-copy, %d copied, "
        "%d already identical locally, %d scalars updated; %d tensors and %d "
        "scalars left to the model's post-load hook",
        stats["shared"],
        stats["cloned"],
        stats["kept_local"],
        stats["scalars"],
        len(result.deferred_tensors),
        len(result.deferred_scalars),
    )
    if changed_scalars:
        logger.info(
            "[weight_cache] scalars updated from daemon (%d):\n  %s",
            len(changed_scalars),
            "\n  ".join(changed_scalars[:60])
            + ("\n  ..." if len(changed_scalars) > 60 else ""),
        )
    if failures:
        raise RuntimeError(
            f"[weight_cache] {len(failures)} piece(s) of load-time state held by "
            f"the daemon could not be applied on the engine:\n  "
            + "\n  ".join(failures[:40])
        )
    return result


def verify_deferred_state(model: nn.Module, result: ExtraStateResult) -> None:
    """After the model's post-load hook: everything deferred must now exist.

    Refuses to serve otherwise: a tensor or flag the daemon derived while
    loading and this engine never rebuilt means incomplete load-time state.
    """
    missing: List[str] = []
    for entry in result.deferred_tensors:
        current = _lookup(model, entry["path"])
        if not _same_tensor_layout(current, entry["shape"], entry["dtype"]):
            missing.append(f"{format_path(entry['path'])} (tensor)")
    for path, value in result.deferred_scalars:
        current = _lookup(model, path)
        if current is _MISSING or type(current) is not type(value) or current != value:
            shown = "<missing>" if current is _MISSING else repr(current)
            missing.append(f"{format_path(path)}: engine {shown}, daemon {value!r}")
    if missing:
        raise RuntimeError(
            f"[weight_cache] {len(missing)} piece(s) of load-time state held by "
            f"the daemon were not reproduced on the engine (apply or "
            f"post_ipc_weight_load must rebuild them); refusing to serve:\n  "
            + "\n  ".join(missing[:40])
        )
    if result.deferred_tensors or result.deferred_scalars:
        logger.info(
            "[weight_cache] post-load hook reproduced %d tensors and %d scalars",
            len(result.deferred_tensors),
            len(result.deferred_scalars),
        )


def _checksum(t: torch.Tensor) -> str:
    """Content fingerprint for debugging: byte sum plus head/tail byte sums."""
    if t.device.type == "meta":
        return "meta"
    try:
        flat = t.detach().contiguous().view(-1)
        if flat.numel() == 0:
            return "empty"
        raw = flat.view(torch.uint8) if not flat.dtype == torch.bool else flat.to(torch.uint8)
        chunk = 64 << 20
        if t.device.type == "cpu" and raw.numel() > (256 << 20):
            # Host tables are tens of GiB: sample both ends only.
            parts = [raw[:chunk], raw[-chunk:]]
        else:
            parts = [raw[i : i + chunk] for i in range(0, raw.numel(), chunk)]
        total = sum(int(p.sum(dtype=torch.int64)) for p in parts)
        head = int(raw[:4096].sum(dtype=torch.int64))
        tail = int(raw[-4096:].sum(dtype=torch.int64))
        return f"{total}/{head}/{tail}"
    except Exception as e:  # noqa: BLE001
        return f"err:{type(e).__name__}"


def describe_state(model: nn.Module, checksums: bool = False) -> Dict[str, str]:
    """Debug fingerprint of the model state; contents only with checksums."""
    state = collect_extra_state(model)
    out: Dict[str, str] = {}

    def tensor_desc(t: torch.Tensor) -> str:
        desc = (
            f"tensor {tuple(t.shape)} {str(t.dtype).replace('torch.', '')} "
            f"{t.device.type}"
        )
        return desc + (f" sum={_checksum(t)}" if checksums else "")

    for path, t in state["tensors"].items():
        out[format_path(path)] = tensor_desc(t)
    for path, v in state["scalars"].items():
        out[format_path(path)] = f"scalar {v!r}"
    for path, name in state["opaque"].items():
        out[format_path(path)] = f"opaque {name}"
    if checksums:
        for name, p in model.named_parameters():
            out[f"param:{name}"] = tensor_desc(p.data)
        for name, b in model.named_buffers():
            out[f"buffer:{name}"] = tensor_desc(b)
    return out
