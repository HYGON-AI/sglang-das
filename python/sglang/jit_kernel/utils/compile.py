"""JIT compilation: load_jit, the build cache, and C++ template arguments."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import os
import pathlib
import re
import shutil
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple, TypeAlias, Union

import torch

from sglang.jit_kernel.utils.arch import get_jit_cuda_arch
from sglang.jit_kernel.utils.common import cache_once, is_hip_runtime
from sglang.jit_kernel.utils.deps import REGISTERED_DEPENDENCIES

if TYPE_CHECKING:
    from tvm_ffi import Module

logger = logging.getLogger(__name__)


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, (+{kernel_name}));"


_QUOTED_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
_ANGLE_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*<(sgl_kernel/[^>]+)>", re.MULTILINE)


def _local_jit_source_hash(source_files: List[str]) -> str:
    """Hash JIT source contents so TVM-FFI cache keys track included headers."""
    digest = hashlib.sha256()
    seen: set[pathlib.Path] = set()
    stack = [pathlib.Path(path).resolve() for path in source_files]
    include_dir = KERNEL_PATH / "include"

    while stack:
        path = stack.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)

        data = path.read_bytes()
        # Relative to kernel root, not absolute: the key must track source
        # content, not install location (differs across runners / job dirs).
        try:
            ident = str(path.relative_to(KERNEL_PATH))
        except ValueError:
            ident = path.name
        digest.update(ident.encode())
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")

        text = data.decode("utf-8", errors="ignore")
        for include in _QUOTED_INCLUDE_RE.findall(text):
            include_path = (path.parent / include).resolve()
            if include_path.is_file():
                stack.append(include_path)
        for include in _ANGLE_INCLUDE_RE.findall(text):
            include_path = (include_dir / include).resolve()
            if include_path.is_file():
                stack.append(include_path)

    return digest.hexdigest()[:16]


@cache_once
def _resolve_kernel_path() -> pathlib.Path:
    # Resolve via the package spec so the lookup is location-independent.
    spec = importlib.util.find_spec("sglang.jit_kernel")
    assert spec is not None and spec.origin is not None
    cur_dir = pathlib.Path(spec.origin).parent.resolve()

    # first, try this directory structure
    def _environment_install():
        candidate = cur_dir.resolve()
        if (candidate / "include").exists() and (candidate / "csrc").exists():
            return candidate
        return None

    def _package_install():
        # TODO: support find path by package
        return None

    path = _environment_install() or _package_install()
    if path is None:
        raise RuntimeError("Cannot find sglang.jit_kernel path")
    return path


KERNEL_PATH = _resolve_kernel_path()
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
DEFAULT_CFLAGS = ["-std=c++20", "-O3", "-Wno-return-type"]
DEFAULT_CUDA_CFLAGS = [
    "-std=c++20",
    "-O3",
    "--expt-relaxed-constexpr",
    "-Wno-return-type",
]
DEFAULT_HIP_CFLAGS = ["-std=c++20", "-O3", "-DUSE_ROCM", "-Wno-return-type"]
DEFAULT_LDFLAGS = []
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, str, bool, torch.dtype]


def _is_rocm_build() -> bool:
    return torch.version.hip is not None


def _default_device_cflags() -> List[str]:
    if not _is_rocm_build():
        return DEFAULT_CUDA_CFLAGS

    flags = list(DEFAULT_HIP_CFLAGS)
    amdgpu_targets = os.environ.get("AMDGPU_TARGET") or os.environ.get(
        "PYTORCH_ROCM_ARCH"
    )
    if not amdgpu_targets and torch.cuda.is_available():
        try:
            amdgpu_targets = torch.cuda.get_device_properties(0).gcnArchName.split(
                ":"
            )[0]
        except Exception:
            amdgpu_targets = None

    normalized_targets = []
    if amdgpu_targets:
        for target in amdgpu_targets.replace(";", ",").split(","):
            target = target.strip().split(":")[0]
            if target:
                normalized_targets.append(target)
                flags.append(f"--amdgpu-target={target}")
    if normalized_targets:
        use_fnuz = any(
            target.startswith(
                ("gfx90a", "gfx936", "gfx938", "gfx940", "gfx941", "gfx942")
            )
            for target in normalized_targets
        )
        flags.append("-DHIP_FP8_TYPE_FNUZ" if use_fnuz else "-DHIP_FP8_TYPE_E4M3")
    return flags


def _find_hipcc() -> str:
    candidates = [
        os.environ.get("HIPCC"),
        "/opt/dtk/bin/hipcc",
        "/opt/dtk/hip/bin/hipcc",
        shutil.which("hipcc"),
    ]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).exists():
            return str(candidate)
    raise RuntimeError(
        "Cannot find hipcc for ROCm JIT compilation. "
        "Please set HIPCC or install hipcc under /opt/dtk/bin."
    )


def _patch_tvm_ffi_load_inline_for_hip() -> None:
    """Teach CUDA-only tvm_ffi.cpp.load_inline builds to use hipcc on ROCm."""
    try:
        load_inline_mod = importlib.import_module("tvm_ffi.cpp.load_inline")
    except ModuleNotFoundError as exc:
        if exc.name != "tvm_ffi.cpp.load_inline":
            raise
        # Newer tvm_ffi merged load_inline into tvm_ffi.cpp.extension and supports
        # the HIP backend natively (auto-detected), so this CUDA-only shim is neither
        # importable nor needed. Skip patching and let tvm_ffi drive hipcc itself.
        return
    if getattr(load_inline_mod, "_sglang_hipcc_patched", False):
        return

    def _generate_hip_ninja_build(
        name: str,
        build_dir: str,
        with_cuda: bool,
        extra_cflags,
        extra_cuda_cflags,
        extra_ldflags,
        extra_include_paths,
    ) -> str:
        default_include_paths = [
            load_inline_mod.find_include_path(),
            load_inline_mod.find_dlpack_include_path(),
        ]
        tvm_ffi_lib = load_inline_mod.find_libtvm_ffi()
        tvm_ffi_lib_path = str(pathlib.Path(tvm_ffi_lib).parent)
        tvm_ffi_lib_name = pathlib.Path(tvm_ffi_lib).stem

        hipcc = _find_hipcc()
        default_cflags = ["-std=c++17", "-fPIC", "-O2"]
        default_cuda_cflags = ["-x", "hip", "-fPIC", "-std=c++17", "-O2"]
        default_ldflags = [
            "-shared",
            f"-L{tvm_ffi_lib_path}",
            f"-l{tvm_ffi_lib_name.removeprefix('lib')}",
        ]

        cflags = default_cflags + [flag.strip() for flag in extra_cflags]
        cuda_cflags = default_cuda_cflags + [
            flag.strip() for flag in extra_cuda_cflags
        ]
        ldflags = default_ldflags + [flag.strip() for flag in extra_ldflags]
        include_paths = default_include_paths + [
            str(pathlib.Path(path).resolve()) for path in extra_include_paths
        ]

        for path in include_paths:
            cflags.append("-I{}".format(path.replace(":", "$:")))
            cuda_cflags.append("-I{}".format(path.replace(":", "$:")))

        ninja = [
            "ninja_required_version = 1.3",
            "cxx = {}".format(os.environ.get("CXX", hipcc)),
            "hipcc = {}".format(hipcc),
            "cflags = {}".format(" ".join(cflags)),
        ]
        if with_cuda:
            ninja.append("cuda_cflags = {}".format(" ".join(cuda_cflags)))
        ninja.append("ldflags = {}".format(" ".join(ldflags)))
        ninja.extend(
            [
                "",
                "rule compile",
                "  depfile = $out.d",
                "  deps = gcc",
                "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out",
                "",
            ]
        )
        if with_cuda:
            ninja.extend(
                [
                    "rule compile_cuda",
                    "  depfile = $out.d",
                    "  deps = gcc",
                    "  command = $hipcc -MMD -MF $out.d $cuda_cflags -c $in -o $out",
                    "",
                ]
            )
        ninja.extend(
            [
                "rule link",
                "  command = $cxx $in $ldflags -o $out",
                "",
                "build main.o: compile {}".format(
                    str((pathlib.Path(build_dir) / "main.cpp").resolve()).replace(
                        ":", "$:"
                    )
                ),
            ]
        )
        if with_cuda:
            ninja.append(
                "build cuda.o: compile_cuda {}".format(
                    str((pathlib.Path(build_dir) / "cuda.cu").resolve()).replace(
                        ":", "$:"
                    )
                )
            )
        ninja.append(
            "build {}.so: link main.o{}".format(
                name, " cuda.o" if with_cuda else ""
            )
        )
        ninja.extend(["", f"default {name}.so", ""])
        return "\n".join(ninja)

    load_inline_mod._generate_ninja_build = _generate_hip_ninja_build
    load_inline_mod._sglang_hipcc_patched = True


class CPPArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


CPP_DTYPE_MAP = {
    torch.float64: "double",
    torch.float32: "fp32_t",
    torch.float16: "fp16_t",
    torch.bfloat16: "bf16_t",
    # The fnuz variants are the ROCm-side torch dtypes; fp8_*_t resolves to
    # the matching HIP type there (see HIP_FP8_TYPE_* in utils.cuh).
    torch.float8_e4m3fn: "fp8_e4m3_t",
    torch.float8_e4m3fnuz: "fp8_e4m3_t",
    torch.float8_e5m2: "fp8_e5m2_t",
    torch.float8_e5m2fnuz: "fp8_e5m2_t",
    torch.int8: "int8_t",
    torch.int16: "int16_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.uint8: "uint8_t",
    torch.uint16: "uint16_t",
    torch.uint32: "uint32_t",
    torch.uint64: "uint64_t",
    torch.bool: "bool",
}


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CPPArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, str, float)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return CPP_DTYPE_MAP[arg]
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CPPArgList(_convert(arg) for arg in args)


@cache_once
def _tvm_ffi_version() -> str:
    try:
        import tvm_ffi

        version = getattr(tvm_ffi, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    try:
        from importlib.metadata import version as dist_version

        return dist_version("apache-tvm-ffi")
    except Exception:
        return "unknown"


def _jit_build_dir_name(module_name: str) -> str:
    # Key on arch + tvm-ffi ABI too (module_name only hashes sources), so a
    # shared cache volume never reuses a cross-arch/ABI .so.
    arch = get_jit_cuda_arch().target_name
    return f"{module_name}__arch_{arch}__tvmffi_{_tvm_ffi_version()}"


def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    extra_dependencies: List[str] | None = None,
    build_directory: str | None = None,
    header_only: bool = True,
) -> Module:
    """
    Loading a JIT module from C++/CUDA source files.
    We define a wrapper as a tuple of (export_name, kernel_name),
    where `export_name` is the name used to called from Python,
    and `kernel_name` is the name of the kernel class in C++/CUDA source.

    :param args: Unique marker of the JIT module. Must be distinct for different kernels.
    :type args: str
    :param cpp_files: A list of C++ source files.
    :type cpp_files: List[str] | None
    :param cuda_files: A list of CUDA source files.
    :type cuda_files: List[str] | None
    :param cpp_wrappers: A list of C++ wrappers, defining the export name and kernel name.
    :type cpp_wrappers: List[Tuple[str, str]] | None
    :param cuda_wrappers: A list of CUDA wrappers, defining the export name and kernel name.
    :type cuda_wrappers: List[Tuple[str, str]] | None
    :param extra_cflags: Extra C++ compiler flags.
    :type extra_cflags: List[str] | None
    :param extra_cuda_cflags: Extra CUDA compiler flags.
    :type extra_cuda_cflags: List[str] | None
    :param extra_ldflags: Extra linker flags.
    :type extra_ldflags: List[str] | None
    :param extra_include_paths: Extra include paths.
    :type extra_include_paths: List[str] | None
    :param extra_dependencies: Extra dependencies for the JIT module, e.g., cutlass.
    :type extra_dependencies: List[str] | None
    :param build_directory: The build directory for JIT compilation.
    :type build_directory: str | None
    :param header_only: Whether the module is header-only.
                        If true, apply the wrappers to export given class/functions.
                        Otherwise, we must export from C++/CUDA side.
    :return: A just-in-time(JIT) compiled module.
    :rtype: Module
    """

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((KERNEL_PATH / "csrc" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "csrc" / f).resolve()) for f in cuda_files]

    for dep in set(extra_dependencies or []):
        if dep not in REGISTERED_DEPENDENCIES:
            raise ValueError(f"Dependency {dep} is not registered.")
        extra_include_paths += REGISTERED_DEPENDENCIES[dep]()

    backend = "hip" if _is_rocm_build() else "cuda"
    module_name = (
        "sgl_kernel_jit_" + backend + "_" + "_".join(str(arg) for arg in args)
    )
    if cpp_files or cuda_files:
        module_name += "_" + _local_jit_source_hash(cpp_files + cuda_files)
    default_device_cflags = _default_device_cflags()

    # A built .so under a deterministic dir is content-addressed: load it
    # directly to skip ninja, whose mtime check rebuilds every CI run (pip
    # install bumps dep header mtimes).
    if build_directory is None:
        cache_dir = os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")
        build_directory = str(
            pathlib.Path(cache_dir).expanduser() / _jit_build_dir_name(module_name)
        )
    prebuilt = pathlib.Path(build_directory) / f"{module_name}.so"
    if prebuilt.is_file():
        from tvm_ffi import load_module

        try:
            module = load_module(str(prebuilt))
            logger.debug("Reused cached JIT module %s", module_name)
            return module
        except Exception:
            logger.warning(
                "Cached JIT module %s failed to load; rebuilding.", module_name
            )

    if header_only:
        from tvm_ffi.cpp import load_inline

        if _is_rocm_build():
            _patch_tvm_ffi_load_inline_for_hip()
        cpp_wrappers = cpp_wrappers or []
        cuda_wrappers = cuda_wrappers or []
        cpp_sources = [f'#include "{path}"' for path in cpp_files]
        cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

        # include cuda files
        cuda_sources = [f'#include "{path}"' for path in cuda_files]
        cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]
        with _jit_compile_context():
            return load_inline(
                module_name,
                cpp_sources=cpp_sources,
                cuda_sources=cuda_sources,
                extra_cflags=DEFAULT_CFLAGS + extra_cflags,
                extra_cuda_cflags=default_device_cflags + extra_cuda_cflags,
                extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
                extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
                build_directory=build_directory,
            )
    else:
        from tvm_ffi.cpp import load

        assert cpp_wrappers is None and cuda_wrappers is None
        with _jit_compile_context():
            return load(
                module_name,
                cpp_files=cpp_files,
                cuda_files=cuda_files,
                extra_cflags=DEFAULT_CFLAGS + extra_cflags,
                extra_cuda_cflags=default_device_cflags + extra_cuda_cflags,
                extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
                extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
                build_directory=build_directory,
            )


@contextmanager
def _jit_compile_context():
    if is_hip_runtime():
        yield  # TODO: support ROCm `TVM_FFI_ROCM_ARCH_LIST` if needed
        return
    env_key = "TVM_FFI_CUDA_ARCH_LIST"
    old_value = os.environ.get(env_key, None)
    os.environ[env_key] = get_jit_cuda_arch().target_name
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_value
