"""
Custom HIP kernels for ROCm builds of PyTorch on Windows and Linux.

They are compiled at runtime with HIPRTC for whatever GPU is present and
launched on PyTorch's current stream. HIPRTC ships with the ROCm runtime that
PyTorch loads, so no compiler or SDK has to be installed.

Adapted from florentia-mysteria/spandrel-rocm (MIT): the depthwise 3x3
convolution and ESRGAN dense-block kernels, the HIPRTC loader and its
code-object cache. Changes: Linux support; the kernels are launched through the
HIP runtime that PyTorch already loaded (found among the libraries loaded in
this process, never loaded a second time); 64-bit sizes and offsets on Windows
too, where `long` has 32 bits; every failure raises HipKernelError.
"""
from __future__ import annotations
from nnarchs.logger import nnlogger

import ctypes
import hashlib
import os
import sys
import threading
from collections.abc import Sequence
from pathlib import Path

import torch



class HipKernelError(RuntimeError):
    """A custom HIP kernel could not be loaded, compiled or launched."""


# What a fast path catches to fall back to the original code (after re-raising
# torch.cuda.OutOfMemoryError, which the original code would hit as well):
# HipKernelError, errors of PyTorch's own calls on the way, and a ctypes
# argument that does not fit a declared C signature.
FAILURES: tuple[type[Exception], ...] = (RuntimeError, OSError, ctypes.ArgumentError)


_DWCONV_SRC = r"""
typedef unsigned short bf16_t;
typedef long long i64;

__device__ __forceinline__ float load_val(const float* p, i64 i) { return p[i]; }
__device__ __forceinline__ float load_val(const _Float16* p, i64 i) { return (float)p[i]; }
__device__ __forceinline__ float load_val(const bf16_t* p, i64 i) {
    return __uint_as_float(((unsigned int)p[i]) << 16);
}
__device__ __forceinline__ void store_val(float* p, i64 i, float v) { p[i] = v; }
__device__ __forceinline__ void store_val(_Float16* p, i64 i, float v) { p[i] = (_Float16)v; }
__device__ __forceinline__ void store_val(bf16_t* p, i64 i, float v) {
    unsigned int u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);  // round to nearest even
    p[i] = v != v ? (bf16_t)0x7FC0 : (bf16_t)(u >> 16);  // NaN as PyTorch stores it
}

#define TW 32
#define TH 16

// Depthwise 3x3 convolution, stride 1, zero padding 1, NCHW layout.
// Each block loads a (TH+2) x (TW+2) tile of one channel into shared memory,
// so every input value is read from global memory about once. Accumulates
// in fp32 regardless of the storage type.
template <typename T>
__device__ void dwconv3x3(const T* __restrict__ x, const float* __restrict__ w,
                          const float* __restrict__ b, T* __restrict__ y,
                          int C, int H, int W) {
    __shared__ float tile[TH + 2][TW + 2];
    const int bc = blockIdx.z;  // batch * C + channel
    const int c = bc % C;
    const i64 plane = (i64)bc * H * W;
    const int x0 = blockIdx.x * TW, y0 = blockIdx.y * TH;
    const int tid = threadIdx.y * TW + threadIdx.x;

    for (int i = tid; i < (TH + 2) * (TW + 2); i += TW * TH) {
        int ty = i / (TW + 2), tx = i % (TW + 2);
        int gy = y0 + ty - 1, gx = x0 + tx - 1;
        tile[ty][tx] = (gy >= 0 && gy < H && gx >= 0 && gx < W)
            ? load_val(x, plane + (i64)gy * W + gx) : 0.0f;
    }
    __syncthreads();

    const int ox = x0 + threadIdx.x, oy = y0 + threadIdx.y;
    if (ox >= W || oy >= H) return;
    const float* k = w + c * 9;
    float acc = b ? b[c] : 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++)
            acc += tile[threadIdx.y + i][threadIdx.x + j] * k[i * 3 + j];
    store_val(y, plane + (i64)oy * W + ox, acc);
}

extern "C" __global__ void dwconv3x3_f32(const float* x, const float* w, const float* b, float* y, int C, int H, int W) { dwconv3x3<float>(x, w, b, y, C, H, W); }
extern "C" __global__ void dwconv3x3_f16(const _Float16* x, const float* w, const float* b, _Float16* y, int C, int H, int W) { dwconv3x3<_Float16>(x, w, b, y, C, H, W); }
extern "C" __global__ void dwconv3x3_bf16(const bf16_t* x, const float* w, const float* b, bf16_t* y, int C, int H, int W) { dwconv3x3<bf16_t>(x, w, b, y, C, H, W); }
"""

_RRDB_SRC = r"""
typedef unsigned short bf16_t;
typedef long long i64;

__device__ __forceinline__ float load_val(const float* p, i64 i) { return p[i]; }
__device__ __forceinline__ float load_val(const _Float16* p, i64 i) { return (float)p[i]; }
__device__ __forceinline__ float load_val(const bf16_t* p, i64 i) {
    return __uint_as_float(((unsigned int)p[i]) << 16);
}
// float -> bf16 like PyTorch: round to nearest even, every NaN to 0x7FC0
__device__ __forceinline__ unsigned short bf16_bits(float v) {
    if (v != v) return 0x7FC0;
    unsigned int u = __float_as_uint(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
__device__ __forceinline__ void store_val(float* p, i64 i, float v) { p[i] = v; }
__device__ __forceinline__ void store_val(_Float16* p, i64 i, float v) { p[i] = (_Float16)v; }
__device__ __forceinline__ void store_val(bf16_t* p, i64 i, float v) { p[i] = bf16_bits(v); }

// Rounds a float to T and back, as PyTorch does after every elementwise op.
__device__ __forceinline__ float round_to(const float*, float v) { return v; }
__device__ __forceinline__ float round_to(const _Float16*, float v) { return (float)(_Float16)v; }
__device__ __forceinline__ float round_to(const bf16_t*, float v) {
    return __uint_as_float(((unsigned int)bf16_bits(v)) << 16);
}

// ESRGAN's dense blocks without torch.cat. Every block works in one buffer
// (B, nf + 4 * gc, H, W) holding [x | x1 | x2 | x3 | x4]; each conv reads a
// channel prefix of it and these kernels write its output into the next slot.
// All tensors are NCHW with H*W contiguous per channel; `*_stride` is the
// per-image element stride (channels of the underlying buffer * H * W).
// blockIdx.y is the image.

// dst[off:off+C] = LeakyReLU(src), computed like PyTorch: x > 0 ? x : x * slope
template <typename T>
__device__ void act_into(const T* __restrict__ src, T* __restrict__ dst, i64 n,
                         i64 dst_stride, i64 dst_off, float slope) {
    const i64 b = blockIdx.y;
    const T* s = src + b * n;
    T* d = dst + b * dst_stride + dst_off;
    for (i64 i = (i64)blockIdx.x * 256 + threadIdx.x; i < n; i += (i64)gridDim.x * 256) {
        const float v = load_val(s, i);
        store_val(d, i, v > 0.0f ? v : v * slope);
    }
}

// dst = a * s1 + x1, then optionally * s2 + x2, rounding to T after every
// multiply and add like the separate PyTorch ops do.
template <typename T>
__device__ void res_add(const T* __restrict__ a, const T* __restrict__ x1, i64 x1_stride,
                        const T* __restrict__ x2, i64 x2_stride, T* __restrict__ dst,
                        i64 dst_stride, i64 n, float s1, float s2, int two) {
    const i64 b = blockIdx.y;
    const T* pa = a + b * n;
    const T* p1 = x1 + b * x1_stride;
    const T* p2 = two ? x2 + b * x2_stride : nullptr;
    T* d = dst + b * dst_stride;
    for (i64 i = (i64)blockIdx.x * 256 + threadIdx.x; i < n; i += (i64)gridDim.x * 256) {
        float t = round_to(d, load_val(pa, i) * s1);
        t = round_to(d, t + load_val(p1, i));
        if (two) {
            t = round_to(d, t * s2);
            t = round_to(d, t + load_val(p2, i));
        }
        store_val(d, i, t);
    }
}

#define RRDB_KERNELS(T, S)                                                                  \
    extern "C" __global__ void act_into_##S(const T* src, T* dst, i64 n, i64 dst_stride,    \
                                            i64 dst_off, float slope) {                     \
        act_into<T>(src, dst, n, dst_stride, dst_off, slope);                               \
    }                                                                                       \
    extern "C" __global__ void res_add_##S(const T* a, const T* x1, i64 x1_stride,          \
                                           const T* x2, i64 x2_stride, T* dst,              \
                                           i64 dst_stride, i64 n, float s1, float s2,       \
                                           int two) {                                       \
        res_add<T>(a, x1, x1_stride, x2, x2_stride, dst, dst_stride, n, s1, s2, two);       \
    }
RRDB_KERNELS(float, f32)
RRDB_KERNELS(_Float16, f16)
RRDB_KERNELS(bf16_t, bf16)
"""

_TW, _TH = 32, 16
_MAX_GRID = 65535  # largest grid y and z
_DTYPE_SUFFIX = {torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16"}
# name -> (source, kernel names, extra compiler options)
_PROGRAMS: dict[str, tuple[str, list[str], list[str]]] = {
    "dwconv": (_DWCONV_SRC, [f"dwconv3x3_{s}" for s in _DTYPE_SUFFIX.values()], []),
    # exact PyTorch rounding: no fused multiply-add
    "rrdb": (
        _RRDB_SRC,
        [f"{k}_{s}" for k in ("act_into", "res_add") for s in _DTYPE_SUFFIX.values()],
        ["-ffp-contract=off"],
    ),
}
PROGRAM_NAMES: tuple[str, ...] = tuple(_PROGRAMS)

_KernelArg = ctypes.c_void_p | ctypes.c_int64 | ctypes.c_int | ctypes.c_float

_Api = dict[str, tuple[type, list[type]]]

_HIP_API: _Api = {
    "hipGetErrorString": (ctypes.c_char_p, [ctypes.c_int]),
    "hipModuleLoadData": (
        ctypes.c_int,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
    ),
    "hipModuleGetFunction": (
        ctypes.c_int,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
    ),
    "hipModuleLaunchKernel": (
        ctypes.c_int,
        # f, grid x/y/z, block x/y/z, shared memory, stream, params, extra
        [ctypes.c_void_p]
        + [ctypes.c_uint] * 7
        + [ctypes.c_void_p]
        + [ctypes.POINTER(ctypes.c_void_p)] * 2,
    ),
}
_HIPRTC_API: _Api = {
    "hiprtcGetErrorString": (ctypes.c_char_p, [ctypes.c_int]),
    "hiprtcVersion": (
        ctypes.c_int,
        [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)],
    ),
    "hiprtcCreateProgram": (
        ctypes.c_int,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p]
        + [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ],
    ),
    "hiprtcCompileProgram": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    ),
    "hiprtcGetProgramLogSize": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)],
    ),
    "hiprtcGetProgramLog": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
    "hiprtcGetCodeSize": (
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)],
    ),
    "hiprtcGetCode": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
    "hiprtcDestroyProgram": (ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)]),
}


def _declare(lib: ctypes.CDLL, path: Path, api: _Api) -> None:
    """Sets the C signatures of the functions in api; HipKernelError if one is missing."""
    for name, (restype, argtypes) in api.items():
        try:
            func = getattr(lib, name)
        except AttributeError as e:
            raise HipKernelError(f"{path} has no {name}") from e
        func.restype = restype
        func.argtypes = argtypes


# --- finding the HIP runtime PyTorch uses ---------------------------------------


def library_kind(name: str) -> str | None:
    """'hip' for the HIP runtime, 'hiprtc' for HIPRTC (by file name), else None."""
    name = name.lower()
    if sys.platform == "win32":
        if not name.endswith(".dll"):
            return None
    elif name.startswith("lib") and ".so" in name:
        name = name[3:]
    else:
        return None
    if name.startswith("amdhip64"):
        return "hip"
    if name.startswith("hiprtc") and not name.startswith("hiprtc-builtins"):
        return "hiprtc"
    return None


if sys.platform == "win32":

    def _windows_modules() -> dict[Path, int | None]:
        """The DLLs loaded in this process and their module handles."""
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        current_process = kernel32.GetCurrentProcess
        current_process.restype = wintypes.HANDLE
        current_process.argtypes = []
        enum_modules = kernel32.K32EnumProcessModules
        enum_modules.restype = wintypes.BOOL
        enum_modules.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.HMODULE),
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        module_file = kernel32.GetModuleFileNameW
        module_file.restype = wintypes.DWORD
        module_file.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]

        process = current_process()
        count = 512
        while True:
            handles = (wintypes.HMODULE * count)()
            needed = wintypes.DWORD()
            if not enum_modules(process, handles, ctypes.sizeof(handles), needed):
                raise ctypes.WinError(ctypes.get_last_error())
            if needed.value <= ctypes.sizeof(handles):
                break
            count = needed.value // ctypes.sizeof(wintypes.HMODULE) + 64

        found: dict[Path, int | None] = {}
        buffer = ctypes.create_unicode_buffer(32768)
        for handle in handles[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            if handle and module_file(handle, buffer, len(buffer)):
                found[Path(buffer.value)] = handle
        return found


def _loaded_libraries() -> dict[Path, int | None]:
    """
    The shared libraries loaded in this process: on Windows with their module
    handles, on Linux from /proc/self/maps (handles None).
    """
    if sys.platform == "win32":
        return _windows_modules()
    found: dict[Path, int | None] = {}
    with open("/proc/self/maps", encoding="utf-8", errors="replace") as maps:
        for line in maps:
            fields = line.rstrip("\n").split(maxsplit=5)
            if (
                len(fields) == 6
                and fields[5].startswith("/")
                and not fields[5].endswith(" (deleted)")
            ):
                found[Path(fields[5])] = None
    return found


def _hiprtc_path(runtime: Path, loaded: Sequence[Path]) -> Path:
    """The HIPRTC library that belongs to the HIP runtime: the one next to it."""
    candidates = [p for p in loaded if p.parent == runtime.parent]
    if not candidates:
        try:
            candidates = [
                p for p in runtime.parent.iterdir() if library_kind(p.name) == "hiprtc"
            ]
        except OSError as e:
            raise HipKernelError(f"cannot list {runtime.parent}: {e}") from e
    if not candidates:
        raise HipKernelError(f"no HIPRTC library next to {runtime}")
    # e.g. libhiprtc.so before libhiprtc.so.7.14.60850 (links to the same file)
    return min(candidates, key=lambda p: (len(p.name), p.name))


def open_hiprtc(path: Path) -> ctypes.CDLL:
    """Loads a HIPRTC library. Compiling needs no GPU."""
    try:
        rtc = ctypes.CDLL(str(path))
    except OSError as e:
        raise HipKernelError(f"cannot load {path}: {e}") from e
    _declare(rtc, path, _HIPRTC_API)
    return rtc


def open_hip_libraries() -> tuple[ctypes.CDLL, ctypes.CDLL, Path]:
    """
    (HIP runtime, HIPRTC, runtime path). The runtime must be the copy PyTorch
    loaded: a second copy would be a separate runtime with its own contexts,
    which cannot use PyTorch's memory or streams.
    """
    try:
        loaded = _loaded_libraries()
    except OSError as e:
        raise HipKernelError(f"cannot list the loaded libraries: {e}") from e
    runtimes = sorted(p for p in loaded if library_kind(p.name) == "hip")
    if len(runtimes) != 1:
        found = ", ".join(map(str, runtimes)) or "none"
        raise HipKernelError(f"expected one loaded HIP runtime, found: {found}")
    runtime = runtimes[0]
    rtc_path = _hiprtc_path(
        runtime, sorted(p for p in loaded if library_kind(p.name) == "hiprtc")
    )
    try:
        if sys.platform == "win32":
            # the module handle of the loaded DLL: nothing is loaded
            hip = ctypes.CDLL(str(runtime), handle=loaded[runtime])
        else:
            # RTLD_NOLOAD: fails instead of loading the library a second time
            hip = ctypes.CDLL(str(runtime), mode=os.RTLD_NOLOAD)
    except OSError as e:
        raise HipKernelError(f"cannot open the loaded {runtime}: {e}") from e
    _declare(hip, runtime, _HIP_API)
    return hip, open_hiprtc(rtc_path), runtime


# --- compiling ------------------------------------------------------------------


def _rtc_check(rtc: ctypes.CDLL, err: int, what: str) -> None:
    if err != 0:
        text = rtc.hiprtcGetErrorString(err) or b"unknown error"
        raise HipKernelError(f"{what} failed: {text.decode(errors='replace')} ({err})")


def hiprtc_version(rtc: ctypes.CDLL) -> str:
    major, minor = ctypes.c_int(), ctypes.c_int()
    _rtc_check(rtc, rtc.hiprtcVersion(major, minor), "hiprtcVersion")
    return f"{major.value}.{minor.value}"


def _options(arch: str, program: str) -> list[bytes]:
    extra = _PROGRAMS[program][2]
    return [f"--offload-arch={arch}".encode(), b"-O3", *(o.encode() for o in extra)]


def compile_program(rtc: ctypes.CDLL, program: str, arch: str) -> bytes:
    """
    Compiles one of PROGRAM_NAMES for a GPU architecture (e.g. "gfx1201") with
    HIPRTC and returns the code object. Needs no GPU.
    """
    source = _PROGRAMS[program][0]
    opts = _options(arch, program)
    prog = ctypes.c_void_p()
    _rtc_check(
        rtc,
        rtc.hiprtcCreateProgram(
            prog, source.encode(), f"{program}.hip".encode(), 0, None, None
        ),
        "hiprtcCreateProgram",
    )
    try:
        err = rtc.hiprtcCompileProgram(
            prog, len(opts), (ctypes.c_char_p * len(opts))(*opts)
        )
        if err != 0:
            size = ctypes.c_size_t()
            log = ctypes.create_string_buffer(1)
            if rtc.hiprtcGetProgramLogSize(prog, size) == 0 and size.value:
                log = ctypes.create_string_buffer(size.value)
                if rtc.hiprtcGetProgramLog(prog, log) != 0:
                    log = ctypes.create_string_buffer(1)
            text = rtc.hiprtcGetErrorString(err) or b"unknown error"
            raise HipKernelError(
                f"compiling {program} for {arch} failed: "
                f"{text.decode(errors='replace')} ({err})\n"
                + log.value.decode(errors="replace")
            )
        size = ctypes.c_size_t()
        _rtc_check(rtc, rtc.hiprtcGetCodeSize(prog, size), "hiprtcGetCodeSize")
        code = ctypes.create_string_buffer(size.value)
        _rtc_check(rtc, rtc.hiprtcGetCode(prog, code), "hiprtcGetCode")
        return code.raw
    finally:
        destroy_err = rtc.hiprtcDestroyProgram(prog)
        if destroy_err != 0:
            nnlogger.debug("hiprtcDestroyProgram failed with %d", destroy_err)


def cache_dir() -> Path | None:
    """
    Where compiled kernels are kept between runs: SPANDREL_HIP_CACHE_DIR if set
    (an empty value disables the cache), otherwise spandrel/hip_cache in
    %LOCALAPPDATA% on Windows and in $XDG_CACHE_HOME (default ~/.cache) on Linux.
    """
    custom = os.environ.get("SPANDREL_HIP_CACHE_DIR")
    if custom is not None:
        return Path(custom) if custom.strip() else None
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        return Path(local) / "spandrel" / "hip_cache" if local else None
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    if os.path.isabs(xdg):
        base = Path(xdg)
    else:
        try:
            base = Path.home() / ".cache"
        except RuntimeError:
            return None
    return base / "spandrel" / "hip_cache"


class _Hip:
    def __init__(self) -> None:
        self.hip, self.rtc, runtime = open_hip_libraries()
        self.rtc_version = hiprtc_version(self.rtc)
        self.funcs: dict[tuple[int, str], ctypes.c_void_p] = {}
        self.compiled: set[tuple[int, str]] = set()
        self.lock = threading.Lock()
        nnlogger.debug("HIP runtime %s, HIPRTC %s", runtime, self.rtc_version)

    def _check(self, err: int, what: str) -> None:
        if err != 0:
            text = self.hip.hipGetErrorString(err) or b"unknown error"
            raise HipKernelError(
                f"{what} failed: {text.decode(errors='replace')} ({err})"
            )

    def _load(self, device_index: int, names: list[str], code: bytes) -> None:
        funcs: dict[tuple[int, str], ctypes.c_void_p] = {}
        with torch.cuda.device(device_index):
            module = ctypes.c_void_p()
            self._check(self.hip.hipModuleLoadData(module, code), "hipModuleLoadData")
            for name in names:
                fn = ctypes.c_void_p()
                self._check(
                    self.hip.hipModuleGetFunction(fn, module, name.encode()),
                    "hipModuleGetFunction",
                )
                funcs[(device_index, name)] = fn
        self.funcs.update(funcs)

    def _compile(self, device_index: int, program: str) -> None:
        """
        Loads a program's kernels for a device, from the on-disk cache if
        possible. Code objects are cached by everything that affects them:
        kernel source, compiler options, GPU architecture and HIPRTC version. A
        changed kernel or driver simply gets a new file; stale files of the same
        program and GPU are removed.
        """
        names = _PROGRAMS[program][1]
        props = torch.cuda.get_device_properties(device_index)
        arch = props.gcnArchName.split(":")[0]
        digest = hashlib.sha256(
            b"|".join(
                [
                    _PROGRAMS[program][0].encode(),
                    *_options(arch, program),
                    self.rtc_version.encode(),
                ]
            )
        ).hexdigest()[:24]
        prefix = f"{program}-{arch}-"
        directory = cache_dir()
        path = directory / f"{prefix}{digest}.co" if directory else None

        if path is not None and path.is_file():
            try:
                self._load(device_index, names, path.read_bytes())
                nnlogger.debug("Loaded cached HIP %s kernels for %s", program, arch)
                return
            except (HipKernelError, OSError) as e:
                nnlogger.info("Ignoring unusable cached HIP kernels %s: %s", path, e)
                try:
                    path.unlink()
                except OSError as unlink_error:
                    nnlogger.debug("Could not remove %s: %s", path, unlink_error)

        code = compile_program(self.rtc, program, arch)
        self._load(device_index, names, code)
        nnlogger.info("Compiled HIP %s kernels for %s", program, arch)

        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                for old in path.parent.glob(f"{prefix}*.co"):
                    if old != path:
                        old.unlink(missing_ok=True)
                # a temp file first, so a concurrent reader never sees half a file
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_bytes(code)
                os.replace(tmp, path)
            except OSError as e:
                nnlogger.debug("Could not cache HIP %s kernels: %s", program, e)

    def get(self, device_index: int, program: str, name: str) -> ctypes.c_void_p:
        fn = self.funcs.get((device_index, name))
        if fn is None:
            with self.lock:
                if (device_index, program) not in self.compiled:
                    self._compile(device_index, program)
                    self.compiled.add((device_index, program))
                fn = self.funcs[(device_index, name)]
        return fn

    def launch(
        self,
        fn: ctypes.c_void_p,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[_KernelArg],
        stream: int,
    ) -> None:
        params = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        self._check(
            self.hip.hipModuleLaunchKernel(fn, *grid, *block, 0, stream, params, None),
            "hipModuleLaunchKernel",
        )


_hip: _Hip | None = None
_hip_lock = threading.Lock()


def _get_hip() -> _Hip:
    global _hip
    if _hip is None:
        with _hip_lock:
            if _hip is None:
                _hip = _Hip()
    return _hip


# the guards use the device-type autocast check of PyTorch 2.4
_TORCH_SUPPORTED = not torch.__version__ < "2.4"


def hip_kernels_available() -> bool:
    """
    Whether the custom kernels may run: a ROCm build of PyTorch (2.4 or newer)
    on Windows or Linux with a GPU, and SPANDREL_HIP_KERNELS is not 0. Never on
    CPU-only or CUDA builds of PyTorch.
    """
    if not torch.version.hip or sys.platform not in ("win32", "linux"):
        return False
    if not _TORCH_SUPPORTED:
        return False
    if os.environ.get("SPANDREL_HIP_KERNELS", "").strip() == "0":
        return False
    return torch.cuda.is_available()


def _on_gpu(t: torch.Tensor) -> bool:
    return t.is_cuda


def _stream(t: torch.Tensor) -> int:
    return torch.cuda.current_stream(t.device).cuda_stream


def _launch(
    t: torch.Tensor,
    program: str,
    name: str,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    args: Sequence[_KernelArg],
) -> None:
    """
    Launches a kernel of a program on t's device, on that device's current
    stream. The device is made current for the launch: the default stream
    (handle 0) is the null stream of whichever device is current.
    """
    hip = _get_hip()
    with torch.cuda.device(t.device):
        fn = hip.get(t.get_device(), program, name)
        hip.launch(fn, grid, block, args, _stream(t))


# --- depthwise 3x3 convolution ----------------------------------------------------


def can_dwconv3x3(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """
    Whether dwconv3x3 can take x for a depthwise conv with this (C, 1, 3, 3)
    weight. Channels-last inputs keep the original conv, which keeps their
    memory format.
    """
    return (
        _on_gpu(x)
        and x.dim() == 4
        and x.dtype in _DTYPE_SUFFIX
        and x.dtype == weight.dtype
        and x.device == weight.device
        and x.shape[1] == weight.shape[0]
        and x.numel() > 0
        and x.shape[0] * x.shape[1] <= _MAX_GRID
        and (
            x.is_contiguous() or not x.is_contiguous(memory_format=torch.channels_last)
        )
    )


def dwconv3x3(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    """
    Depthwise 3x3 convolution, stride 1, zero padding 1.

    x: (B, C, H, W). weight: (C, 9) float32 contiguous. bias: (C,) float32 or None.
    """
    x = x.contiguous()
    b, c, h, w = x.shape
    y = torch.empty_like(x)
    args: list[_KernelArg] = [
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(bias.data_ptr() if bias is not None else 0),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int(c),
        ctypes.c_int(h),
        ctypes.c_int(w),
    ]
    grid = ((w + _TW - 1) // _TW, (h + _TH - 1) // _TH, b * c)
    _launch(
        x, "dwconv", f"dwconv3x3_{_DTYPE_SUFFIX[x.dtype]}", grid, (_TW, _TH, 1), args
    )
    return y


# --- ESRGAN dense blocks ------------------------------------------------------------


def _image_stride(t: torch.Tensor) -> int | None:
    """
    The per-image element stride of an NCHW tensor whose channels each hold a
    contiguous H*W plane (e.g. a channel prefix of a larger buffer), or None.
    """
    _, c, h, w = t.shape
    if t.stride(3) != 1 or t.stride(2) != w or t.stride(1) != h * w:
        return None
    return t.stride(0) if t.shape[0] > 1 else c * h * w


def can_rrdb(x: torch.Tensor) -> bool:
    return (
        _on_gpu(x)
        and x.dim() == 4
        and x.dtype in _DTYPE_SUFFIX
        and x.numel() > 0
        and x.shape[0] <= _MAX_GRID
        and _image_stride(x) is not None
    )


def _grid(n: int, batch: int) -> tuple[int, int, int]:
    return (max(1, min((n + 255) // 256, 4096)), batch, 1)


def act_into(
    src: torch.Tensor, dst: torch.Tensor, channel_offset: int, slope: float
) -> None:
    """
    dst[:, channel_offset : channel_offset + C] = LeakyReLU(src), where dst is
    an NCHW buffer and src a (B, C, H, W) tensor of the same dtype.
    """
    b, c, h, w = src.shape
    dst_stride = _image_stride(dst)
    if (
        dst_stride is None
        or dst.dtype != src.dtype
        or dst.device != src.device
        or dst.shape[0] != b
        or dst.shape[2:] != src.shape[2:]
        or channel_offset + c > dst.shape[1]
    ):
        raise HipKernelError(
            f"act_into: unsupported tensors {tuple(src.shape)} {src.dtype} -> "
            f"{tuple(dst.shape)} {dst.dtype} at channel {channel_offset}"
        )
    src = src.contiguous()
    n = c * h * w
    args: list[_KernelArg] = [
        ctypes.c_void_p(src.data_ptr()),
        ctypes.c_void_p(dst.data_ptr()),
        ctypes.c_int64(n),
        ctypes.c_int64(dst_stride),
        ctypes.c_int64(channel_offset * h * w),
        ctypes.c_float(slope),
    ]
    _launch(
        src,
        "rrdb",
        f"act_into_{_DTYPE_SUFFIX[src.dtype]}",
        _grid(n, b),
        (256, 1, 1),
        args,
    )


def residual_add(
    a: torch.Tensor,
    x1: torch.Tensor,
    dst: torch.Tensor,
    s1: float,
    x2: torch.Tensor | None = None,
    s2: float = 0.0,
) -> None:
    """
    dst = a * s1 + x1, then * s2 + x2 if x2 is given, with the rounding of the
    separate PyTorch ops. All are (B, C, H, W) of one dtype; x1, x2 and dst may
    be channel prefixes of larger NCHW buffers.
    """
    b, c, h, w = a.shape
    x1_stride = _image_stride(x1)
    dst_stride = _image_stride(dst)
    x2_stride = _image_stride(x2) if x2 is not None else 0
    others = [x1, dst] if x2 is None else [x1, dst, x2]
    if (
        x1_stride is None
        or dst_stride is None
        or x2_stride is None
        or any(
            t.shape != a.shape or t.dtype != a.dtype or t.device != a.device
            for t in others
        )
    ):
        raise HipKernelError(
            "residual_add: unsupported tensors "
            + ", ".join(f"{tuple(t.shape)} {t.dtype}" for t in [a, *others])
        )
    a = a.contiguous()
    n = c * h * w
    args: list[_KernelArg] = [
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(x1.data_ptr()),
        ctypes.c_int64(x1_stride),
        ctypes.c_void_p(x2.data_ptr() if x2 is not None else 0),
        ctypes.c_int64(x2_stride),
        ctypes.c_void_p(dst.data_ptr()),
        ctypes.c_int64(dst_stride),
        ctypes.c_int64(n),
        ctypes.c_float(s1),
        ctypes.c_float(s2),
        ctypes.c_int(1 if x2 is not None else 0),
    ]
    _launch(
        a, "rrdb", f"res_add_{_DTYPE_SUFFIX[a.dtype]}", _grid(n, b), (256, 1, 1), args
    )
