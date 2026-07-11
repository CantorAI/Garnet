import ctypes
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]


def add_dll_dirs(garnet_dll):
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for dll_dir in [
        garnet_dll.parent,
        REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "out" / "build" / "x64-debug" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "lib",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"),
        Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/debug_nonredist/x64/Microsoft.VC145.DebugCRT"),
    ]:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))


def load_dll():
    garnet_dll = Path(os.environ.get("GARNET_DLL_PATH", REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll"))
    if not garnet_dll.exists():
        print(f"SKIP: garnet.dll not found: {garnet_dll}")
        raise SystemExit(0)
    add_dll_dirs(garnet_dll)
    dll = ctypes.CDLL(str(garnet_dll))
    dll.GarnetCreateDevicePagedKVFP32.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_longlong), ctypes.c_char_p, ctypes.c_int,
    ]
    dll.GarnetCreateDevicePagedKVFP32.restype = ctypes.c_int
    dll.GarnetDestroyDevicePagedKVFP32.argtypes = [ctypes.c_longlong, ctypes.c_char_p, ctypes.c_int]
    dll.GarnetDestroyDevicePagedKVFP32.restype = ctypes.c_int
    dll.GarnetDevicePagedKVWriteFP32.argtypes = [
        ctypes.c_longlong, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ]
    dll.GarnetDevicePagedKVWriteFP32.restype = ctypes.c_int
    dll.GarnetDevicePagedKVAttentionFP32.argtypes = [
        ctypes.c_longlong, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ]
    dll.GarnetDevicePagedKVAttentionFP32.restype = ctypes.c_int
    return dll


def reference(q, k_all, v_all):
    sequence_length, kv_heads, head_dim = k_all.shape
    q_heads = q.shape[0]
    group_size = q_heads // kv_heads
    out = np.empty_like(q)
    scale = 1.0 / np.sqrt(np.float32(head_dim))
    for q_head in range(q_heads):
        kv_head = q_head // group_size
        scores = (k_all[:, kv_head, :] @ q[q_head, :]) * scale
        scores = scores - np.max(scores)
        probs = np.exp(scores)
        probs = probs / np.sum(probs)
        out[q_head, :] = probs @ v_all[:, kv_head, :]
    return out


def main():
    rng = np.random.default_rng(20260719)
    sequence_length = int(os.environ.get("GARNET_DEVICE_KV_TEST_SEQUENCE_LENGTH", "41"))
    split_at = int(os.environ.get("GARNET_DEVICE_KV_TEST_SPLIT_AT", "29"))
    page_size = int(os.environ.get("GARNET_DEVICE_KV_TEST_PAGE_SIZE", "8"))
    q_heads = int(os.environ.get("GARNET_DEVICE_KV_TEST_Q_HEADS", "16"))
    kv_heads = int(os.environ.get("GARNET_DEVICE_KV_TEST_KV_HEADS", "8"))
    head_dim = int(os.environ.get("GARNET_DEVICE_KV_TEST_HEAD_DIM", "128"))
    logical_pages = (sequence_length + page_size - 1) // page_size
    physical_pages = logical_pages + 3
    page_table = np.roll(np.arange(logical_pages, dtype=np.int32), 1)

    q_all = rng.normal(0.0, 0.2, size=(sequence_length, q_heads, head_dim)).astype(np.float32)
    k_all = rng.normal(0.0, 0.2, size=(sequence_length, kv_heads, head_dim)).astype(np.float32)
    v_all = rng.normal(0.0, 0.2, size=(sequence_length, kv_heads, head_dim)).astype(np.float32)
    qkv = np.concatenate(
        [
            q_all.reshape(sequence_length, q_heads * head_dim),
            k_all.reshape(sequence_length, kv_heads * head_dim),
            v_all.reshape(sequence_length, kv_heads * head_dim),
        ],
        axis=1,
    ).astype(np.float32)
    q_last = np.ascontiguousarray(q_all[-1])
    expected = reference(q_last, k_all, v_all)
    actual = np.empty_like(expected)

    dll = load_dll()
    error = ctypes.create_string_buffer(512)
    handle = ctypes.c_longlong(0)
    rc = dll.GarnetCreateDevicePagedKVFP32(
        physical_pages,
        page_size,
        logical_pages,
        q_heads,
        kv_heads,
        head_dim,
        page_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.byref(handle),
        error,
        len(error),
    )
    if rc != 0:
        raise AssertionError(f"GarnetCreateDevicePagedKVFP32 failed rc={rc}: {error.value.decode(errors='ignore')}")
    try:
        first = np.ascontiguousarray(qkv[:split_at])
        second = np.ascontiguousarray(qkv[split_at:])
        rc = dll.GarnetDevicePagedKVWriteFP32(
            handle,
            first.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            split_at,
            0,
            error,
            len(error),
        )
        if rc != 0:
            raise AssertionError(f"device KV first write failed rc={rc}: {error.value.decode(errors='ignore')}")
        rc = dll.GarnetDevicePagedKVWriteFP32(
            handle,
            second.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            sequence_length - split_at,
            split_at,
            error,
            len(error),
        )
        if rc != 0:
            raise AssertionError(f"device KV second write failed rc={rc}: {error.value.decode(errors='ignore')}")
        rc = dll.GarnetDevicePagedKVAttentionFP32(
            handle,
            q_last.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            sequence_length,
            error,
            len(error),
        )
        if rc != 0:
            raise AssertionError(f"device KV attention failed rc={rc}: {error.value.decode(errors='ignore')}")
    finally:
        destroy_error = ctypes.create_string_buffer(512)
        rc_destroy = dll.GarnetDestroyDevicePagedKVFP32(handle, destroy_error, len(destroy_error))
        if rc_destroy != 0:
            raise AssertionError(f"destroy device KV failed rc={rc_destroy}: {destroy_error.value.decode(errors='ignore')}")

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    max_error = float(np.max(np.abs(actual - expected)))
    mean_error = float(np.mean(np.abs(actual - expected)))
    print("Phase 19: device-resident paged KV cache passed.")
    print(f"sequence_length={sequence_length} split_at={split_at} page_size={page_size}")
    print(f"pages={logical_pages}/{physical_pages} q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim}")
    print(f"max_error={max_error:.8f} mean_error={mean_error:.8f}")


if __name__ == "__main__":
    main()
