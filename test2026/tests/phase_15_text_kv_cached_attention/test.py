# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

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
    dll.GarnetRunTextKVCachedAttentionFP32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dll.GarnetRunTextKVCachedAttentionFP32.restype = ctypes.c_int
    dll.GarnetRunTextPagedKVCachedAttentionFP32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dll.GarnetRunTextPagedKVCachedAttentionFP32.restype = ctypes.c_int
    dll.GarnetRunTextPagedKVWriteFP32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dll.GarnetRunTextPagedKVWriteFP32.restype = ctypes.c_int
    return dll


def reference(q, k_cache, v_cache):
    sequence_length, kv_heads, head_dim = k_cache.shape
    q_heads = q.shape[0]
    group_size = q_heads // kv_heads
    out = np.empty_like(q)
    scale = 1.0 / np.sqrt(np.float32(head_dim))
    for q_head in range(q_heads):
        kv_head = q_head // group_size
        scores = (k_cache[:, kv_head, :] @ q[q_head, :]) * scale
        scores = scores - np.max(scores)
        probs = np.exp(scores)
        probs = probs / np.sum(probs)
        out[q_head, :] = probs @ v_cache[:, kv_head, :]
    return out


def full_causal_last_token_reference(q_all, k_all, v_all):
    sequence_length, q_heads, head_dim = q_all.shape
    kv_heads = k_all.shape[1]
    group_size = q_heads // kv_heads
    out = np.empty((q_heads, head_dim), dtype=np.float32)
    scale = 1.0 / np.sqrt(np.float32(head_dim))
    last = sequence_length - 1
    for q_head in range(q_heads):
        kv_head = q_head // group_size
        scores = (k_all[:, kv_head, :] @ q_all[last, q_head, :]) * scale
        scores = scores - np.max(scores)
        probs = np.exp(scores)
        probs = probs / np.sum(probs)
        out[q_head, :] = probs @ v_all[:, kv_head, :]
    return out


def main():
    rng = np.random.default_rng(20260710)
    sequence_length = int(os.environ.get("GARNET_TEXT_KV_TEST_SEQUENCE_LENGTH", "37"))
    q_heads = int(os.environ.get("GARNET_TEXT_KV_TEST_Q_HEADS", "16"))
    kv_heads = int(os.environ.get("GARNET_TEXT_KV_TEST_KV_HEADS", "2"))
    head_dim = int(os.environ.get("GARNET_TEXT_KV_TEST_HEAD_DIM", "128"))

    q = rng.normal(0.0, 0.2, size=(q_heads, head_dim)).astype(np.float32)
    k_cache = rng.normal(0.0, 0.2, size=(sequence_length, kv_heads, head_dim)).astype(np.float32)
    v_cache = rng.normal(0.0, 0.2, size=(sequence_length, kv_heads, head_dim)).astype(np.float32)
    expected = reference(q, k_cache, v_cache)
    actual = np.empty_like(q)
    error = ctypes.create_string_buffer(512)

    dll = load_dll()
    rc = dll.GarnetRunTextKVCachedAttentionFP32(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v_cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        actual.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sequence_length,
        q_heads,
        kv_heads,
        head_dim,
        error,
        len(error),
    )
    if rc != 0:
        raise AssertionError(f"GarnetRunTextKVCachedAttentionFP32 failed rc={rc}: {error.value.decode(errors='ignore')}")

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    max_abs_error = float(np.max(np.abs(actual - expected)))

    page_size = int(os.environ.get("GARNET_TEXT_KV_TEST_PAGE_SIZE", "8"))
    logical_pages = (sequence_length + page_size - 1) // page_size
    physical_pages = logical_pages + 3
    page_table = np.asarray([2, 0, 4, 1, 3, 5, 6, 7][:logical_pages], dtype=np.int32)
    if page_table.shape[0] < logical_pages:
        page_table = np.arange(logical_pages, dtype=np.int32)
    key_pages = np.zeros((physical_pages, page_size, kv_heads, head_dim), dtype=np.float32)
    value_pages = np.zeros_like(key_pages)
    for pos in range(sequence_length):
        logical_page = pos // page_size
        page_offset = pos % page_size
        physical_page = int(page_table[logical_page])
        key_pages[physical_page, page_offset, :, :] = k_cache[pos, :, :]
        value_pages[physical_page, page_offset, :, :] = v_cache[pos, :, :]

    actual_paged = np.empty_like(q)
    error2 = ctypes.create_string_buffer(512)
    rc = dll.GarnetRunTextPagedKVCachedAttentionFP32(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        key_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        value_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        page_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        actual_paged.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sequence_length,
        page_size,
        physical_pages,
        q_heads,
        kv_heads,
        head_dim,
        error2,
        len(error2),
    )
    if rc != 0:
        raise AssertionError(f"GarnetRunTextPagedKVCachedAttentionFP32 failed rc={rc}: {error2.value.decode(errors='ignore')}")
    np.testing.assert_allclose(actual_paged, expected, rtol=2e-5, atol=2e-5)
    max_abs_error_paged = float(np.max(np.abs(actual_paged - expected)))

    q_all = rng.normal(0.0, 0.2, size=(sequence_length, q_heads, head_dim)).astype(np.float32)
    k_all = rng.normal(0.0, 0.2, size=(sequence_length, kv_heads, head_dim)).astype(np.float32)
    v_all = rng.normal(0.0, 0.2, size=(sequence_length, kv_heads, head_dim)).astype(np.float32)
    expected_last = full_causal_last_token_reference(q_all, k_all, v_all)
    last_q = np.ascontiguousarray(q_all[-1])
    key_pages2 = np.zeros((physical_pages, page_size, kv_heads, head_dim), dtype=np.float32)
    value_pages2 = np.zeros_like(key_pages2)
    for pos in range(sequence_length):
        logical_page = pos // page_size
        page_offset = pos % page_size
        physical_page = int(page_table[logical_page])
        key_pages2[physical_page, page_offset, :, :] = k_all[pos, :, :]
        value_pages2[physical_page, page_offset, :, :] = v_all[pos, :, :]

    actual_last = np.empty_like(last_q)
    error3 = ctypes.create_string_buffer(512)
    rc = dll.GarnetRunTextPagedKVCachedAttentionFP32(
        last_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        key_pages2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        value_pages2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        page_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        actual_last.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sequence_length,
        page_size,
        physical_pages,
        q_heads,
        kv_heads,
        head_dim,
        error3,
        len(error3),
    )
    if rc != 0:
        raise AssertionError(f"last-token paged KV attention failed rc={rc}: {error3.value.decode(errors='ignore')}")
    np.testing.assert_allclose(actual_last, expected_last, rtol=2e-5, atol=2e-5)
    max_abs_error_last = float(np.max(np.abs(actual_last - expected_last)))

    qkv_all = np.concatenate(
        [
            q_all.reshape(sequence_length, q_heads * head_dim),
            k_all.reshape(sequence_length, kv_heads * head_dim),
            v_all.reshape(sequence_length, kv_heads * head_dim),
        ],
        axis=1,
    ).astype(np.float32)
    write_key_pages = np.full((physical_pages, page_size, kv_heads, head_dim), -777.0, dtype=np.float32)
    write_value_pages = np.full_like(write_key_pages, -333.0)
    write_error = ctypes.create_string_buffer(512)
    rc = dll.GarnetRunTextPagedKVWriteFP32(
        qkv_all.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        write_key_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        write_value_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        page_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        sequence_length,
        0,
        page_size,
        physical_pages,
        q_heads,
        kv_heads,
        head_dim,
        write_error,
        len(write_error),
    )
    if rc != 0:
        raise AssertionError(f"GarnetRunTextPagedKVWriteFP32 failed rc={rc}: {write_error.value.decode(errors='ignore')}")
    for pos in range(sequence_length):
        logical_page = pos // page_size
        page_offset = pos % page_size
        physical_page = int(page_table[logical_page])
        np.testing.assert_allclose(write_key_pages[physical_page, page_offset], k_all[pos], rtol=0, atol=0)
        np.testing.assert_allclose(write_value_pages[physical_page, page_offset], v_all[pos], rtol=0, atol=0)

    actual_written_last = np.empty_like(last_q)
    read_written_error = ctypes.create_string_buffer(512)
    rc = dll.GarnetRunTextPagedKVCachedAttentionFP32(
        last_q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        write_key_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        write_value_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        page_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        actual_written_last.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sequence_length,
        page_size,
        physical_pages,
        q_heads,
        kv_heads,
        head_dim,
        read_written_error,
        len(read_written_error),
    )
    if rc != 0:
        raise AssertionError(f"paged KV attention after write failed rc={rc}: {read_written_error.value.decode(errors='ignore')}")
    np.testing.assert_allclose(actual_written_last, expected_last, rtol=2e-5, atol=2e-5)
    max_abs_error_written_last = float(np.max(np.abs(actual_written_last - expected_last)))

    print("Phase 15: text KV cached attention CUDA primitive passed.")
    print(f"shape: q=[{q_heads}, {head_dim}], kv=[{sequence_length}, {kv_heads}, {head_dim}]")
    print(f"max_abs_error={max_abs_error:.8f}")
    print(f"paged: page_size={page_size}, logical_pages={logical_pages}, physical_pages={physical_pages}")
    print(f"max_abs_error_paged={max_abs_error_paged:.8f}")
    print(f"last_token_full_causal_parity_max_abs_error={max_abs_error_last:.8f}")
    print(f"write_then_read_last_token_max_abs_error={max_abs_error_written_last:.8f}")


if __name__ == "__main__":
    main()
