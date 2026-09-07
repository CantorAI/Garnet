import os
import re
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ARTIFACT_DIR = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_subgraphs"
GARNET_FOR_TENSOR_COPY = None


def env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def skip(message):
    print(f"SKIP: {message}")
    raise SystemExit(0)


def default_model_dir():
    explicit = os.environ.get("HF_QWEN_VL_MODEL_DIR") or os.environ.get("GARNET_QWEN_VL_WEIGHT_INDEX_OR_DIR")
    if explicit:
        return Path(explicit)

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
    if cache_root.exists():
        snapshots = sorted(cache_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            return snapshots[0]
    return None


def load_weight_map(model_dir):
    try:
        from safetensors import safe_open
    except Exception as exc:
        skip(f"safetensors is not available: {exc}")

    model_dir = Path(model_dir)
    files = list(model_dir.glob("*.safetensors"))
    if not files:
        skip(f"no safetensors files found in {model_dir}")

    key_to_file = {}
    for file_path in files:
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                normalized = key[6:] if key.startswith("model.") else key
                key_to_file[normalized] = file_path
    return key_to_file


def load_tensor(key_to_file, key):
    from safetensors import safe_open

    if key not in key_to_file:
        raise AssertionError(f"missing real Qwen weight: {key}")
    with safe_open(str(key_to_file[key]), framework="pt", device="cpu") as handle:
        raw_key = key if key in handle.keys() else "model." + key
        tensor = handle.get_tensor(raw_key).float().contiguous()
        return tensor.numpy().astype(np.float32, copy=False)


def load_bfloat16_gpu_tensor(garnet, key_to_file, key, reshape=None):
    from safetensors import safe_open
    import torch

    if key not in key_to_file:
        raise AssertionError(f"missing real Qwen weight: {key}")
    with safe_open(str(key_to_file[key]), framework="pt", device="cpu") as handle:
        raw_key = key if key in handle.keys() else "model." + key
        tensor = handle.get_tensor(raw_key).to(torch.bfloat16).contiguous()
        if reshape is not None:
            tensor = tensor.reshape(*reshape).contiguous()
        bits = tensor.view(torch.uint16).numpy()
    value = garnet.tensor_from_bfloat16_bits(bits)
    if value is None:
        raise AssertionError(f"failed to create BF16 GPU X::Tensor for {key}")
    return value


def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def make_text_mrope(tokens, head_dim=128, rope_theta=5000000.0, mrope_section=(24, 20, 20)):
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    # Deliberately use different temporal/height/width ids so interleaving is covered.
    position_ids = np.stack([
        np.arange(tokens, dtype=np.float32),
        np.arange(tokens, dtype=np.float32) + 17.0,
        np.arange(tokens, dtype=np.float32) + 31.0,
    ], axis=0)
    freqs = position_ids[:, :, None] * inv_freq[None, None, :]
    interleaved = freqs[0].copy()
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        interleaved[:, offset:length:3] = freqs[dim, :, offset:length:3]
    emb = np.concatenate([interleaved, interleaved], axis=-1).astype(np.float32)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def qwen3vl_vision_position_ids_numpy(start_position, grid_thw, temp_merge_size=1, spatial_merge_size=1, time_interval=1):
    grid = np.asarray(grid_thw, dtype=np.int64)
    llm_grid_t = int(grid[0]) // int(temp_merge_size)
    llm_grid_h = int(grid[1]) // int(spatial_merge_size)
    llm_grid_w = int(grid[2]) // int(spatial_merge_size)
    position_temporal = np.arange(llm_grid_t, dtype=np.int64) * int(time_interval)
    position_height = np.arange(llm_grid_h, dtype=np.int64) + int(start_position)
    position_width = np.arange(llm_grid_w, dtype=np.int64) + int(start_position)
    t_grid, h_grid, w_grid = np.meshgrid(position_temporal, position_height, position_width, indexing="ij")
    out = np.stack([t_grid, h_grid, w_grid], axis=0).reshape(3, -1)
    out[0] += int(start_position)
    return out


def qwen3vl_mrope_position_ids_numpy(input_ids, mm_token_type_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None, spatial_merge_size=2):
    input_ids = np.asarray(input_ids, dtype=np.int64)
    mm_token_type_ids = np.asarray(mm_token_type_ids, dtype=np.int64)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    if mm_token_type_ids.ndim == 1:
        mm_token_type_ids = mm_token_type_ids[None, :]
    if attention_mask is not None:
        attention_mask = np.asarray(attention_mask).astype(bool)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask[None, :]

    image_grids = [] if image_grid_thw is None else [np.asarray(x, dtype=np.int64) for x in np.asarray(image_grid_thw)]
    video_grids = []
    if video_grid_thw is not None:
        for grid in np.asarray(video_grid_thw, dtype=np.int64):
            for _ in range(int(grid[0])):
                frame_grid = grid.copy()
                frame_grid[0] = 1
                video_grids.append(frame_grid)
    grid_iters = {1: iter(image_grids), 2: iter(video_grids)}

    position_ids = np.zeros((3, input_ids.shape[0], input_ids.shape[1]), dtype=np.int64)
    deltas = []
    for batch_idx in range(input_ids.shape[0]):
        token_types = mm_token_type_ids[batch_idx]
        valid_mask = attention_mask[batch_idx] if attention_mask is not None else np.ones(input_ids.shape[1], dtype=bool)
        valid_types = token_types[valid_mask]

        current_pos = 0
        pieces = []
        idx = 0
        while idx < valid_types.shape[0]:
            modality = int(valid_types[idx])
            end = idx + 1
            while end < valid_types.shape[0] and int(valid_types[end]) == modality:
                end += 1
            span_len = end - idx
            if modality == 0:
                pieces.append(np.arange(span_len, dtype=np.int64)[None, :].repeat(3, axis=0) + current_pos)
                current_pos += span_len
            else:
                grid = next(grid_iters[modality])
                vision_pos = qwen3vl_vision_position_ids_numpy(
                    current_pos,
                    grid,
                    temp_merge_size=1,
                    spatial_merge_size=spatial_merge_size,
                )
                if vision_pos.shape[1] != span_len:
                    raise AssertionError(
                        f"vision token span length {span_len} does not match grid-derived length {vision_pos.shape[1]}"
                    )
                pieces.append(vision_pos)
                current_pos += max(int(grid[1]), int(grid[2])) // int(spatial_merge_size)
            idx = end

        llm_positions = np.concatenate(pieces, axis=1) if pieces else np.zeros((3, 0), dtype=np.int64)
        if attention_mask is not None:
            position_ids[:, batch_idx, valid_mask] = llm_positions
        else:
            position_ids[:, batch_idx, :] = llm_positions
        deltas.append(int(llm_positions.max() + 1 - valid_types.shape[0]) if llm_positions.size else 0)
    return position_ids, np.asarray(deltas, dtype=np.int64).reshape(-1, 1)


def qwen3vl_text_mrope_cos_sin_numpy(position_ids, head_dim=128, rope_theta=5000000.0, mrope_section=(24, 20, 20)):
    position_ids = np.asarray(position_ids, dtype=np.float32)
    if position_ids.ndim == 2:
        position_ids = np.broadcast_to(position_ids[None, :, :], (3, position_ids.shape[0], position_ids.shape[1]))
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = np.transpose(position_ids[:, :, None, :] * inv_freq[None, None, :, None], (0, 1, 3, 2))
    interleaved = freqs[0].copy()
    for dim, offset in enumerate((1, 2), start=1):
        length = int(mrope_section[dim]) * 3
        interleaved[..., offset:length:3] = freqs[dim, ..., offset:length:3]
    emb = np.concatenate([interleaved, interleaved], axis=-1).astype(np.float32)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def run_native_mrope_positions_against_hf(model_dir):
    try:
        import torch
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    except Exception as exc:
        skip(f"HF Qwen3-VL MRoPE helpers are not available: {exc}")

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    input_ids = np.asarray([[101, 102, 103, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 104, 105]], dtype=np.int64)
    mm_types = np.asarray([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=np.int64)
    image_grid = np.asarray([[1, 4, 8]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)

    expected_pos, expected_delta = qwen3vl_mrope_position_ids_numpy(
        input_ids,
        mm_types,
        image_grid_thw=image_grid,
        attention_mask=attention_mask,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    )

    model = object.__new__(Qwen3VLModel)
    model.config = config
    with torch.inference_mode():
        actual_pos, actual_delta = Qwen3VLModel.get_rope_index(
            model,
            torch.from_numpy(input_ids),
            mm_token_type_ids=torch.from_numpy(mm_types),
            image_grid_thw=torch.from_numpy(image_grid),
            attention_mask=torch.from_numpy(attention_mask),
        )
    actual_pos = actual_pos.detach().cpu().numpy()
    actual_delta = actual_delta.detach().cpu().numpy()
    np.testing.assert_array_equal(actual_pos, expected_pos)
    np.testing.assert_array_equal(actual_delta, expected_delta)
    return {
        "input_shape": list(input_ids.shape),
        "image_grid_thw": image_grid.tolist(),
        "position_ids_shape": list(actual_pos.shape),
        "mrope_position_deltas": [int(x) for x in actual_delta.reshape(-1).tolist()],
        "max_position": int(actual_pos.max()),
        "visual_span_len": int(image_grid[0, 0] * (image_grid[0, 1] // config.vision_config.spatial_merge_size) * (image_grid[0, 2] // config.vision_config.spatial_merge_size)),
    }


def run_native_mrope_cos_sin_against_hf(model_dir):
    try:
        import torch
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    except Exception as exc:
        skip(f"HF Qwen3-VL text rotary helpers are not available: {exc}")

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    text_config = config.text_config
    input_ids = np.asarray([[101, 102, 103, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 104, 105]], dtype=np.int64)
    mm_types = np.asarray([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=np.int64)
    image_grid = np.asarray([[1, 4, 8]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    position_ids, _ = qwen3vl_mrope_position_ids_numpy(
        input_ids,
        mm_types,
        image_grid_thw=image_grid,
        attention_mask=attention_mask,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    )
    expected_cos, expected_sin = qwen3vl_text_mrope_cos_sin_numpy(
        position_ids,
        head_dim=text_config.head_dim,
        rope_theta=text_config.rope_parameters["rope_theta"],
        mrope_section=text_config.rope_parameters.get("mrope_section", [24, 20, 20]),
    )
    with torch.inference_mode():
        rotary = Qwen3VLTextRotaryEmbedding(text_config).eval()
        dummy = torch.zeros((1, input_ids.shape[1], text_config.hidden_size), dtype=torch.float32)
        actual_cos, actual_sin = rotary(dummy, torch.from_numpy(position_ids))
    actual_cos = actual_cos.detach().cpu().numpy().astype(np.float32)
    actual_sin = actual_sin.detach().cpu().numpy().astype(np.float32)
    np.testing.assert_allclose(actual_cos, expected_cos, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_sin, expected_sin, rtol=1e-6, atol=1e-6)
    return {
        "position_ids_shape": list(position_ids.shape),
        "cos_shape": list(actual_cos.shape),
        "sin_shape": list(actual_sin.shape),
        "cos_max_error": float(np.max(np.abs(actual_cos - expected_cos))),
        "sin_max_error": float(np.max(np.abs(actual_sin - expected_sin))),
    }


def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if type(value).__name__ == "XlangObject" and GARNET_FOR_TENSOR_COPY is not None:
        cpu_value = GARNET_FOR_TENSOR_COPY.tensor_to_cpu(value)
        if cpu_value is not None:
            value = cpu_value
            to_array_fn = getattr(value, "toarray", None)
            if callable(to_array_fn):
                return to_array_fn()
    to_numpy_fn = getattr(value, "numpy", None)
    to_array_fn = getattr(value, "toarray", None)
    to_list_fn = getattr(value, "tolist", None)
    if callable(to_numpy_fn):
        return to_numpy_fn()
    if callable(to_array_fn):
        return to_array_fn()
    if callable(to_list_fn):
        return np.array(to_list_fn(), dtype=np.float32)
    raise AssertionError("Garnet output has no numpy/tolist conversion")


def logits_to_numpy_or_cpu_copy(garnet, value, rows, vocab_size):
    try:
        return to_numpy(value).reshape(rows, vocab_size)
    except AssertionError:
        tensor_to_cpu = getattr(garnet, "tensor_to_cpu", None)
        if not callable(tensor_to_cpu):
            raise
        cpu_value = tensor_to_cpu(value)
        if cpu_value is None:
            raise
        return to_numpy(cpu_value).reshape(rows, vocab_size)


def import_garnet():
    try:
        import xlang3
    except Exception as exc:
        skip(f"xlang Python module not available: {exc}")

    garnet_dll = Path(os.environ.get(
        "GARNET_DLL_PATH",
        REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
    ))
    if not garnet_dll.exists():
        skip(f"garnet.dll not found: {garnet_dll}")

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for dll_dir in [
            garnet_dll.parent,
            REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
        ]:
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))

    try:
        global GARNET_FOR_TENSOR_COPY
        garnet = xlang3.importModule("garnet", fromPath=str(garnet_dll))
        GARNET_FOR_TENSOR_COPY = garnet
        return garnet, garnet_dll
    except Exception as exc:
        skip(f"failed to import Garnet from {garnet_dll}: {exc}")


def add_garnet_dll_directories(garnet_dll):
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for dll_dir in [
        garnet_dll.parent,
        REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "out" / "build" / "x64-Debug" / "bin",
        REPO_ROOT.parent / "out" / "build" / "x64-debug" / "bin",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"),
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
        REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "lib",
        Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/debug_nonredist/x64/Microsoft.VC145.DebugCRT"),
    ]:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))


def load_garnet_dll_ctypes():
    import ctypes

    garnet_dll = Path(os.environ.get(
        "GARNET_DLL_PATH",
        REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
    ))
    add_garnet_dll_directories(garnet_dll)
    return ctypes.CDLL(str(garnet_dll))


def configure_paged_kv_attention_ctypes(dll):
    import ctypes

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
    dll.GarnetCreateDevicePagedKVFP32.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dll.GarnetCreateDevicePagedKVFP32.restype = ctypes.c_int
    dll.GarnetDestroyDevicePagedKVFP32.argtypes = [ctypes.c_longlong, ctypes.c_char_p, ctypes.c_int]
    dll.GarnetDestroyDevicePagedKVFP32.restype = ctypes.c_int
    dll.GarnetDevicePagedKVWriteFP32.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dll.GarnetDevicePagedKVWriteFP32.restype = ctypes.c_int
    dll.GarnetDevicePagedKVAttentionFP32.argtypes = [
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    dll.GarnetDevicePagedKVAttentionFP32.restype = ctypes.c_int
    return dll


def configure_native_tokenizer_ctypes(dll, model_dir):
    import ctypes

    dll.GarnetQwenTokenizerDecode.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int,
    ]
    dll.GarnetQwenTokenizerDecode.restype = ctypes.c_int
    dll.GarnetQwenTokenizerTokenId.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    dll.GarnetQwenTokenizerTokenId.restype = ctypes.c_longlong

    def native_token_id(token):
        return int(dll.GarnetQwenTokenizerTokenId(str(model_dir).encode("utf-8"), token.encode("utf-8")))

    def native_decode(ids, skip_special=True):
        arr = (ctypes.c_longlong * len(ids))(*[int(x) for x in ids])
        byte_count = ctypes.c_int(0)
        decode_error = ctypes.create_string_buffer(512)
        rc_decode = dll.GarnetQwenTokenizerDecode(
            str(model_dir).encode("utf-8"), arr, len(ids), 1 if skip_special else 0,
            None, 0, ctypes.byref(byte_count), decode_error, len(decode_error)
        )
        if rc_decode not in (0, 3):
            raise AssertionError(f"GarnetQwenTokenizerDecode count failed rc={rc_decode}: {decode_error.value.decode(errors='ignore')}")
        text_buf = ctypes.create_string_buffer(byte_count.value + 1)
        rc_decode = dll.GarnetQwenTokenizerDecode(
            str(model_dir).encode("utf-8"), arr, len(ids), 1 if skip_special else 0,
            text_buf, len(text_buf), ctypes.byref(byte_count), decode_error, len(decode_error)
        )
        if rc_decode != 0:
            raise AssertionError(f"GarnetQwenTokenizerDecode failed rc={rc_decode}: {decode_error.value.decode(errors='ignore')}")
        return text_buf.value.decode("utf-8", errors="replace")

    return native_token_id, native_decode


def prepare_qwen_vl_request_one_call(model_dir, image_path, user_prompt, processor_pixels):
    import ctypes

    dll = load_garnet_dll_ctypes()
    fn = dll.GarnetQwenVLPrepareJpegPrompt
    fn.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_longlong), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p, ctypes.c_int,
    ]
    fn.restype = ctypes.c_int

    input_count = ctypes.c_int(0)
    pixel_count = ctypes.c_int(0)
    grid = (ctypes.c_longlong * 3)()
    source_h = ctypes.c_int(0)
    source_w = ctypes.c_int(0)
    resized_h = ctypes.c_int(0)
    resized_w = ctypes.c_int(0)
    error = ctypes.create_string_buffer(512)
    rc = fn(
        str(model_dir).encode("utf-8"),
        str(image_path).encode("utf-8"),
        user_prompt.encode("utf-8"),
        int(processor_pixels),
        int(processor_pixels),
        None,
        0,
        ctypes.byref(input_count),
        None,
        0,
        None,
        0,
        ctypes.byref(pixel_count),
        grid,
        ctypes.byref(source_h),
        ctypes.byref(source_w),
        ctypes.byref(resized_h),
        ctypes.byref(resized_w),
        error,
        len(error),
    )
    if rc not in (0, 3):
        raise AssertionError(f"GarnetQwenVLPrepareJpegPrompt count failed rc={rc}: {error.value.decode(errors='ignore')}")

    input_ids_buf = (ctypes.c_longlong * input_count.value)()
    mm_types_buf = (ctypes.c_longlong * input_count.value)()
    pixel_values_buf = (ctypes.c_float * pixel_count.value)()
    error2 = ctypes.create_string_buffer(512)
    rc = fn(
        str(model_dir).encode("utf-8"),
        str(image_path).encode("utf-8"),
        user_prompt.encode("utf-8"),
        int(processor_pixels),
        int(processor_pixels),
        input_ids_buf,
        input_count.value,
        ctypes.byref(input_count),
        mm_types_buf,
        input_count.value,
        pixel_values_buf,
        pixel_count.value,
        ctypes.byref(pixel_count),
        grid,
        ctypes.byref(source_h),
        ctypes.byref(source_w),
        ctypes.byref(resized_h),
        ctypes.byref(resized_w),
        error2,
        len(error2),
    )
    if rc != 0:
        raise AssertionError(f"GarnetQwenVLPrepareJpegPrompt failed rc={rc}: {error2.value.decode(errors='ignore')}")

    grid_np = np.asarray([[int(grid[0]), int(grid[1]), int(grid[2])]], dtype=np.int64)
    patch_count = int(grid_np[0, 1] * grid_np[0, 2])
    feature_dim = 3 * 2 * 16 * 16
    if pixel_count.value != patch_count * feature_dim:
        raise AssertionError(f"pixel count {pixel_count.value} != expected {patch_count * feature_dim}")
    pixel_values = np.ctypeslib.as_array(pixel_values_buf).copy().reshape(patch_count, feature_dim).astype(np.float32)
    input_ids = np.ctypeslib.as_array(input_ids_buf).copy().astype(np.int64)
    mm_types = np.ctypeslib.as_array(mm_types_buf).copy().astype(np.int64)
    native_token_id, native_decode = configure_native_tokenizer_ctypes(dll, model_dir)
    return {
        "input_ids": input_ids,
        "mm_types": mm_types,
        "image_grid_thw": grid_np,
        "pixel_values": pixel_values,
        "native_token_id": native_token_id,
        "native_decode": native_decode,
        "processor_npz": "garnet_one_call_prepare_jpeg_prompt",
        "source_size": [int(source_h.value), int(source_w.value)],
        "resized_size": [int(resized_h.value), int(resized_w.value)],
    }


def run_rms_norm(garnet, key_to_file):
    key = "language_model.layers.0.input_layernorm.weight"
    weight = load_tensor(key_to_file, key)
    rng = np.random.default_rng(20260708)
    x = rng.normal(0.0, 0.02, size=(7, weight.shape[0])).astype(np.float32)
    denom = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + 1.0e-6)
    expected = (x / denom) * weight

    engine = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={key: weight},
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="rms_norm",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real RMSNorm")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    return {
        "input_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_layer_norm(garnet, key_to_file):
    weight_key = "visual.blocks.0.norm1.weight"
    bias_key = "visual.blocks.0.norm1.bias"
    weight = load_tensor(key_to_file, weight_key)
    bias = load_tensor(key_to_file, bias_key)
    rng = np.random.default_rng(20260709)
    x = rng.normal(0.0, 0.02, size=(9, weight.shape[0])).astype(np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    expected = ((x - mean) / np.sqrt(variance + 1.0e-6)) * weight + bias

    engine = garnet.load_model(
        str(SCRIPT_DIR / "layer_norm_trt.x"),
        weights={weight_key: weight, bias_key: bias},
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="layer_norm",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real LayerNorm")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    return {
        "input_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_text_qkv_proj(garnet, key_to_file):
    prefix = "language_model.layers.0.self_attn"
    weights = {
        prefix + ".q_proj.weight": load_tensor(key_to_file, prefix + ".q_proj.weight"),
        prefix + ".k_proj.weight": load_tensor(key_to_file, prefix + ".k_proj.weight"),
        prefix + ".v_proj.weight": load_tensor(key_to_file, prefix + ".v_proj.weight"),
    }
    rng = np.random.default_rng(20260710)
    x = rng.normal(0.0, 0.02, size=(4, weights[prefix + ".q_proj.weight"].shape[1])).astype(np.float32)
    q = x @ weights[prefix + ".q_proj.weight"].T
    k = x @ weights[prefix + ".k_proj.weight"].T
    v = x @ weights[prefix + ".v_proj.weight"].T
    expected = np.concatenate([q, k, v], axis=-1)

    engine = garnet.load_model(
        str(SCRIPT_DIR / "text_qkv_proj_trt.x"),
        weights=weights,
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="text_qkv_proj",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real TextQKV projection")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=7e-3, atol=7e-3)
    return {
        "input_shape": list(x.shape),
        "q_shape": list(weights[prefix + ".q_proj.weight"].shape),
        "k_shape": list(weights[prefix + ".k_proj.weight"].shape),
        "v_shape": list(weights[prefix + ".v_proj.weight"].shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_text_qkv_head_norm(garnet, key_to_file):
    prefix = "language_model.layers.0.self_attn"
    weights = {
        prefix + ".q_proj.weight": load_tensor(key_to_file, prefix + ".q_proj.weight"),
        prefix + ".k_proj.weight": load_tensor(key_to_file, prefix + ".k_proj.weight"),
        prefix + ".v_proj.weight": load_tensor(key_to_file, prefix + ".v_proj.weight"),
        prefix + ".q_norm.weight": load_tensor(key_to_file, prefix + ".q_norm.weight"),
        prefix + ".k_norm.weight": load_tensor(key_to_file, prefix + ".k_norm.weight"),
    }
    head_dim = weights[prefix + ".q_norm.weight"].shape[0]
    q_out = weights[prefix + ".q_proj.weight"].shape[0]
    k_out = weights[prefix + ".k_proj.weight"].shape[0]
    q_heads = q_out // head_dim
    kv_heads = k_out // head_dim

    rng = np.random.default_rng(20260711)
    x = rng.normal(0.0, 0.02, size=(4, weights[prefix + ".q_proj.weight"].shape[1])).astype(np.float32)
    q = x @ weights[prefix + ".q_proj.weight"].T
    k = x @ weights[prefix + ".k_proj.weight"].T
    v = x @ weights[prefix + ".v_proj.weight"].T

    q_heads_tensor = q.reshape(x.shape[0], q_heads, head_dim)
    k_heads_tensor = k.reshape(x.shape[0], kv_heads, head_dim)
    q_norm = q_heads_tensor / np.sqrt(np.mean(np.square(q_heads_tensor), axis=-1, keepdims=True) + 1.0e-6)
    k_norm = k_heads_tensor / np.sqrt(np.mean(np.square(k_heads_tensor), axis=-1, keepdims=True) + 1.0e-6)
    q_norm = (q_norm * weights[prefix + ".q_norm.weight"]).reshape(x.shape[0], q_out)
    k_norm = (k_norm * weights[prefix + ".k_norm.weight"]).reshape(x.shape[0], k_out)
    expected = np.concatenate([q_norm, k_norm, v], axis=-1)

    engine = garnet.load_model(
        str(SCRIPT_DIR / "text_qkv_head_norm_trt.x"),
        weights=weights,
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="text_qkv_head_norm",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real TextQKV head norm")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=7e-3, atol=7e-3)
    q_actual = actual[:, :q_out]
    k_actual = actual[:, q_out:q_out + k_out]
    v_actual = actual[:, q_out + k_out:]
    return {
        "input_shape": list(x.shape),
        "q_heads": int(q_heads),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
        "q_max_error": float(np.max(np.abs(q_actual - q_norm))),
        "k_max_error": float(np.max(np.abs(k_actual - k_norm))),
        "v_max_error": float(np.max(np.abs(v_actual - v))),
    }


def run_text_o_proj(garnet, key_to_file):
    key = "language_model.layers.0.self_attn.o_proj.weight"
    weight = load_tensor(key_to_file, key)
    rng = np.random.default_rng(20260712)
    x = rng.normal(0.0, 0.02, size=(4, weight.shape[1])).astype(np.float32)
    expected = x @ weight.T

    engine = garnet.load_model(
        str(SCRIPT_DIR / "text_o_proj_trt.x"),
        weights={key: weight},
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="text_o_proj",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real text o_proj")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=7e-3, atol=7e-3)
    return {
        "input_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_text_rope_apply(garnet, key_to_file):
    prefix = "language_model.layers.0.self_attn"
    weights = {
        prefix + ".q_proj.weight": load_tensor(key_to_file, prefix + ".q_proj.weight"),
        prefix + ".k_proj.weight": load_tensor(key_to_file, prefix + ".k_proj.weight"),
        prefix + ".v_proj.weight": load_tensor(key_to_file, prefix + ".v_proj.weight"),
        prefix + ".q_norm.weight": load_tensor(key_to_file, prefix + ".q_norm.weight"),
        prefix + ".k_norm.weight": load_tensor(key_to_file, prefix + ".k_norm.weight"),
    }
    head_dim = weights[prefix + ".q_norm.weight"].shape[0]
    q_out = weights[prefix + ".q_proj.weight"].shape[0]
    k_out = weights[prefix + ".k_proj.weight"].shape[0]
    q_heads = q_out // head_dim
    kv_heads = k_out // head_dim

    rng = np.random.default_rng(20260713)
    x = rng.normal(0.0, 0.02, size=(4, weights[prefix + ".q_proj.weight"].shape[1])).astype(np.float32)
    q = x @ weights[prefix + ".q_proj.weight"].T
    k = x @ weights[prefix + ".k_proj.weight"].T
    v = x @ weights[prefix + ".v_proj.weight"].T
    q_heads_tensor = q.reshape(x.shape[0], q_heads, head_dim)
    k_heads_tensor = k.reshape(x.shape[0], kv_heads, head_dim)
    q_norm = q_heads_tensor / np.sqrt(np.mean(np.square(q_heads_tensor), axis=-1, keepdims=True) + 1.0e-6)
    k_norm = k_heads_tensor / np.sqrt(np.mean(np.square(k_heads_tensor), axis=-1, keepdims=True) + 1.0e-6)
    q_norm = q_norm * weights[prefix + ".q_norm.weight"]
    k_norm = k_norm * weights[prefix + ".k_norm.weight"]

    cos, sin = make_text_mrope(x.shape[0], head_dim=head_dim)
    q_rope = (q_norm * cos[:, None, :]) + (rotate_half(q_norm) * sin[:, None, :])
    k_rope = (k_norm * cos[:, None, :]) + (rotate_half(k_norm) * sin[:, None, :])
    expected = np.concatenate([
        q_rope.reshape(x.shape[0], q_out),
        k_rope.reshape(x.shape[0], k_out),
        v,
    ], axis=-1)
    qkv = np.concatenate([
        q_norm.reshape(x.shape[0], q_out),
        k_norm.reshape(x.shape[0], k_out),
        v,
    ], axis=-1).astype(np.float32)

    engine = garnet.load_model(
        str(SCRIPT_DIR / "text_rope_apply_trt.x"),
        weights={},
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(qkv.shape), list(cos.shape), list(sin.shape)],
        subgraph="text_rope_apply",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real text RoPE apply")

    actual = to_numpy(engine.forward(qkv, cos, sin)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=8e-3, atol=8e-3)
    return {
        "input_shape": list(qkv.shape),
        "cos_shape": list(cos.shape),
        "sin_shape": list(sin.shape),
        "q_heads": int(q_heads),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
        "v_max_error": float(np.max(np.abs(actual[:, q_out + k_out:] - v))),
    }


def build_text_attention_inputs(key_to_file):
    prefix = "language_model.layers.0.self_attn"
    weights = {
        prefix + ".q_proj.weight": load_tensor(key_to_file, prefix + ".q_proj.weight"),
        prefix + ".k_proj.weight": load_tensor(key_to_file, prefix + ".k_proj.weight"),
        prefix + ".v_proj.weight": load_tensor(key_to_file, prefix + ".v_proj.weight"),
        prefix + ".q_norm.weight": load_tensor(key_to_file, prefix + ".q_norm.weight"),
        prefix + ".k_norm.weight": load_tensor(key_to_file, prefix + ".k_norm.weight"),
    }
    head_dim = weights[prefix + ".q_norm.weight"].shape[0]
    q_out = weights[prefix + ".q_proj.weight"].shape[0]
    k_out = weights[prefix + ".k_proj.weight"].shape[0]
    q_heads = q_out // head_dim
    kv_heads = k_out // head_dim
    rng = np.random.default_rng(20260714)
    x = rng.normal(0.0, 0.02, size=(4, weights[prefix + ".q_proj.weight"].shape[1])).astype(np.float32)

    q = x @ weights[prefix + ".q_proj.weight"].T
    k = x @ weights[prefix + ".k_proj.weight"].T
    v = x @ weights[prefix + ".v_proj.weight"].T
    q = q.reshape(x.shape[0], q_heads, head_dim)
    k = k.reshape(x.shape[0], kv_heads, head_dim)
    v = v.reshape(x.shape[0], kv_heads, head_dim)
    q = q / np.sqrt(np.mean(np.square(q), axis=-1, keepdims=True) + 1.0e-6)
    k = k / np.sqrt(np.mean(np.square(k), axis=-1, keepdims=True) + 1.0e-6)
    q = q * weights[prefix + ".q_norm.weight"]
    k = k * weights[prefix + ".k_norm.weight"]

    cos, sin = make_text_mrope(x.shape[0], head_dim=head_dim)
    q = (q * cos[:, None, :]) + (rotate_half(q) * sin[:, None, :])
    k = (k * cos[:, None, :]) + (rotate_half(k) * sin[:, None, :])
    qkv_rope = np.concatenate([
        q.reshape(x.shape[0], q_out),
        k.reshape(x.shape[0], k_out),
        v.reshape(x.shape[0], k_out),
    ], axis=-1).astype(np.float32)
    return qkv_rope, q, k, v, q_heads, kv_heads, head_dim


def text_attention_reference_from_hidden(x, key_to_file):
    prefix = "language_model.layers.0.self_attn"
    q_w = load_tensor(key_to_file, prefix + ".q_proj.weight")
    k_w = load_tensor(key_to_file, prefix + ".k_proj.weight")
    v_w = load_tensor(key_to_file, prefix + ".v_proj.weight")
    q_norm_w = load_tensor(key_to_file, prefix + ".q_norm.weight")
    k_norm_w = load_tensor(key_to_file, prefix + ".k_norm.weight")
    head_dim = q_norm_w.shape[0]
    q_out = q_w.shape[0]
    k_out = k_w.shape[0]
    q_heads = q_out // head_dim
    kv_heads = k_out // head_dim

    q = (x @ q_w.T).reshape(x.shape[0], q_heads, head_dim)
    k = (x @ k_w.T).reshape(x.shape[0], kv_heads, head_dim)
    v = (x @ v_w.T).reshape(x.shape[0], kv_heads, head_dim)
    q = q / np.sqrt(np.mean(np.square(q), axis=-1, keepdims=True) + 1.0e-6)
    k = k / np.sqrt(np.mean(np.square(k), axis=-1, keepdims=True) + 1.0e-6)
    q = q * q_norm_w
    k = k * k_norm_w
    cos, sin = make_text_mrope(x.shape[0], head_dim=head_dim)
    q = (q * cos[:, None, :]) + (rotate_half(q) * sin[:, None, :])
    k = (k * cos[:, None, :]) + (rotate_half(k) * sin[:, None, :])

    q_htd = np.transpose(q, (1, 0, 2))
    k_htd = np.transpose(k, (1, 0, 2))
    v_htd = np.transpose(v, (1, 0, 2))
    repeat = q_heads // kv_heads
    k_rep = np.repeat(k_htd, repeat, axis=0)
    v_rep = np.repeat(v_htd, repeat, axis=0)
    scores = np.matmul(q_htd, np.swapaxes(k_rep, 1, 2)) * (head_dim ** -0.5)
    causal_mask = np.triu(np.full((x.shape[0], x.shape[0]), -10000.0, dtype=np.float32), k=1)
    attn_weights = softmax(scores + causal_mask[None, :, :], axis=-1).astype(np.float32)
    context = np.matmul(attn_weights, v_rep)
    return np.transpose(context, (1, 0, 2)).reshape(x.shape[0], q_out)


def run_text_attention_core(garnet, key_to_file):
    qkv_rope, q, k, v, q_heads, kv_heads, head_dim = build_text_attention_inputs(key_to_file)
    q_htd = np.transpose(q, (1, 0, 2))
    k_htd = np.transpose(k, (1, 0, 2))
    v_htd = np.transpose(v, (1, 0, 2))
    repeat = q_heads // kv_heads
    k_rep = np.repeat(k_htd, repeat, axis=0)
    v_rep = np.repeat(v_htd, repeat, axis=0)
    scores = np.matmul(q_htd, np.swapaxes(k_rep, 1, 2)) * (head_dim ** -0.5)
    causal_mask = np.triu(np.full((qkv_rope.shape[0], qkv_rope.shape[0]), -10000.0, dtype=np.float32), k=1)
    weights = softmax(scores + causal_mask[None, :, :], axis=-1).astype(np.float32)
    context = np.matmul(weights, v_rep)
    expected = np.transpose(context, (1, 0, 2)).reshape(qkv_rope.shape[0], q_heads * head_dim)

    engine = garnet.load_model(
        str(SCRIPT_DIR / "text_attention_core_trt.x"),
        weights={},
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(qkv_rope.shape)],
        subgraph="text_attention_core",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real text attention core")

    actual = to_numpy(engine.forward(qkv_rope)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=8e-3, atol=8e-3)
    return {
        "input_shape": list(qkv_rope.shape),
        "q_heads": int(q_heads),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_text_decoder_layer_chain(garnet, key_to_file):
    layer_prefix = "language_model.layers.0"
    attn_prefix = layer_prefix + ".self_attn"
    mlp_prefix = layer_prefix + ".mlp"
    weights = {
        layer_prefix + ".input_layernorm.weight": load_tensor(key_to_file, layer_prefix + ".input_layernorm.weight"),
        layer_prefix + ".post_attention_layernorm.weight": load_tensor(key_to_file, layer_prefix + ".post_attention_layernorm.weight"),
        attn_prefix + ".q_proj.weight": load_tensor(key_to_file, attn_prefix + ".q_proj.weight"),
        attn_prefix + ".k_proj.weight": load_tensor(key_to_file, attn_prefix + ".k_proj.weight"),
        attn_prefix + ".v_proj.weight": load_tensor(key_to_file, attn_prefix + ".v_proj.weight"),
        attn_prefix + ".q_norm.weight": load_tensor(key_to_file, attn_prefix + ".q_norm.weight"),
        attn_prefix + ".k_norm.weight": load_tensor(key_to_file, attn_prefix + ".k_norm.weight"),
        attn_prefix + ".o_proj.weight": load_tensor(key_to_file, attn_prefix + ".o_proj.weight"),
        mlp_prefix + ".gate_proj.weight": load_tensor(key_to_file, mlp_prefix + ".gate_proj.weight"),
        mlp_prefix + ".up_proj.weight": load_tensor(key_to_file, mlp_prefix + ".up_proj.weight"),
        mlp_prefix + ".down_proj.weight": load_tensor(key_to_file, mlp_prefix + ".down_proj.weight"),
    }
    rng = np.random.default_rng(20260715)
    x = rng.normal(0.0, 0.02, size=(4, weights[layer_prefix + ".input_layernorm.weight"].shape[0])).astype(np.float32)

    x_norm = x / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + 1.0e-6)
    x_norm = x_norm * weights[layer_prefix + ".input_layernorm.weight"]
    attn_core = text_attention_reference_from_hidden(x_norm, key_to_file)
    attn_out = attn_core @ weights[attn_prefix + ".o_proj.weight"].T
    hidden = x + attn_out
    post_norm = hidden / np.sqrt(np.mean(np.square(hidden), axis=-1, keepdims=True) + 1.0e-6)
    post_norm = post_norm * weights[layer_prefix + ".post_attention_layernorm.weight"]
    gate = post_norm @ weights[mlp_prefix + ".gate_proj.weight"].T
    up = post_norm @ weights[mlp_prefix + ".up_proj.weight"].T
    mlp = (silu(gate) * up) @ weights[mlp_prefix + ".down_proj.weight"].T
    expected = hidden + mlp

    cache_dir = SCRIPT_DIR / "cache_decoder_layer"
    input_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={layer_prefix + ".input_layernorm.weight": weights[layer_prefix + ".input_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[list(x.shape)],
        subgraph="rms_norm",
    )
    gx_norm = to_numpy(input_norm_model.forward(x)).reshape(x.shape)

    qkv_model = garnet.load_model(
        str(SCRIPT_DIR / "text_qkv_head_norm_trt.x"),
        weights={
            attn_prefix + ".q_proj.weight": weights[attn_prefix + ".q_proj.weight"],
            attn_prefix + ".k_proj.weight": weights[attn_prefix + ".k_proj.weight"],
            attn_prefix + ".v_proj.weight": weights[attn_prefix + ".v_proj.weight"],
            attn_prefix + ".q_norm.weight": weights[attn_prefix + ".q_norm.weight"],
            attn_prefix + ".k_norm.weight": weights[attn_prefix + ".k_norm.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[list(x.shape)],
        subgraph="text_qkv_head_norm",
    )
    gqkv = to_numpy(qkv_model.forward(gx_norm)).reshape(x.shape[0], 4096)

    cos, sin = make_text_mrope(x.shape[0], head_dim=128)
    rope_model = garnet.load_model(
        str(SCRIPT_DIR / "text_rope_apply_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 4096], [x.shape[0], 128], [x.shape[0], 128]],
        subgraph="text_rope_apply",
    )
    grope = to_numpy(rope_model.forward(gqkv, cos, sin)).reshape(x.shape[0], 4096)

    attention_model = garnet.load_model(
        str(SCRIPT_DIR / "text_attention_core_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 4096]],
        subgraph="text_attention_core",
    )
    gattn = to_numpy(attention_model.forward(grope)).reshape(x.shape[0], 2048)

    o_proj_model = garnet.load_model(
        str(SCRIPT_DIR / "text_o_proj_trt.x"),
        weights={attn_prefix + ".o_proj.weight": weights[attn_prefix + ".o_proj.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 2048]],
        subgraph="text_o_proj",
    )
    go = to_numpy(o_proj_model.forward(gattn)).reshape(x.shape[0], 2048)
    ghidden = x + go

    post_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "text_post_rms_norm_trt.x"),
        weights={layer_prefix + ".post_attention_layernorm.weight": weights[layer_prefix + ".post_attention_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 2048]],
        subgraph="text_post_attention_rms_norm",
    )
    gpost = to_numpy(post_norm_model.forward(ghidden)).reshape(x.shape)

    mlp_model = garnet.load_model(
        str(SCRIPT_DIR / "text_mlp_trt.x"),
        weights={
            mlp_prefix + ".gate_proj.weight": weights[mlp_prefix + ".gate_proj.weight"],
            mlp_prefix + ".up_proj.weight": weights[mlp_prefix + ".up_proj.weight"],
            mlp_prefix + ".down_proj.weight": weights[mlp_prefix + ".down_proj.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 2048]],
        subgraph="qwen3_text_mlp",
    )
    gmlp = to_numpy(mlp_model.forward(gpost)).reshape(x.shape[0], 2048)
    actual = ghidden + gmlp

    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-1)
    return {
        "input_shape": list(x.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
        "post_norm_path": "garnet_trt",
    }


def load_hf_text_layer(model_dir, key_to_file, layer_idx):
    try:
        import torch
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
            Qwen3VLTextRotaryEmbedding,
            create_causal_mask,
        )
    except Exception as exc:
        skip(f"HF text decoder layer dependencies are not available: {exc}")

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True).text_config
    config._attn_implementation = "eager"
    layer = Qwen3VLTextDecoderLayer(config, layer_idx=layer_idx).eval()
    state = {}
    prefix = f"language_model.layers.{layer_idx}."
    for key in [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]:
        state[key] = torch.from_numpy(load_tensor(key_to_file, prefix + key))
    missing, unexpected = layer.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise AssertionError(f"HF layer-0 load mismatch: missing={missing}, unexpected={unexpected}")

    rotary = Qwen3VLTextRotaryEmbedding(config).eval()
    return torch, config, layer, rotary, create_causal_mask


def run_text_decoder_layer_against_hf_module(garnet, model_dir, key_to_file):
    torch, config, layer, rotary, create_causal_mask = load_hf_text_layer(model_dir, key_to_file, 0)
    layer_prefix = "language_model.layers.0"
    attn_prefix = layer_prefix + ".self_attn"
    mlp_prefix = layer_prefix + ".mlp"
    weights = {
        layer_prefix + ".input_layernorm.weight": load_tensor(key_to_file, layer_prefix + ".input_layernorm.weight"),
        layer_prefix + ".post_attention_layernorm.weight": load_tensor(key_to_file, layer_prefix + ".post_attention_layernorm.weight"),
        attn_prefix + ".q_proj.weight": load_tensor(key_to_file, attn_prefix + ".q_proj.weight"),
        attn_prefix + ".k_proj.weight": load_tensor(key_to_file, attn_prefix + ".k_proj.weight"),
        attn_prefix + ".v_proj.weight": load_tensor(key_to_file, attn_prefix + ".v_proj.weight"),
        attn_prefix + ".q_norm.weight": load_tensor(key_to_file, attn_prefix + ".q_norm.weight"),
        attn_prefix + ".k_norm.weight": load_tensor(key_to_file, attn_prefix + ".k_norm.weight"),
        attn_prefix + ".o_proj.weight": load_tensor(key_to_file, attn_prefix + ".o_proj.weight"),
        mlp_prefix + ".gate_proj.weight": load_tensor(key_to_file, mlp_prefix + ".gate_proj.weight"),
        mlp_prefix + ".up_proj.weight": load_tensor(key_to_file, mlp_prefix + ".up_proj.weight"),
        mlp_prefix + ".down_proj.weight": load_tensor(key_to_file, mlp_prefix + ".down_proj.weight"),
    }
    rng = np.random.default_rng(20260716)
    x = rng.normal(0.0, 0.02, size=(4, config.hidden_size)).astype(np.float32)

    with torch.inference_mode():
        hx = torch.from_numpy(x).unsqueeze(0)
        position_ids = torch.stack([
            torch.arange(x.shape[0], dtype=torch.long),
            torch.arange(x.shape[0], dtype=torch.long) + 17,
            torch.arange(x.shape[0], dtype=torch.long) + 31,
        ], dim=0).unsqueeze(1)
        text_position_ids = torch.arange(x.shape[0], dtype=torch.long).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=config,
            inputs_embeds=hx,
            attention_mask=torch.ones(1, x.shape[0], dtype=torch.long),
            past_key_values=None,
            position_ids=text_position_ids,
        )
        cos, sin = rotary(hx, position_ids)
        expected = layer(
            hx,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        ).squeeze(0).detach().cpu().numpy().astype(np.float32)

    cache_dir = SCRIPT_DIR / "cache_decoder_layer_hf"
    input_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={layer_prefix + ".input_layernorm.weight": weights[layer_prefix + ".input_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[list(x.shape)],
        subgraph="rms_norm",
    )
    gx_norm = to_numpy(input_norm_model.forward(x)).reshape(x.shape)

    qkv_model = garnet.load_model(
        str(SCRIPT_DIR / "text_qkv_head_norm_trt.x"),
        weights={
            attn_prefix + ".q_proj.weight": weights[attn_prefix + ".q_proj.weight"],
            attn_prefix + ".k_proj.weight": weights[attn_prefix + ".k_proj.weight"],
            attn_prefix + ".v_proj.weight": weights[attn_prefix + ".v_proj.weight"],
            attn_prefix + ".q_norm.weight": weights[attn_prefix + ".q_norm.weight"],
            attn_prefix + ".k_norm.weight": weights[attn_prefix + ".k_norm.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[list(x.shape)],
        subgraph="text_qkv_head_norm",
    )
    gqkv = to_numpy(qkv_model.forward(gx_norm)).reshape(x.shape[0], 4096)

    rope_model = garnet.load_model(
        str(SCRIPT_DIR / "text_rope_apply_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 4096], [x.shape[0], 128], [x.shape[0], 128]],
        subgraph="text_rope_apply",
    )
    grope = to_numpy(rope_model.forward(
        gqkv,
        cos.squeeze(0).detach().cpu().numpy().astype(np.float32),
        sin.squeeze(0).detach().cpu().numpy().astype(np.float32),
    )).reshape(x.shape[0], 4096)

    attention_model = garnet.load_model(
        str(SCRIPT_DIR / "text_attention_core_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 4096]],
        subgraph="text_attention_core",
    )
    gattn = to_numpy(attention_model.forward(grope)).reshape(x.shape[0], 2048)

    o_proj_model = garnet.load_model(
        str(SCRIPT_DIR / "text_o_proj_trt.x"),
        weights={attn_prefix + ".o_proj.weight": weights[attn_prefix + ".o_proj.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 2048]],
        subgraph="text_o_proj",
    )
    go = to_numpy(o_proj_model.forward(gattn)).reshape(x.shape[0], 2048)
    ghidden = x + go

    post_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "text_post_rms_norm_trt.x"),
        weights={layer_prefix + ".post_attention_layernorm.weight": weights[layer_prefix + ".post_attention_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 2048]],
        subgraph="text_post_attention_rms_norm",
    )
    gpost = to_numpy(post_norm_model.forward(ghidden)).reshape(x.shape)

    mlp_model = garnet.load_model(
        str(SCRIPT_DIR / "text_mlp_trt.x"),
        weights={
            mlp_prefix + ".gate_proj.weight": weights[mlp_prefix + ".gate_proj.weight"],
            mlp_prefix + ".up_proj.weight": weights[mlp_prefix + ".up_proj.weight"],
            mlp_prefix + ".down_proj.weight": weights[mlp_prefix + ".down_proj.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 2048]],
        subgraph="qwen3_text_mlp",
    )
    gmlp = to_numpy(mlp_model.forward(gpost)).reshape(x.shape[0], 2048)
    actual = ghidden + gmlp

    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-1)
    return {
        "input_shape": list(x.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
        "reference": "transformers.Qwen3VLTextDecoderLayer(layer_idx=0,eager)",
    }


def load_text_layer_weights_as_layer0_aliases(key_to_file, layer_idx):
    src_layer = f"language_model.layers.{layer_idx}"
    dst_layer = "language_model.layers.0"
    aliases = {}
    for suffix in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]:
        aliases[f"{dst_layer}.{suffix}"] = load_tensor(key_to_file, f"{src_layer}.{suffix}")
    return aliases


def load_text_layer_mixed_weights_as_layer0_aliases(garnet, key_to_file, layer_idx):
    src_layer = f"language_model.layers.{layer_idx}"
    dst_layer = "language_model.layers.0"
    aliases = {}
    for suffix in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]:
        source_key = f"{src_layer}.{suffix}"
        destination_key = f"{dst_layer}.{suffix}"
        if suffix in {"input_layernorm.weight", "post_attention_layernorm.weight"}:
            aliases[destination_key] = load_tensor(key_to_file, source_key)
        else:
            aliases[destination_key] = load_bfloat16_gpu_tensor(garnet, key_to_file, source_key)
    return aliases


def text_layer_count_from_weights(key_to_file):
    indices = []
    for key in key_to_file:
        match = re.match(r"language_model\.layers\.(\d+)\.input_layernorm\.weight$", key)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        raise AssertionError("no language_model.layers.* weights found")
    return max(indices) + 1


def run_garnet_text_decoder_layer(garnet, layer_weights, x, cos_np, sin_np, cache_dir):
    layer_prefix = "language_model.layers.0"
    attn_prefix = layer_prefix + ".self_attn"
    mlp_prefix = layer_prefix + ".mlp"

    input_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={layer_prefix + ".input_layernorm.weight": layer_weights[layer_prefix + ".input_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[list(x.shape)],
        subgraph="rms_norm",
    )
    gx_norm = to_numpy(input_norm_model.forward(x)).reshape(x.shape)

    qkv_model = garnet.load_model(
        str(SCRIPT_DIR / "text_qkv_head_norm_trt.x"),
        weights={
            attn_prefix + ".q_proj.weight": layer_weights[attn_prefix + ".q_proj.weight"],
            attn_prefix + ".k_proj.weight": layer_weights[attn_prefix + ".k_proj.weight"],
            attn_prefix + ".v_proj.weight": layer_weights[attn_prefix + ".v_proj.weight"],
            attn_prefix + ".q_norm.weight": layer_weights[attn_prefix + ".q_norm.weight"],
            attn_prefix + ".k_norm.weight": layer_weights[attn_prefix + ".k_norm.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[list(x.shape)],
        subgraph="text_qkv_head_norm",
    )
    gqkv = to_numpy(qkv_model.forward(gx_norm)).reshape(x.shape[0], 4096)

    rope_model = garnet.load_model(
        str(SCRIPT_DIR / "text_rope_apply_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 4096], [x.shape[0], 128], [x.shape[0], 128]],
        subgraph="text_rope_apply",
    )
    grope = to_numpy(rope_model.forward(gqkv, cos_np, sin_np)).reshape(x.shape[0], 4096)

    attention_model = garnet.load_model(
        str(SCRIPT_DIR / "text_attention_core_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 4096]],
        subgraph="text_attention_core",
    )
    gattn = to_numpy(attention_model.forward(grope)).reshape(x.shape[0], 2048)

    o_proj_model = garnet.load_model(
        str(SCRIPT_DIR / "text_o_proj_trt.x"),
        weights={attn_prefix + ".o_proj.weight": layer_weights[attn_prefix + ".o_proj.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 2048]],
        subgraph="text_o_proj",
    )
    go = to_numpy(o_proj_model.forward(gattn)).reshape(x.shape[0], 2048)
    ghidden = x + go

    post_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "text_post_rms_norm_trt.x"),
        weights={layer_prefix + ".post_attention_layernorm.weight": layer_weights[layer_prefix + ".post_attention_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 2048]],
        subgraph="text_post_attention_rms_norm",
    )
    gpost = to_numpy(post_norm_model.forward(ghidden)).reshape(x.shape)

    mlp_model = garnet.load_model(
        str(SCRIPT_DIR / "text_mlp_trt.x"),
        weights={
            mlp_prefix + ".gate_proj.weight": layer_weights[mlp_prefix + ".gate_proj.weight"],
            mlp_prefix + ".up_proj.weight": layer_weights[mlp_prefix + ".up_proj.weight"],
            mlp_prefix + ".down_proj.weight": layer_weights[mlp_prefix + ".down_proj.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[[x.shape[0], 2048]],
        subgraph="qwen3_text_mlp",
    )
    gmlp = to_numpy(mlp_model.forward(gpost)).reshape(x.shape[0], 2048)
    return ghidden + gmlp


def run_garnet_text_decoder_layer_cached_last_token_parity(garnet, key_to_file):
    import ctypes

    tokens = int(os.environ.get("GARNET_CACHED_DECODER_LAYER_TOKENS", "17"))
    page_size = int(os.environ.get("GARNET_CACHED_DECODER_LAYER_PAGE_SIZE", "8"))
    q_heads = 16
    kv_heads = 8
    head_dim = 128
    hidden_size = q_heads * head_dim
    qkv_size = hidden_size + (2 * kv_heads * head_dim)
    layer_weights = load_text_layer_weights_as_layer0_aliases(key_to_file, 0)
    rng = np.random.default_rng(20260716)
    x = rng.normal(0.0, 0.02, size=(tokens, hidden_size)).astype(np.float32)
    cos_np, sin_np = make_text_mrope(tokens)
    cache_dir = SCRIPT_DIR / "cache_cached_decoder_layer"

    full = run_garnet_text_decoder_layer(garnet, layer_weights, x, cos_np, sin_np, cache_dir)

    layer_prefix = "language_model.layers.0"
    attn_prefix = layer_prefix + ".self_attn"
    mlp_prefix = layer_prefix + ".mlp"

    input_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={layer_prefix + ".input_layernorm.weight": layer_weights[layer_prefix + ".input_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[list(x.shape)],
        subgraph="rms_norm",
    )
    gx_norm = to_numpy(input_norm_model.forward(x)).reshape(x.shape)

    qkv_model = garnet.load_model(
        str(SCRIPT_DIR / "text_qkv_head_norm_trt.x"),
        weights={
            attn_prefix + ".q_proj.weight": layer_weights[attn_prefix + ".q_proj.weight"],
            attn_prefix + ".k_proj.weight": layer_weights[attn_prefix + ".k_proj.weight"],
            attn_prefix + ".v_proj.weight": layer_weights[attn_prefix + ".v_proj.weight"],
            attn_prefix + ".q_norm.weight": layer_weights[attn_prefix + ".q_norm.weight"],
            attn_prefix + ".k_norm.weight": layer_weights[attn_prefix + ".k_norm.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[list(x.shape)],
        subgraph="text_qkv_head_norm",
    )
    gqkv = to_numpy(qkv_model.forward(gx_norm)).reshape(tokens, qkv_size)

    rope_model = garnet.load_model(
        str(SCRIPT_DIR / "text_rope_apply_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[tokens, qkv_size], [tokens, head_dim], [tokens, head_dim]],
        subgraph="text_rope_apply",
    )
    grope = to_numpy(rope_model.forward(gqkv, cos_np, sin_np)).reshape(tokens, qkv_size).astype(np.float32, copy=False)

    logical_pages = (tokens + page_size - 1) // page_size
    physical_pages = logical_pages + 2
    page_table = np.arange(logical_pages, dtype=np.int32)
    if logical_pages > 1:
        page_table = page_table[::-1].copy()
    key_pages = np.zeros((physical_pages, page_size, kv_heads, head_dim), dtype=np.float32)
    value_pages = np.zeros_like(key_pages)

    dll = configure_paged_kv_attention_ctypes(load_garnet_dll_ctypes())
    write_error = ctypes.create_string_buffer(512)
    rc = dll.GarnetRunTextPagedKVWriteFP32(
        grope.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        key_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        value_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        page_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tokens,
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

    q_last = np.ascontiguousarray(grope[-1, :hidden_size].reshape(q_heads, head_dim))
    gattn_last_heads = np.empty((q_heads, head_dim), dtype=np.float32)
    attn_error = ctypes.create_string_buffer(512)
    rc = dll.GarnetRunTextPagedKVCachedAttentionFP32(
        q_last.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        key_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        value_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        page_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        gattn_last_heads.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tokens,
        page_size,
        physical_pages,
        q_heads,
        kv_heads,
        head_dim,
        attn_error,
        len(attn_error),
    )
    if rc != 0:
        raise AssertionError(f"GarnetRunTextPagedKVCachedAttentionFP32 failed rc={rc}: {attn_error.value.decode(errors='ignore')}")
    gattn_last = gattn_last_heads.reshape(1, hidden_size)

    o_proj_model = garnet.load_model(
        str(SCRIPT_DIR / "text_o_proj_trt.x"),
        weights={attn_prefix + ".o_proj.weight": layer_weights[attn_prefix + ".o_proj.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="text_o_proj",
    )
    go_last = to_numpy(o_proj_model.forward(gattn_last)).reshape(1, hidden_size)
    ghidden_last = x[-1:] + go_last

    post_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "text_post_rms_norm_trt.x"),
        weights={layer_prefix + ".post_attention_layernorm.weight": layer_weights[layer_prefix + ".post_attention_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="text_post_attention_rms_norm",
    )
    gpost_last = to_numpy(post_norm_model.forward(ghidden_last)).reshape(1, hidden_size)

    mlp_model = garnet.load_model(
        str(SCRIPT_DIR / "text_mlp_trt.x"),
        weights={
            mlp_prefix + ".gate_proj.weight": layer_weights[mlp_prefix + ".gate_proj.weight"],
            mlp_prefix + ".up_proj.weight": layer_weights[mlp_prefix + ".up_proj.weight"],
            mlp_prefix + ".down_proj.weight": layer_weights[mlp_prefix + ".down_proj.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="qwen3_text_mlp",
    )
    gmlp_last = to_numpy(mlp_model.forward(gpost_last)).reshape(1, hidden_size)
    cached_last = ghidden_last + gmlp_last
    expected_last = full[-1:]
    max_error = float(np.max(np.abs(cached_last - expected_last)))
    mean_error = float(np.mean(np.abs(cached_last - expected_last)))
    np.testing.assert_allclose(cached_last, expected_last, rtol=2e-3, atol=2e-3)

    return {
        "tokens": tokens,
        "page_size": page_size,
        "logical_pages": int(logical_pages),
        "physical_pages": int(physical_pages),
        "page_table": [int(x) for x in page_table.tolist()],
        "qkv_shape": list(grope.shape),
        "key_pages_shape": list(key_pages.shape),
        "value_pages_shape": list(value_pages.shape),
        "last_token_output_shape": list(cached_last.shape),
        "max_error": max_error,
        "mean_error": mean_error,
    }


def run_garnet_text_decoder_layer_prefill_with_paged_kv(
    garnet,
    dll,
    layer_weights,
    x,
    cos_np,
    sin_np,
    cache_dir,
    page_size,
    physical_pages,
    page_table,
):
    import ctypes

    layer_prefix = "language_model.layers.0"
    attn_prefix = layer_prefix + ".self_attn"
    mlp_prefix = layer_prefix + ".mlp"
    tokens = int(x.shape[0])
    q_heads = 16
    kv_heads = 8
    head_dim = 128
    hidden_size = 2048
    qkv_size = 4096

    input_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={layer_prefix + ".input_layernorm.weight": layer_weights[layer_prefix + ".input_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[list(x.shape)],
        subgraph="rms_norm",
    )
    gx_norm = to_numpy(input_norm_model.forward(x)).reshape(x.shape)

    qkv_model = garnet.load_model(
        str(SCRIPT_DIR / "text_qkv_head_norm_trt.x"),
        weights={
            attn_prefix + ".q_proj.weight": layer_weights[attn_prefix + ".q_proj.weight"],
            attn_prefix + ".k_proj.weight": layer_weights[attn_prefix + ".k_proj.weight"],
            attn_prefix + ".v_proj.weight": layer_weights[attn_prefix + ".v_proj.weight"],
            attn_prefix + ".q_norm.weight": layer_weights[attn_prefix + ".q_norm.weight"],
            attn_prefix + ".k_norm.weight": layer_weights[attn_prefix + ".k_norm.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[list(x.shape)],
        subgraph="text_qkv_head_norm",
    )
    gqkv = to_numpy(qkv_model.forward(gx_norm)).reshape(tokens, qkv_size)

    rope_model = garnet.load_model(
        str(SCRIPT_DIR / "text_rope_apply_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[tokens, qkv_size], [tokens, head_dim], [tokens, head_dim]],
        subgraph="text_rope_apply",
    )
    grope = to_numpy(rope_model.forward(gqkv, cos_np, sin_np)).reshape(tokens, qkv_size).astype(np.float32, copy=False)

    key_pages = np.zeros((physical_pages, page_size, kv_heads, head_dim), dtype=np.float32)
    value_pages = np.zeros_like(key_pages)
    write_error = ctypes.create_string_buffer(512)
    rc = dll.GarnetRunTextPagedKVWriteFP32(
        grope.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        key_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        value_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        page_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        tokens,
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
        raise AssertionError(f"prefill GarnetRunTextPagedKVWriteFP32 failed rc={rc}: {write_error.value.decode(errors='ignore')}")

    attention_model = garnet.load_model(
        str(SCRIPT_DIR / "text_attention_core_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[tokens, qkv_size]],
        subgraph="text_attention_core",
    )
    gattn = to_numpy(attention_model.forward(grope)).reshape(tokens, hidden_size)

    o_proj_model = garnet.load_model(
        str(SCRIPT_DIR / "text_o_proj_trt.x"),
        weights={attn_prefix + ".o_proj.weight": layer_weights[attn_prefix + ".o_proj.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[tokens, hidden_size]],
        subgraph="text_o_proj",
    )
    go = to_numpy(o_proj_model.forward(gattn)).reshape(tokens, hidden_size)
    ghidden = x + go

    post_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "text_post_rms_norm_trt.x"),
        weights={layer_prefix + ".post_attention_layernorm.weight": layer_weights[layer_prefix + ".post_attention_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[tokens, hidden_size]],
        subgraph="text_post_attention_rms_norm",
    )
    gpost = to_numpy(post_norm_model.forward(ghidden)).reshape(tokens, hidden_size)

    mlp_model = garnet.load_model(
        str(SCRIPT_DIR / "text_mlp_trt.x"),
        weights={
            mlp_prefix + ".gate_proj.weight": layer_weights[mlp_prefix + ".gate_proj.weight"],
            mlp_prefix + ".up_proj.weight": layer_weights[mlp_prefix + ".up_proj.weight"],
            mlp_prefix + ".down_proj.weight": layer_weights[mlp_prefix + ".down_proj.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[[tokens, hidden_size]],
        subgraph="qwen3_text_mlp",
    )
    gmlp = to_numpy(mlp_model.forward(gpost)).reshape(tokens, hidden_size)
    return ghidden + gmlp, key_pages, value_pages


def run_garnet_text_decoder_layer_decode_with_paged_kv(
    garnet,
    dll,
    layer_weights,
    x,
    cos_np,
    sin_np,
    cache_dir,
    key_pages,
    value_pages,
    page_size,
    physical_pages,
    page_table,
    token_offset,
):
    import ctypes

    layer_prefix = "language_model.layers.0"
    attn_prefix = layer_prefix + ".self_attn"
    mlp_prefix = layer_prefix + ".mlp"
    q_heads = 16
    kv_heads = 8
    head_dim = 128
    hidden_size = 2048
    qkv_size = 4096
    sequence_length = int(token_offset) + 1

    input_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={layer_prefix + ".input_layernorm.weight": layer_weights[layer_prefix + ".input_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="rms_norm",
    )
    gx_norm = to_numpy(input_norm_model.forward(x)).reshape(1, hidden_size)

    qkv_model = garnet.load_model(
        str(SCRIPT_DIR / "text_qkv_head_norm_trt.x"),
        weights={
            attn_prefix + ".q_proj.weight": layer_weights[attn_prefix + ".q_proj.weight"],
            attn_prefix + ".k_proj.weight": layer_weights[attn_prefix + ".k_proj.weight"],
            attn_prefix + ".v_proj.weight": layer_weights[attn_prefix + ".v_proj.weight"],
            attn_prefix + ".q_norm.weight": layer_weights[attn_prefix + ".q_norm.weight"],
            attn_prefix + ".k_norm.weight": layer_weights[attn_prefix + ".k_norm.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="text_qkv_head_norm",
    )
    gqkv = to_numpy(qkv_model.forward(gx_norm)).reshape(1, qkv_size)

    rope_model = garnet.load_model(
        str(SCRIPT_DIR / "text_rope_apply_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[1, qkv_size], [1, head_dim], [1, head_dim]],
        subgraph="text_rope_apply",
    )
    grope = to_numpy(rope_model.forward(gqkv, cos_np, sin_np)).reshape(1, qkv_size).astype(np.float32, copy=False)

    write_error = ctypes.create_string_buffer(512)
    rc = dll.GarnetRunTextPagedKVWriteFP32(
        grope.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        key_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        value_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        page_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        1,
        int(token_offset),
        page_size,
        physical_pages,
        q_heads,
        kv_heads,
        head_dim,
        write_error,
        len(write_error),
    )
    if rc != 0:
        raise AssertionError(f"decode GarnetRunTextPagedKVWriteFP32 failed rc={rc}: {write_error.value.decode(errors='ignore')}")

    q = np.ascontiguousarray(grope[0, :hidden_size].reshape(q_heads, head_dim))
    attn_heads = np.empty((q_heads, head_dim), dtype=np.float32)
    attn_error = ctypes.create_string_buffer(512)
    rc = dll.GarnetRunTextPagedKVCachedAttentionFP32(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        key_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        value_pages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        page_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        attn_heads.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sequence_length,
        page_size,
        physical_pages,
        q_heads,
        kv_heads,
        head_dim,
        attn_error,
        len(attn_error),
    )
    if rc != 0:
        raise AssertionError(f"GarnetRunTextPagedKVCachedAttentionFP32 failed rc={rc}: {attn_error.value.decode(errors='ignore')}")
    gattn = attn_heads.reshape(1, hidden_size)

    o_proj_model = garnet.load_model(
        str(SCRIPT_DIR / "text_o_proj_trt.x"),
        weights={attn_prefix + ".o_proj.weight": layer_weights[attn_prefix + ".o_proj.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="text_o_proj",
    )
    go = to_numpy(o_proj_model.forward(gattn)).reshape(1, hidden_size)
    ghidden = x + go

    post_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "text_post_rms_norm_trt.x"),
        weights={layer_prefix + ".post_attention_layernorm.weight": layer_weights[layer_prefix + ".post_attention_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="text_post_attention_rms_norm",
    )
    gpost = to_numpy(post_norm_model.forward(ghidden)).reshape(1, hidden_size)

    mlp_model = garnet.load_model(
        str(SCRIPT_DIR / "text_mlp_trt.x"),
        weights={
            mlp_prefix + ".gate_proj.weight": layer_weights[mlp_prefix + ".gate_proj.weight"],
            mlp_prefix + ".up_proj.weight": layer_weights[mlp_prefix + ".up_proj.weight"],
            mlp_prefix + ".down_proj.weight": layer_weights[mlp_prefix + ".down_proj.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="qwen3_text_mlp",
    )
    gmlp = to_numpy(mlp_model.forward(gpost)).reshape(1, hidden_size)
    return ghidden + gmlp


def create_device_paged_kv_handle(dll, physical_pages, page_size, logical_pages, q_heads, kv_heads, head_dim, page_table):
    import ctypes

    handle = ctypes.c_longlong(0)
    error = ctypes.create_string_buffer(512)
    rc = dll.GarnetCreateDevicePagedKVFP32(
        int(physical_pages),
        int(page_size),
        int(logical_pages),
        int(q_heads),
        int(kv_heads),
        int(head_dim),
        page_table.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.byref(handle),
        error,
        len(error),
    )
    if rc != 0:
        raise AssertionError(f"GarnetCreateDevicePagedKVFP32 failed rc={rc}: {error.value.decode(errors='ignore')}")
    return int(handle.value)


def destroy_device_paged_kv_handle(dll, handle):
    import ctypes

    error = ctypes.create_string_buffer(512)
    rc = dll.GarnetDestroyDevicePagedKVFP32(int(handle), error, len(error))
    if rc != 0:
        raise AssertionError(f"GarnetDestroyDevicePagedKVFP32 failed rc={rc}: {error.value.decode(errors='ignore')}")


def device_paged_kv_write(dll, handle, qkv, token_count, start_position):
    import ctypes

    error = ctypes.create_string_buffer(512)
    qkv = np.ascontiguousarray(qkv.astype(np.float32, copy=False))
    rc = dll.GarnetDevicePagedKVWriteFP32(
        int(handle),
        qkv.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(token_count),
        int(start_position),
        error,
        len(error),
    )
    if rc != 0:
        raise AssertionError(f"GarnetDevicePagedKVWriteFP32 failed rc={rc}: {error.value.decode(errors='ignore')}")


def device_paged_kv_attention(dll, handle, q, sequence_length):
    import ctypes

    q = np.ascontiguousarray(q.astype(np.float32, copy=False))
    output = np.empty_like(q)
    error = ctypes.create_string_buffer(512)
    rc = dll.GarnetDevicePagedKVAttentionFP32(
        int(handle),
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(sequence_length),
        error,
        len(error),
    )
    if rc != 0:
        raise AssertionError(f"GarnetDevicePagedKVAttentionFP32 failed rc={rc}: {error.value.decode(errors='ignore')}")
    return output


def run_garnet_text_decoder_layer_prefill_with_device_paged_kv(
    garnet,
    dll,
    layer_weights,
    x,
    cos_np,
    sin_np,
    cache_dir,
    device_kv_handle,
):
    layer_prefix = "language_model.layers.0"
    attn_prefix = layer_prefix + ".self_attn"
    mlp_prefix = layer_prefix + ".mlp"
    tokens = int(x.shape[0])
    head_dim = 128
    hidden_size = 2048
    qkv_size = 4096

    input_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={layer_prefix + ".input_layernorm.weight": layer_weights[layer_prefix + ".input_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[list(x.shape)],
        subgraph="rms_norm",
    )
    gx_norm = to_numpy(input_norm_model.forward(x)).reshape(x.shape)

    qkv_model = garnet.load_model(
        str(SCRIPT_DIR / "text_qkv_head_norm_trt.x"),
        weights={
            attn_prefix + ".q_proj.weight": layer_weights[attn_prefix + ".q_proj.weight"],
            attn_prefix + ".k_proj.weight": layer_weights[attn_prefix + ".k_proj.weight"],
            attn_prefix + ".v_proj.weight": layer_weights[attn_prefix + ".v_proj.weight"],
            attn_prefix + ".q_norm.weight": layer_weights[attn_prefix + ".q_norm.weight"],
            attn_prefix + ".k_norm.weight": layer_weights[attn_prefix + ".k_norm.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[list(x.shape)],
        subgraph="text_qkv_head_norm",
    )
    gqkv = to_numpy(qkv_model.forward(gx_norm)).reshape(tokens, qkv_size)

    rope_model = garnet.load_model(
        str(SCRIPT_DIR / "text_rope_apply_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[tokens, qkv_size], [tokens, head_dim], [tokens, head_dim]],
        subgraph="text_rope_apply",
    )
    grope = to_numpy(rope_model.forward(gqkv, cos_np, sin_np)).reshape(tokens, qkv_size).astype(np.float32, copy=False)
    device_paged_kv_write(dll, device_kv_handle, grope, tokens, 0)

    attention_model = garnet.load_model(
        str(SCRIPT_DIR / "text_attention_core_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[tokens, qkv_size]],
        subgraph="text_attention_core",
    )
    gattn = to_numpy(attention_model.forward(grope)).reshape(tokens, hidden_size)

    o_proj_model = garnet.load_model(
        str(SCRIPT_DIR / "text_o_proj_trt.x"),
        weights={attn_prefix + ".o_proj.weight": layer_weights[attn_prefix + ".o_proj.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[tokens, hidden_size]],
        subgraph="text_o_proj",
    )
    go = to_numpy(o_proj_model.forward(gattn)).reshape(tokens, hidden_size)
    ghidden = x + go

    post_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "text_post_rms_norm_trt.x"),
        weights={layer_prefix + ".post_attention_layernorm.weight": layer_weights[layer_prefix + ".post_attention_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[tokens, hidden_size]],
        subgraph="text_post_attention_rms_norm",
    )
    gpost = to_numpy(post_norm_model.forward(ghidden)).reshape(tokens, hidden_size)

    mlp_model = garnet.load_model(
        str(SCRIPT_DIR / "text_mlp_trt.x"),
        weights={
            mlp_prefix + ".gate_proj.weight": layer_weights[mlp_prefix + ".gate_proj.weight"],
            mlp_prefix + ".up_proj.weight": layer_weights[mlp_prefix + ".up_proj.weight"],
            mlp_prefix + ".down_proj.weight": layer_weights[mlp_prefix + ".down_proj.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[[tokens, hidden_size]],
        subgraph="qwen3_text_mlp",
    )
    gmlp = to_numpy(mlp_model.forward(gpost)).reshape(tokens, hidden_size)
    return ghidden + gmlp


def run_garnet_text_decoder_layer_decode_with_device_paged_kv(
    garnet,
    dll,
    layer_weights,
    x,
    cos_np,
    sin_np,
    cache_dir,
    device_kv_handle,
    token_offset,
):
    layer_prefix = "language_model.layers.0"
    attn_prefix = layer_prefix + ".self_attn"
    mlp_prefix = layer_prefix + ".mlp"
    q_heads = 16
    head_dim = 128
    hidden_size = 2048
    qkv_size = 4096
    sequence_length = int(token_offset) + 1

    input_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={layer_prefix + ".input_layernorm.weight": layer_weights[layer_prefix + ".input_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="rms_norm",
    )
    gx_norm = to_numpy(input_norm_model.forward(x)).reshape(1, hidden_size)

    qkv_model = garnet.load_model(
        str(SCRIPT_DIR / "text_qkv_head_norm_trt.x"),
        weights={
            attn_prefix + ".q_proj.weight": layer_weights[attn_prefix + ".q_proj.weight"],
            attn_prefix + ".k_proj.weight": layer_weights[attn_prefix + ".k_proj.weight"],
            attn_prefix + ".v_proj.weight": layer_weights[attn_prefix + ".v_proj.weight"],
            attn_prefix + ".q_norm.weight": layer_weights[attn_prefix + ".q_norm.weight"],
            attn_prefix + ".k_norm.weight": layer_weights[attn_prefix + ".k_norm.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="text_qkv_head_norm",
    )
    gqkv = to_numpy(qkv_model.forward(gx_norm)).reshape(1, qkv_size)

    rope_model = garnet.load_model(
        str(SCRIPT_DIR / "text_rope_apply_trt.x"),
        weights={},
        cache_dir=str(cache_dir),
        input_shapes=[[1, qkv_size], [1, head_dim], [1, head_dim]],
        subgraph="text_rope_apply",
    )
    grope = to_numpy(rope_model.forward(gqkv, cos_np, sin_np)).reshape(1, qkv_size).astype(np.float32, copy=False)
    device_paged_kv_write(dll, device_kv_handle, grope, 1, int(token_offset))

    q = np.ascontiguousarray(grope[0, :hidden_size].reshape(q_heads, head_dim))
    attn_heads = device_paged_kv_attention(dll, device_kv_handle, q, sequence_length)
    gattn = attn_heads.reshape(1, hidden_size)

    o_proj_model = garnet.load_model(
        str(SCRIPT_DIR / "text_o_proj_trt.x"),
        weights={attn_prefix + ".o_proj.weight": layer_weights[attn_prefix + ".o_proj.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="text_o_proj",
    )
    go = to_numpy(o_proj_model.forward(gattn)).reshape(1, hidden_size)
    ghidden = x + go

    post_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "text_post_rms_norm_trt.x"),
        weights={layer_prefix + ".post_attention_layernorm.weight": layer_weights[layer_prefix + ".post_attention_layernorm.weight"]},
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="text_post_attention_rms_norm",
    )
    gpost = to_numpy(post_norm_model.forward(ghidden)).reshape(1, hidden_size)

    mlp_model = garnet.load_model(
        str(SCRIPT_DIR / "text_mlp_trt.x"),
        weights={
            mlp_prefix + ".gate_proj.weight": layer_weights[mlp_prefix + ".gate_proj.weight"],
            mlp_prefix + ".up_proj.weight": layer_weights[mlp_prefix + ".up_proj.weight"],
            mlp_prefix + ".down_proj.weight": layer_weights[mlp_prefix + ".down_proj.weight"],
        },
        cache_dir=str(cache_dir),
        input_shapes=[[1, hidden_size]],
        subgraph="qwen3_text_mlp",
    )
    gmlp = to_numpy(mlp_model.forward(gpost)).reshape(1, hidden_size)
    return ghidden + gmlp


def run_garnet_text_prefill_decode_cached_parity(garnet, key_to_file):
    prompt_tokens = int(os.environ.get("GARNET_CACHED_DECODE_PROMPT_TOKENS", "11"))
    page_size = int(os.environ.get("GARNET_CACHED_DECODE_PAGE_SIZE", "8"))
    layer_count_env = os.environ.get("GARNET_CACHED_DECODE_LAYER_COUNT", "2").strip().lower()
    available_layers = text_layer_count_from_weights(key_to_file)
    layer_count = available_layers if layer_count_env in {"all", "full"} else int(layer_count_env)
    layer_count = max(1, min(layer_count, available_layers))
    total_tokens = prompt_tokens + 1
    logical_pages = (total_tokens + page_size - 1) // page_size
    physical_pages = logical_pages + 2
    page_table = np.arange(logical_pages, dtype=np.int32)
    if logical_pages > 1:
        page_table = np.roll(page_table, 1).astype(np.int32)

    embed_tokens = load_tensor(key_to_file, "language_model.embed_tokens.weight")
    final_norm = load_tensor(key_to_file, "language_model.norm.weight")
    prompt_ids = np.arange(1000, 1000 + prompt_tokens, dtype=np.int64)
    next_id = np.asarray([1000 + prompt_tokens], dtype=np.int64)
    input_ids_full = np.concatenate([prompt_ids, next_id])
    cos_full, sin_full = make_text_mrope(total_tokens)
    cos_prompt = cos_full[:prompt_tokens]
    sin_prompt = sin_full[:prompt_tokens]
    cos_decode = cos_full[prompt_tokens:prompt_tokens + 1]
    sin_decode = sin_full[prompt_tokens:prompt_tokens + 1]
    cache_dir = SCRIPT_DIR / "cache_cached_decode_layers" / f"layers_{layer_count}_tokens_{prompt_tokens}"
    dll = configure_paged_kv_attention_ctypes(load_garnet_dll_ctypes())

    full_hidden = embed_tokens[input_ids_full].astype(np.float32)
    for layer_idx in range(layer_count):
        layer_weights = load_text_layer_weights_as_layer0_aliases(key_to_file, layer_idx)
        full_hidden = run_garnet_text_decoder_layer(garnet, layer_weights, full_hidden, cos_full, sin_full, cache_dir / "full" / f"layer_{layer_idx}")

    prompt_hidden = embed_tokens[prompt_ids].astype(np.float32)
    layer_caches = []
    for layer_idx in range(layer_count):
        layer_weights = load_text_layer_weights_as_layer0_aliases(key_to_file, layer_idx)
        prompt_hidden, key_pages, value_pages = run_garnet_text_decoder_layer_prefill_with_paged_kv(
            garnet,
            dll,
            layer_weights,
            prompt_hidden,
            cos_prompt,
            sin_prompt,
            cache_dir / "prefill" / f"layer_{layer_idx}",
            page_size,
            physical_pages,
            page_table,
        )
        layer_caches.append((key_pages, value_pages))

    decode_hidden = embed_tokens[next_id].astype(np.float32)
    for layer_idx in range(layer_count):
        layer_weights = load_text_layer_weights_as_layer0_aliases(key_to_file, layer_idx)
        key_pages, value_pages = layer_caches[layer_idx]
        decode_hidden = run_garnet_text_decoder_layer_decode_with_paged_kv(
            garnet,
            dll,
            layer_weights,
            decode_hidden,
            cos_decode,
            sin_decode,
            cache_dir / "decode" / f"layer_{layer_idx}",
            key_pages,
            value_pages,
            page_size,
            physical_pages,
            page_table,
            prompt_tokens,
        )

    final_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={"language_model.layers.0.input_layernorm.weight": final_norm},
        cache_dir=str(cache_dir / "norm"),
        input_shapes=[[1, 2048]],
        subgraph="rms_norm",
    )
    expected_norm = to_numpy(final_norm_model.forward(full_hidden[-1:])).reshape(1, 2048)
    actual_norm = to_numpy(final_norm_model.forward(decode_hidden)).reshape(1, 2048)

    lm_head_model = garnet.load_model(
        str(SCRIPT_DIR / "text_lm_head_trt.x"),
        weights={"language_model.embed_tokens.weight": embed_tokens},
        cache_dir=str(cache_dir / "lm_head"),
        input_shapes=[[1, 2048]],
        subgraph="text_lm_head",
    )
    expected_logits = to_numpy(lm_head_model.forward(expected_norm)).reshape(1, embed_tokens.shape[0])
    actual_logits = to_numpy(lm_head_model.forward(actual_norm)).reshape(1, embed_tokens.shape[0])
    hidden_max_error = float(np.max(np.abs(decode_hidden - full_hidden[-1:])))
    hidden_mean_error = float(np.mean(np.abs(decode_hidden - full_hidden[-1:])))
    logits_max_error = float(np.max(np.abs(actual_logits - expected_logits)))
    logits_mean_error = float(np.mean(np.abs(actual_logits - expected_logits)))
    expected_top1 = int(np.argmax(expected_logits[-1]))
    actual_top1 = int(np.argmax(actual_logits[-1]))
    np.testing.assert_allclose(decode_hidden, full_hidden[-1:], rtol=5e-3, atol=5e-3)

    return {
        "prompt_tokens": int(prompt_tokens),
        "total_tokens_after_decode": int(total_tokens),
        "layer_count": int(layer_count),
        "available_layers": int(available_layers),
        "page_size": int(page_size),
        "logical_pages": int(logical_pages),
        "physical_pages": int(physical_pages),
        "page_table": [int(x) for x in page_table.tolist()],
        "hidden_max_error": hidden_max_error,
        "hidden_mean_error": hidden_mean_error,
        "logits_max_error": logits_max_error,
        "logits_mean_error": logits_mean_error,
        "expected_top1": expected_top1,
        "actual_top1": actual_top1,
        "top1_match": expected_top1 == actual_top1,
    }


def run_garnet_text_prefill_decode_device_kv_parity(garnet, key_to_file):
    prompt_tokens = int(os.environ.get("GARNET_DEVICE_CACHED_DECODE_PROMPT_TOKENS", "11"))
    page_size = int(os.environ.get("GARNET_DEVICE_CACHED_DECODE_PAGE_SIZE", "8"))
    layer_count_env = os.environ.get("GARNET_DEVICE_CACHED_DECODE_LAYER_COUNT", "2").strip().lower()
    available_layers = text_layer_count_from_weights(key_to_file)
    layer_count = available_layers if layer_count_env in {"all", "full"} else int(layer_count_env)
    layer_count = max(1, min(layer_count, available_layers))
    total_tokens = prompt_tokens + 1
    logical_pages = (total_tokens + page_size - 1) // page_size
    physical_pages = logical_pages + 2
    page_table = np.arange(logical_pages, dtype=np.int32)
    if logical_pages > 1:
        page_table = np.roll(page_table, 1).astype(np.int32)

    embed_tokens = load_tensor(key_to_file, "language_model.embed_tokens.weight")
    final_norm = load_tensor(key_to_file, "language_model.norm.weight")
    prompt_ids = np.arange(1000, 1000 + prompt_tokens, dtype=np.int64)
    next_id = np.asarray([1000 + prompt_tokens], dtype=np.int64)
    input_ids_full = np.concatenate([prompt_ids, next_id])
    cos_full, sin_full = make_text_mrope(total_tokens)
    cos_prompt = cos_full[:prompt_tokens]
    sin_prompt = sin_full[:prompt_tokens]
    cos_decode = cos_full[prompt_tokens:prompt_tokens + 1]
    sin_decode = sin_full[prompt_tokens:prompt_tokens + 1]
    cache_dir = SCRIPT_DIR / "cache_device_cached_decode_layers" / f"layers_{layer_count}_tokens_{prompt_tokens}"
    dll = configure_paged_kv_attention_ctypes(load_garnet_dll_ctypes())

    full_hidden = embed_tokens[input_ids_full].astype(np.float32)
    for layer_idx in range(layer_count):
        layer_weights = load_text_layer_weights_as_layer0_aliases(key_to_file, layer_idx)
        full_hidden = run_garnet_text_decoder_layer(garnet, layer_weights, full_hidden, cos_full, sin_full, cache_dir / "full" / f"layer_{layer_idx}")

    device_handles = []
    prompt_hidden = embed_tokens[prompt_ids].astype(np.float32)
    try:
        for layer_idx in range(layer_count):
            handle = create_device_paged_kv_handle(dll, physical_pages, page_size, logical_pages, 16, 8, 128, page_table)
            device_handles.append(handle)
            layer_weights = load_text_layer_weights_as_layer0_aliases(key_to_file, layer_idx)
            prompt_hidden = run_garnet_text_decoder_layer_prefill_with_device_paged_kv(
                garnet,
                dll,
                layer_weights,
                prompt_hidden,
                cos_prompt,
                sin_prompt,
                cache_dir / "prefill" / f"layer_{layer_idx}",
                handle,
            )

        decode_hidden = embed_tokens[next_id].astype(np.float32)
        for layer_idx in range(layer_count):
            layer_weights = load_text_layer_weights_as_layer0_aliases(key_to_file, layer_idx)
            decode_hidden = run_garnet_text_decoder_layer_decode_with_device_paged_kv(
                garnet,
                dll,
                layer_weights,
                decode_hidden,
                cos_decode,
                sin_decode,
                cache_dir / "decode" / f"layer_{layer_idx}",
                device_handles[layer_idx],
                prompt_tokens,
            )
    finally:
        for handle in device_handles:
            destroy_device_paged_kv_handle(dll, handle)

    final_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={"language_model.layers.0.input_layernorm.weight": final_norm},
        cache_dir=str(cache_dir / "norm"),
        input_shapes=[[1, 2048]],
        subgraph="rms_norm",
    )
    expected_norm = to_numpy(final_norm_model.forward(full_hidden[-1:])).reshape(1, 2048)
    actual_norm = to_numpy(final_norm_model.forward(decode_hidden)).reshape(1, 2048)

    lm_head_model = garnet.load_model(
        str(SCRIPT_DIR / "text_lm_head_trt.x"),
        weights={"language_model.embed_tokens.weight": embed_tokens},
        cache_dir=str(cache_dir / "lm_head"),
        input_shapes=[[1, 2048]],
        subgraph="text_lm_head",
    )
    expected_logits = to_numpy(lm_head_model.forward(expected_norm)).reshape(1, embed_tokens.shape[0])
    actual_logits = to_numpy(lm_head_model.forward(actual_norm)).reshape(1, embed_tokens.shape[0])
    hidden_max_error = float(np.max(np.abs(decode_hidden - full_hidden[-1:])))
    hidden_mean_error = float(np.mean(np.abs(decode_hidden - full_hidden[-1:])))
    logits_max_error = float(np.max(np.abs(actual_logits - expected_logits)))
    logits_mean_error = float(np.mean(np.abs(actual_logits - expected_logits)))
    expected_top1 = int(np.argmax(expected_logits[-1]))
    actual_top1 = int(np.argmax(actual_logits[-1]))
    np.testing.assert_allclose(decode_hidden, full_hidden[-1:], rtol=5e-3, atol=5e-3)

    return {
        "prompt_tokens": int(prompt_tokens),
        "total_tokens_after_decode": int(total_tokens),
        "layer_count": int(layer_count),
        "available_layers": int(available_layers),
        "page_size": int(page_size),
        "logical_pages": int(logical_pages),
        "physical_pages": int(physical_pages),
        "page_table": [int(x) for x in page_table.tolist()],
        "device_kv": True,
        "hidden_max_error": hidden_max_error,
        "hidden_mean_error": hidden_mean_error,
        "logits_max_error": logits_max_error,
        "logits_mean_error": logits_mean_error,
        "expected_top1": expected_top1,
        "actual_top1": actual_top1,
        "top1_match": expected_top1 == actual_top1,
    }


def make_processor_window_embeddings(garnet, model_dir, key_to_file, token_count, use_all_vision):
    processor_npz = latest_processor_npz()
    arrays = np.load(processor_npz)
    if "input_ids" not in arrays or "mm_token_type_ids" not in arrays:
        raise AssertionError(f"processor dump missing input_ids/mm_token_type_ids: {processor_npz}")
    input_ids = arrays["input_ids"][0, :token_count].astype(np.int64)
    mm_types = arrays["mm_token_type_ids"][0, :token_count].astype(np.int64)
    visual_positions = np.flatnonzero(mm_types == 1)
    embed_tokens = load_tensor(key_to_file, "language_model.embed_tokens.weight")
    expected_x = embed_tokens[input_ids].astype(np.float32)
    actual_x = expected_x.copy()
    visual_npz = processor_npz
    grid_thw = None
    vision_block_count = 0
    visual_token_max_error = 0.0
    visual_token_mean_error = 0.0

    if visual_positions.size:
        if use_all_vision:
            expected_visual_tokens, actual_visual_tokens, visual_npz, grid_thw, vision_block_count = vision_all_blocks_merger_tokens_pair(
                garnet, model_dir, key_to_file, int(visual_positions.size)
            )
        else:
            expected_visual_tokens, visual_npz, grid_thw = vision_patch_pos_merger_tokens_numpy(key_to_file, int(visual_positions.size))
            actual_visual_tokens = expected_visual_tokens
        expected_x[visual_positions] = expected_visual_tokens
        actual_x[visual_positions] = actual_visual_tokens
        visual_token_max_error = float(np.max(np.abs(actual_visual_tokens - expected_visual_tokens)))
        visual_token_mean_error = float(np.mean(np.abs(actual_visual_tokens - expected_visual_tokens)))

    return {
        "processor_npz": processor_npz,
        "visual_npz": visual_npz,
        "grid_thw": grid_thw,
        "input_ids": input_ids,
        "visual_positions": visual_positions,
        "embed_tokens": embed_tokens,
        "expected_x": expected_x,
        "actual_x": actual_x,
        "vision_block_count": vision_block_count,
        "visual_token_max_error": visual_token_max_error,
        "visual_token_mean_error": visual_token_mean_error,
    }


def run_two_text_decoder_layers_against_hf_modules(garnet, model_dir, key_to_file):
    torch, config, layer0, rotary, create_causal_mask = load_hf_text_layer(model_dir, key_to_file, 0)
    _, _, layer1, _, _ = load_hf_text_layer(model_dir, key_to_file, 1)
    rng = np.random.default_rng(20260717)
    x = rng.normal(0.0, 0.02, size=(4, config.hidden_size)).astype(np.float32)

    with torch.inference_mode():
        hx = torch.from_numpy(x).unsqueeze(0)
        position_ids = torch.stack([
            torch.arange(x.shape[0], dtype=torch.long),
            torch.arange(x.shape[0], dtype=torch.long) + 17,
            torch.arange(x.shape[0], dtype=torch.long) + 31,
        ], dim=0).unsqueeze(1)
        text_position_ids = torch.arange(x.shape[0], dtype=torch.long).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=config,
            inputs_embeds=hx,
            attention_mask=torch.ones(1, x.shape[0], dtype=torch.long),
            past_key_values=None,
            position_ids=text_position_ids,
        )
        cos, sin = rotary(hx, position_ids)
        expected = layer0(
            hx,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        )
        expected = layer1(
            expected,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        ).squeeze(0).detach().cpu().numpy().astype(np.float32)
        cos_np = cos.squeeze(0).detach().cpu().numpy().astype(np.float32)
        sin_np = sin.squeeze(0).detach().cpu().numpy().astype(np.float32)

    actual = x
    cache_dir = SCRIPT_DIR / "cache_two_decoder_layers_hf"
    for layer_idx in range(2):
        layer_weights = load_text_layer_weights_as_layer0_aliases(key_to_file, layer_idx)
        actual = run_garnet_text_decoder_layer(garnet, layer_weights, actual, cos_np, sin_np, cache_dir)

    np.testing.assert_allclose(actual, expected, rtol=5e-2, atol=5e-1)
    return {
        "layers": 2,
        "input_shape": list(x.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
        "reference": "two transformers.Qwen3VLTextDecoderLayer modules, eager",
    }


def run_full_prompt_window_logits_against_hf_modules(garnet, model_dir, key_to_file):
    torch, config, _, rotary, create_causal_mask = load_hf_text_layer(model_dir, key_to_file, 0)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception:
        tokenizer = None
    token_count = int(os.environ.get("GARNET_QWEN_FULL_TEXT_TOKEN_WINDOW", "8"))
    use_all_vision = env_flag("GARNET_QWEN_FULL_TEXT_USE_ALL_VISION_BLOCKS")
    window = make_processor_window_embeddings(garnet, model_dir, key_to_file, token_count, use_all_vision)
    expected_x = window["expected_x"]
    actual = window["actual_x"]
    embed_tokens = window["embed_tokens"]
    final_norm = load_tensor(key_to_file, "language_model.norm.weight")
    layer_count = text_layer_count_from_weights(key_to_file)

    with torch.inference_mode():
        hidden = torch.from_numpy(expected_x).unsqueeze(0)
        position_ids = torch.stack([
            torch.arange(expected_x.shape[0], dtype=torch.long),
            torch.arange(expected_x.shape[0], dtype=torch.long) + 17,
            torch.arange(expected_x.shape[0], dtype=torch.long) + 31,
        ], dim=0).unsqueeze(1)
        text_position_ids = torch.arange(expected_x.shape[0], dtype=torch.long).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=config,
            inputs_embeds=hidden,
            attention_mask=torch.ones(1, expected_x.shape[0], dtype=torch.long),
            past_key_values=None,
            position_ids=text_position_ids,
        )
        cos, sin = rotary(hidden, position_ids)
        cos_np = cos.squeeze(0).detach().cpu().numpy().astype(np.float32)
        sin_np = sin.squeeze(0).detach().cpu().numpy().astype(np.float32)
        for layer_idx in range(layer_count):
            _, _, layer, _, _ = load_hf_text_layer(model_dir, key_to_file, layer_idx)
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=(cos, sin),
            )
            del layer
        expected_hidden = hidden.squeeze(0).detach().cpu().numpy().astype(np.float32)

    expected_norm = expected_hidden / np.sqrt(np.mean(np.square(expected_hidden), axis=-1, keepdims=True) + 1.0e-6)
    expected_norm = expected_norm * final_norm
    expected_logits = expected_norm @ embed_tokens.T

    cache_base = "cache_full_prompt_all_vision_logits_hf" if use_all_vision else "cache_full_prompt_logits_hf"
    cache_dir = SCRIPT_DIR / cache_base / f"tokens_{token_count}"
    for layer_idx in range(layer_count):
        layer_weights = load_text_layer_weights_as_layer0_aliases(key_to_file, layer_idx)
        actual = run_garnet_text_decoder_layer(garnet, layer_weights, actual, cos_np, sin_np, cache_dir)

    final_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={"language_model.layers.0.input_layernorm.weight": final_norm},
        cache_dir=str(cache_dir),
        input_shapes=[list(actual.shape)],
        subgraph="rms_norm",
    )
    actual_norm = to_numpy(final_norm_model.forward(actual)).reshape(actual.shape)
    lm_head_model = garnet.load_model(
        str(SCRIPT_DIR / "text_lm_head_trt.x"),
        weights={"language_model.embed_tokens.weight": embed_tokens},
        cache_dir=str(cache_dir),
        input_shapes=[list(actual_norm.shape)],
        subgraph="text_lm_head",
    )
    actual_logits = to_numpy(lm_head_model.forward(actual_norm)).reshape(expected_logits.shape)

    expected_top = np.argsort(expected_logits[-1])[-10:][::-1]
    actual_top = np.argsort(actual_logits[-1])[-10:][::-1]
    overlap = len(set(int(x) for x in expected_top[:10]) & set(int(x) for x in actual_top[:10]))
    expected_next_id = int(expected_top[0])
    actual_next_id = int(actual_top[0])
    expected_next_text = tokenizer.decode([expected_next_id]) if tokenizer is not None else ""
    actual_next_text = tokenizer.decode([actual_next_id]) if tokenizer is not None else ""
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=3e-1, atol=8.0)
    if overlap < 8:
        raise AssertionError(f"full prompt top-10 token overlap too low: {overlap}/10")
    return {
        "processor_npz": str(window["processor_npz"]),
        "visual_npz": str(window["visual_npz"]),
        "grid_thw": None if window["grid_thw"] is None else window["grid_thw"].tolist(),
        "input_ids": [int(x) for x in window["input_ids"].tolist()],
        "visual_positions": [int(x) for x in window["visual_positions"].tolist()],
        "use_all_vision_blocks": bool(use_all_vision),
        "vision_block_count": int(window["vision_block_count"]),
        "text_layer_count": int(layer_count),
        "logits_shape": list(actual_logits.shape),
        "max_error": float(np.max(np.abs(actual_logits - expected_logits))),
        "mean_error": float(np.mean(np.abs(actual_logits - expected_logits))),
        "last_token_top10_overlap": overlap,
        "expected_next_token_id": expected_next_id,
        "actual_next_token_id": actual_next_id,
        "expected_next_token_text": expected_next_text,
        "actual_next_token_text": actual_next_text,
        "expected_top10_ids": [int(x) for x in expected_top.tolist()],
        "actual_top10_ids": [int(x) for x in actual_top.tolist()],
        "visual_token_max_error": float(window["visual_token_max_error"]),
        "visual_token_mean_error": float(window["visual_token_mean_error"]),
    }


class GarnetQwen3VLForwardFacade:
    def __init__(self, garnet, model_dir, key_to_file, cache_root, use_all_vision_blocks=True):
        self.garnet = garnet
        self.model_dir = model_dir
        self.key_to_file = key_to_file
        self.cache_root = Path(cache_root)
        self.use_all_vision_blocks = bool(use_all_vision_blocks)
        try:
            from transformers import AutoConfig
            self.config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        except Exception:
            self.config = None
        self.embed_tokens = load_tensor(key_to_file, "language_model.embed_tokens.weight")
        self.final_norm = load_tensor(key_to_file, "language_model.norm.weight")
        self.text_layer_count = text_layer_count_from_weights(key_to_file)
        self.vision_block_count = vision_block_count_from_weights(key_to_file)
        self.cpp_vision_runner_enabled = os.environ.get("GARNET_QWEN_CPP_VISION_RUNNER", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.gpu_vision_layer_weights = {}
        self.gpu_vision_models = {}
        self.gpu_vision_runner = None
        self.gpu_vision_runner_bundles = None
        self.gpu_vision_merger_bundle = None
        self.gpu_vision_deepstack_bundles = []
        self.deepstack_visual_tokens = []
        self.gpu_vision_patch_model = None
        self.gpu_vision_patch_weights = None
        self.vision_runner_prepare_ms = 0.0
        self.vision_runner_warmup_ms = 0.0
        self.vision_runner_warmed = False
        self.vision_patch_ms = 0.0
        self.vision_execute_ms = 0.0

    def _gpu_vision_models_for_layer(self, layer_idx, patch_count):
        layer_idx = int(layer_idx)
        cached = self.gpu_vision_models.get(layer_idx)
        if cached is not None:
            return cached
        prefix = f"visual.blocks.{layer_idx}"
        cpu_weights = {
            "norm1_w": load_tensor(self.key_to_file, prefix + ".norm1.weight"),
            "norm1_b": load_tensor(self.key_to_file, prefix + ".norm1.bias"),
            "norm2_w": load_tensor(self.key_to_file, prefix + ".norm2.weight"),
            "norm2_b": load_tensor(self.key_to_file, prefix + ".norm2.bias"),
            "qkv_w": load_tensor(self.key_to_file, prefix + ".attn.qkv.weight"),
            "qkv_b": load_tensor(self.key_to_file, prefix + ".attn.qkv.bias"),
            "proj_w": load_tensor(self.key_to_file, prefix + ".attn.proj.weight"),
            "proj_b": load_tensor(self.key_to_file, prefix + ".attn.proj.bias"),
            "fc1_w": load_tensor(self.key_to_file, prefix + ".mlp.linear_fc1.weight"),
            "fc1_b": load_tensor(self.key_to_file, prefix + ".mlp.linear_fc1.bias"),
            "fc2_w": load_tensor(self.key_to_file, prefix + ".mlp.linear_fc2.weight"),
            "fc2_b": load_tensor(self.key_to_file, prefix + ".mlp.linear_fc2.bias"),
        }
        # TRT engines own their device-side constant weights. Keeping a second
        # X::Tensor CUDA copy here nearly doubles model VRAM for no runtime gain.
        weights = {
            "norm1_w": cpu_weights["norm1_w"],
            "norm1_b": cpu_weights["norm1_b"],
            "norm2_w": cpu_weights["norm2_w"],
            "norm2_b": cpu_weights["norm2_b"],
            "qkv_w": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, prefix + ".attn.qkv.weight"),
            "qkv_b": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, prefix + ".attn.qkv.bias"),
            "proj_w": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, prefix + ".attn.proj.weight"),
            "proj_b": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, prefix + ".attn.proj.bias"),
            "fc1_w": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, prefix + ".mlp.linear_fc1.weight"),
            "fc1_b": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, prefix + ".mlp.linear_fc1.bias"),
            "fc2_w": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, prefix + ".mlp.linear_fc2.weight"),
            "fc2_b": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, prefix + ".mlp.linear_fc2.bias"),
        }
        self.gpu_vision_layer_weights[layer_idx] = weights
        cache_dir = self.cache_root / "gpu_vision_runner" / f"patches_{patch_count}"
        payloads = {
            "norm1": {"visual.blocks.0.norm1.weight": weights["norm1_w"], "visual.blocks.0.norm1.bias": weights["norm1_b"]},
            "norm2": {"visual.blocks.0.norm1.weight": weights["norm2_w"], "visual.blocks.0.norm1.bias": weights["norm2_b"]},
            "qkv": {"W": weights["qkv_w"], "B": weights["qkv_b"]},
            "proj": {"W": weights["proj_w"], "B": weights["proj_b"]},
            "mlp_fc1": {"W": weights["fc1_w"], "B": weights["fc1_b"]},
            "mlp_fc2": {"W": weights["fc2_w"], "B": weights["fc2_b"]},
        }
        models = {
            "norm1": self.garnet.load_model(str(SCRIPT_DIR / "layer_norm_trt.x"), weights=payloads["norm1"], cache_dir=str(cache_dir / "norm"), input_shapes=[[patch_count, 1024]], subgraph="layer_norm"),
            "qkv": self.garnet.load_model(str(SCRIPT_DIR / "linear_bias_trt.x"), weights=payloads["qkv"], cache_dir=str(cache_dir / "qkv"), input_shapes=[[patch_count, 1024]], subgraph="linear_bias"),
            "attention": self.garnet.load_model(str(SCRIPT_DIR / "vision_attention_core_trt.x"), weights={}, cache_dir=str(cache_dir / "attention"), input_shapes=[[patch_count, 3072]], subgraph="vision_attention_core"),
            "proj": self.garnet.load_model(str(SCRIPT_DIR / "linear_bias_trt.x"), weights=payloads["proj"], cache_dir=str(cache_dir / "proj"), input_shapes=[[patch_count, 1024]], subgraph="linear_bias"),
            "norm2": self.garnet.load_model(str(SCRIPT_DIR / "layer_norm_trt.x"), weights=payloads["norm2"], cache_dir=str(cache_dir / "norm"), input_shapes=[[patch_count, 1024]], subgraph="layer_norm"),
            "mlp_fc1": self.garnet.load_model(str(SCRIPT_DIR / "linear_bias_trt.x"), weights=payloads["mlp_fc1"], cache_dir=str(cache_dir / "mlp_fc1"), input_shapes=[[patch_count, 1024]], subgraph="linear_bias"),
            "mlp_fc2": self.garnet.load_model(str(SCRIPT_DIR / "linear_bias_trt.x"), weights=payloads["mlp_fc2"], cache_dir=str(cache_dir / "mlp_fc2"), input_shapes=[[patch_count, 4096]], subgraph="linear_bias"),
            "_payloads": payloads,
        }
        self.gpu_vision_models[layer_idx] = models
        return models

    def _create_gpu_vision_runner(self, patch_count):
        if self.gpu_vision_runner is not None:
            return self.gpu_vision_runner
        bundle_keys = ["norm1", "qkv", "attention", "proj", "norm2", "mlp_fc1", "mlp_fc2"]
        bundles = []
        for layer_idx in range(self.vision_block_count):
            models = self._gpu_vision_models_for_layer(layer_idx, patch_count)
            bundles.append({name: models[name] for name in bundle_keys})

        visual_tokens = patch_count // 4
        for slot, layer_idx in enumerate(self.config.vision_config.deepstack_visual_indexes):
            prefix = f"visual.deepstack_merger_list.{slot}"
            merger_weights = {
                "norm_w": load_tensor(self.key_to_file, prefix + ".norm.weight"),
                "norm_b": load_tensor(self.key_to_file, prefix + ".norm.bias"),
                "fc1_w": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, prefix + ".linear_fc1.weight"),
                "fc1_b": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, prefix + ".linear_fc1.bias"),
                "fc2_w": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, prefix + ".linear_fc2.weight"),
                "fc2_b": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, prefix + ".linear_fc2.bias"),
            }
            cache_dir = self.cache_root / "gpu_vision_runner" / f"patches_{patch_count}" / f"deepstack_{slot}"
            payloads = {
                "norm": {"visual.blocks.0.norm1.weight": merger_weights["norm_w"], "visual.blocks.0.norm1.bias": merger_weights["norm_b"]},
                "fc1": {"W": merger_weights["fc1_w"], "B": merger_weights["fc1_b"]},
                "fc2": {"W": merger_weights["fc2_w"], "B": merger_weights["fc2_b"]},
            }
            deepstack_bundle = {
                "norm": self.garnet.load_model(str(SCRIPT_DIR / "layer_norm_trt.x"), weights=payloads["norm"], cache_dir=str(cache_dir / "norm"), input_shapes=[[visual_tokens, 4096]], subgraph="layer_norm"),
                "fc1": self.garnet.load_model(str(SCRIPT_DIR / "linear_bias_trt.x"), weights=payloads["fc1"], cache_dir=str(cache_dir / "fc1"), input_shapes=[[visual_tokens, 4096]], subgraph="linear_bias"),
                "fc2": self.garnet.load_model(str(SCRIPT_DIR / "linear_bias_trt.x"), weights=payloads["fc2"], cache_dir=str(cache_dir / "fc2"), input_shapes=[[visual_tokens, 4096]], subgraph="linear_bias"),
                "_weights": merger_weights,
                "_payloads": payloads,
            }
            bundles[int(layer_idx)]["deepstack"] = {
                name: deepstack_bundle[name] for name in ["norm", "fc1", "fc2"]
            }
            self.gpu_vision_deepstack_bundles.append(deepstack_bundle)

        merger_cpu = {
            "norm_w": load_tensor(self.key_to_file, "visual.merger.norm.weight"),
            "norm_b": load_tensor(self.key_to_file, "visual.merger.norm.bias"),
            "fc1_w": load_tensor(self.key_to_file, "visual.merger.linear_fc1.weight"),
            "fc1_b": load_tensor(self.key_to_file, "visual.merger.linear_fc1.bias"),
            "fc2_w": load_tensor(self.key_to_file, "visual.merger.linear_fc2.weight"),
            "fc2_b": load_tensor(self.key_to_file, "visual.merger.linear_fc2.bias"),
        }
        merger_weights = {
            "norm_w": merger_cpu["norm_w"],
            "norm_b": merger_cpu["norm_b"],
            "fc1_w": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, "visual.merger.linear_fc1.weight"),
            "fc1_b": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, "visual.merger.linear_fc1.bias"),
            "fc2_w": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, "visual.merger.linear_fc2.weight"),
            "fc2_b": load_bfloat16_gpu_tensor(self.garnet, self.key_to_file, "visual.merger.linear_fc2.bias"),
        }
        cache_dir = self.cache_root / "gpu_vision_runner" / f"patches_{patch_count}" / "merger"
        merger_payloads = {
            "norm": {"visual.blocks.0.norm1.weight": merger_weights["norm_w"], "visual.blocks.0.norm1.bias": merger_weights["norm_b"]},
            "fc1": {"W": merger_weights["fc1_w"], "B": merger_weights["fc1_b"]},
            "fc2": {"W": merger_weights["fc2_w"], "B": merger_weights["fc2_b"]},
        }
        merger_bundle = {
            "norm": self.garnet.load_model(str(SCRIPT_DIR / "layer_norm_trt.x"), weights=merger_payloads["norm"], cache_dir=str(cache_dir / "norm"), input_shapes=[[patch_count, 1024]], subgraph="layer_norm"),
            "fc1": self.garnet.load_model(str(SCRIPT_DIR / "linear_bias_trt.x"), weights=merger_payloads["fc1"], cache_dir=str(cache_dir / "fc1"), input_shapes=[[visual_tokens, 4096]], subgraph="linear_bias"),
            "fc2": self.garnet.load_model(str(SCRIPT_DIR / "linear_bias_trt.x"), weights=merger_payloads["fc2"], cache_dir=str(cache_dir / "fc2"), input_shapes=[[visual_tokens, 4096]], subgraph="linear_bias"),
            "_weights": merger_weights,
            "_payloads": merger_payloads,
        }
        runner = self.garnet.QwenVisionRunner(bundles, {name: merger_bundle[name] for name in ["norm", "fc1", "fc2"]})
        if runner is None or int(runner.stats()["layer_count"]) != self.vision_block_count:
            raise AssertionError("failed to create C++ QwenVisionRunner")
        self.gpu_vision_runner_bundles = bundles
        self.gpu_vision_merger_bundle = merger_bundle
        self.gpu_vision_runner = runner
        return runner

    def _visual_tokens_gpu(self, pixel_values, image_grid_thw, visual_token_count):
        prepare_start = time.perf_counter()
        patch_count = int(visual_token_count) * 4
        pixel_values = np.asarray(pixel_values[:patch_count], dtype=np.float32)
        grid_thw = np.asarray(image_grid_thw, dtype=np.int64)
        patch_weight_5d = load_tensor(self.key_to_file, "visual.patch_embed.proj.weight")
        patch_weight = patch_weight_5d.reshape(patch_weight_5d.shape[0], -1).astype(np.float32)
        patch_bias = load_tensor(self.key_to_file, "visual.patch_embed.proj.bias")
        pos_weight = load_tensor(self.key_to_file, "visual.pos_embed.weight")
        bilinear_indices, bilinear_weights = qwen3_vl_vision_bilinear_numpy(grid_thw)
        pos = np.sum(pos_weight[bilinear_indices[:, :patch_count]] * bilinear_weights[:, :patch_count, None], axis=0).astype(np.float32)

        if self.gpu_vision_patch_model is None:
            patch_weights = {
                "visual.patch_embed.proj.weight": load_bfloat16_gpu_tensor(
                    self.garnet, self.key_to_file, "visual.patch_embed.proj.weight",
                    reshape=(patch_weight.shape[0], patch_weight.shape[1]),
                ),
                "visual.patch_embed.proj.bias": load_bfloat16_gpu_tensor(
                    self.garnet, self.key_to_file, "visual.patch_embed.proj.bias"
                ),
            }
            self.gpu_vision_patch_weights = patch_weights
            self.gpu_vision_patch_model = self.garnet.load_model(
                str(SCRIPT_DIR / "vision_patch_embed_trt.x"),
                weights=patch_weights,
                cache_dir=str(self.cache_root / "gpu_vision_runner" / f"patches_{patch_count}" / "patch"),
                input_shapes=[[patch_count, pixel_values.shape[1]]],
                subgraph="vision_patch_embed",
            )
        runner = self._create_gpu_vision_runner(patch_count)
        cos, sin = vision_rope_cos_sin_numpy(grid_thw, self.model_dir, patch_count)
        self.vision_runner_prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

        patch_start = time.perf_counter()
        patch = self.gpu_vision_patch_model.forward(pixel_values)
        hidden = self.garnet.tensor_add(patch, pos)
        self.vision_patch_ms = (time.perf_counter() - patch_start) * 1000.0
        if (not self.vision_runner_warmed and
                os.environ.get("GARNET_QWEN_VISION_RUNNER_WARMUP", "1").strip().lower() in {"1", "true", "yes", "on"}):
            warmup_start = time.perf_counter()
            warmup_result = runner.forward(hidden, cos, sin)
            if warmup_result is None or warmup_result["pooler_output"] is None:
                raise AssertionError("C++ QwenVisionRunner warm-up failed")
            self.vision_runner_warmup_ms = (time.perf_counter() - warmup_start) * 1000.0
            self.vision_runner_warmed = True
        else:
            self.vision_runner_warmup_ms = 0.0
        execute_start = time.perf_counter()
        output = runner.forward(hidden, cos, sin)
        self.vision_execute_ms = (time.perf_counter() - execute_start) * 1000.0
        if output is None or output["pooler_output"] is None:
            raise AssertionError("C++ QwenVisionRunner forward failed")
        visual_tokens = output["pooler_output"]
        self.deepstack_visual_tokens = output["deepstack_features"]
        return visual_tokens

    def _visual_tokens(self, pixel_values, image_grid_thw, visual_token_count):
        if getattr(self, "gpu_tensor_chain", False) and self.cpp_vision_runner_enabled:
            return self._visual_tokens_gpu(pixel_values, image_grid_thw, visual_token_count)
        patch_count = int(visual_token_count) * 4
        x = vision_patch_with_pos_from_arrays(self.key_to_file, pixel_values, image_grid_thw, patch_count).astype(np.float32)
        if self.use_all_vision_blocks:
            for layer_idx in range(self.vision_block_count):
                x = run_garnet_vision_block(
                    self.garnet,
                    x,
                    image_grid_thw,
                    self.model_dir,
                    self.key_to_file,
                    layer_idx,
                    self.cache_root / f"vision_patch_{patch_count}" / f"block_{layer_idx}",
                )
        return run_garnet_vision_merger(
            self.garnet,
            x,
            self.key_to_file,
            self.cache_root / f"vision_patch_{patch_count}" / "merger",
        )

    def forward(self, input_ids, pixel_values, image_grid_thw, mm_token_type_ids, cos_np=None, sin_np=None, last_token_logits_only=False, visual_tokens_override=None):
        input_ids = np.asarray(input_ids, dtype=np.int64)
        mm_token_type_ids = np.asarray(mm_token_type_ids, dtype=np.int64)
        image_grid_thw = np.asarray(image_grid_thw, dtype=np.int64)
        if cos_np is None or sin_np is None:
            if self.config is None:
                raise AssertionError("native MRoPE cos/sin requires model config")
            position_ids, _ = qwen3vl_mrope_position_ids_numpy(
                input_ids[None, :],
                mm_token_type_ids[None, :],
                image_grid_thw=image_grid_thw,
                attention_mask=np.ones((1, input_ids.shape[0]), dtype=np.int64),
                spatial_merge_size=self.config.vision_config.spatial_merge_size,
            )
            cos, sin = qwen3vl_text_mrope_cos_sin_numpy(
                position_ids,
                head_dim=self.config.text_config.head_dim,
                rope_theta=self.config.text_config.rope_parameters["rope_theta"],
                mrope_section=self.config.text_config.rope_parameters.get("mrope_section", [24, 20, 20]),
            )
            cos_np = cos[0]
            sin_np = sin[0]
        x = self.embed_tokens[input_ids].astype(np.float32)
        visual_positions = np.flatnonzero(mm_token_type_ids == 1)
        if visual_positions.size:
            if visual_tokens_override is not None:
                visual_tokens = np.asarray(visual_tokens_override, dtype=np.float32)
                if visual_tokens.shape[0] != visual_positions.size:
                    raise AssertionError(
                        f"visual token cache has {visual_tokens.shape[0]} tokens, "
                        f"but prompt has {visual_positions.size} visual placeholders"
                    )
            else:
                visual_tokens = self._visual_tokens(pixel_values, image_grid_thw, int(visual_positions.size))
            x[visual_positions] = visual_tokens
        else:
            visual_tokens = np.zeros((0, self.embed_tokens.shape[1]), dtype=np.float32)

        cache_dir = self.cache_root / f"text_tokens_{input_ids.shape[0]}"
        hidden = x
        for layer_idx in range(self.text_layer_count):
            layer_weights = load_text_layer_weights_as_layer0_aliases(self.key_to_file, layer_idx)
            hidden = run_garnet_text_decoder_layer(self.garnet, layer_weights, hidden, cos_np, sin_np, cache_dir)

        final_norm_model = self.garnet.load_model(
            str(SCRIPT_DIR / "rms_norm_trt.x"),
            weights={"language_model.layers.0.input_layernorm.weight": self.final_norm},
            cache_dir=str(cache_dir),
            input_shapes=[list(hidden.shape)],
            subgraph="rms_norm",
        )
        normed = to_numpy(final_norm_model.forward(hidden)).reshape(hidden.shape)

        lm_head_input = normed[-1:, :] if last_token_logits_only else normed
        lm_head_model = self.garnet.load_model(
            str(SCRIPT_DIR / "text_lm_head_trt.x"),
            weights={"language_model.embed_tokens.weight": self.embed_tokens},
            cache_dir=str(cache_dir),
            input_shapes=[list(lm_head_input.shape)],
            subgraph="text_lm_head",
        )
        lm_head_output = lm_head_model.forward(lm_head_input)
        logits = logits_to_numpy_or_cpu_copy(self.garnet, lm_head_output, lm_head_input.shape[0], self.embed_tokens.shape[0])
        return {
            "logits": logits,
            "hidden_states": hidden,
            "visual_positions": visual_positions,
            "visual_tokens": visual_tokens,
            "text_layer_count": self.text_layer_count,
            "last_token_logits_only": bool(last_token_logits_only),
            "vision_block_count": self.vision_block_count if self.use_all_vision_blocks else 0,
        }


class GarnetQwen3VLCachedForwardFacade(GarnetQwen3VLForwardFacade):
    def __init__(self, garnet, model_dir, key_to_file, cache_root, use_all_vision_blocks=True, max_new_tokens=1, page_size=16, text_layer_count=None):
        super().__init__(garnet, model_dir, key_to_file, cache_root, use_all_vision_blocks=use_all_vision_blocks)
        self.max_new_tokens = int(max_new_tokens)
        self.page_size = int(page_size)
        if text_layer_count is None:
            self.cached_text_layer_count = self.text_layer_count
        else:
            self.cached_text_layer_count = max(1, min(int(text_layer_count), self.text_layer_count))
        self.dll = configure_paged_kv_attention_ctypes(load_garnet_dll_ctypes())
        self.layer_caches = []
        self.device_kv_handles = []
        self.page_table = None
        self.physical_pages = 0
        self.prefill_length = 0
        self.logical_length = 0
        self.image_grid_thw = None
        self.mm_token_type_ids = None
        self.gpu_tensor_chain = os.environ.get("GARNET_QWEN_GPU_TENSOR_CHAIN", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.gpu_layer_models = {}
        self.gpu_output_models = {}
        self.gpu_layer_weights = {}
        self.gpu_embed_tokens = None
        self.gpu_final_norm = None
        self.cpp_text_runner_enabled = os.environ.get("GARNET_QWEN_CPP_TEXT_RUNNER", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.cpp_text_runners = {}
        self.cpp_text_runner_bundles = {}

    def close(self):
        for handle in self.device_kv_handles:
            destroy_device_paged_kv_handle(self.dll, handle)
        self.device_kv_handles = []

    def _gpu_models_for_layer(self, layer_idx, tokens):
        key = (int(layer_idx), int(tokens))
        cached = self.gpu_layer_models.get(key)
        if cached is not None:
            return cached

        layer_weights = self.gpu_layer_weights.get(int(layer_idx))
        if layer_weights is None:
            if os.environ.get("GARNET_QWEN_BFLOAT16_WEIGHTS", "1").strip().lower() in {"1", "true", "yes", "on"}:
                layer_weights = load_text_layer_mixed_weights_as_layer0_aliases(
                    self.garnet, self.key_to_file, layer_idx
                )
            else:
                layer_weights = load_text_layer_weights_as_layer0_aliases(self.key_to_file, layer_idx)
            self.gpu_layer_weights[int(layer_idx)] = layer_weights
        layer_prefix = "language_model.layers.0"
        attn_prefix = layer_prefix + ".self_attn"
        mlp_prefix = layer_prefix + ".mlp"
        cache_dir = self.cache_root / "gpu_tensor_chain" / f"tokens_{tokens}"
        weight_payloads = {
            "input_norm": {layer_prefix + ".input_layernorm.weight": layer_weights[layer_prefix + ".input_layernorm.weight"]},
            "qkv": {
                attn_prefix + ".q_proj.weight": layer_weights[attn_prefix + ".q_proj.weight"],
                attn_prefix + ".k_proj.weight": layer_weights[attn_prefix + ".k_proj.weight"],
                attn_prefix + ".v_proj.weight": layer_weights[attn_prefix + ".v_proj.weight"],
                attn_prefix + ".q_norm.weight": layer_weights[attn_prefix + ".q_norm.weight"],
                attn_prefix + ".k_norm.weight": layer_weights[attn_prefix + ".k_norm.weight"],
            },
            "o_proj": {attn_prefix + ".o_proj.weight": layer_weights[attn_prefix + ".o_proj.weight"]},
            "post_norm": {layer_prefix + ".post_attention_layernorm.weight": layer_weights[layer_prefix + ".post_attention_layernorm.weight"]},
            "mlp": {
                mlp_prefix + ".gate_proj.weight": layer_weights[mlp_prefix + ".gate_proj.weight"],
                mlp_prefix + ".up_proj.weight": layer_weights[mlp_prefix + ".up_proj.weight"],
                mlp_prefix + ".down_proj.weight": layer_weights[mlp_prefix + ".down_proj.weight"],
            },
        }
        models = {
            "input_norm": self.garnet.load_model(
                str(SCRIPT_DIR / "rms_norm_trt.x"),
                weights=weight_payloads["input_norm"],
                cache_dir=str(cache_dir / "input_norm"),
                input_shapes=[[tokens, 2048]],
                subgraph="rms_norm",
            ),
            "qkv": self.garnet.load_model(
                str(SCRIPT_DIR / "text_qkv_head_norm_trt.x"),
                weights=weight_payloads["qkv"],
                cache_dir=str(cache_dir / "qkv"),
                input_shapes=[[tokens, 2048]],
                subgraph="text_qkv_head_norm",
            ),
            "rope": self.garnet.load_model(
                str(SCRIPT_DIR / "text_rope_apply_trt.x"),
                weights={},
                cache_dir=str(cache_dir / "rope"),
                input_shapes=[[tokens, 4096], [tokens, 128], [tokens, 128]],
                subgraph="text_rope_apply",
            ),
            "attention": self.garnet.load_model(
                str(SCRIPT_DIR / "text_attention_core_trt.x"),
                weights={},
                cache_dir=str(cache_dir / "attention"),
                input_shapes=[[tokens, 4096]],
                subgraph="text_attention_core",
            ),
            "o_proj": self.garnet.load_model(
                str(SCRIPT_DIR / "text_o_proj_trt.x"),
                weights=weight_payloads["o_proj"],
                cache_dir=str(cache_dir / "o_proj"),
                input_shapes=[[tokens, 2048]],
                subgraph="text_o_proj",
            ),
            "post_norm": self.garnet.load_model(
                str(SCRIPT_DIR / "text_post_rms_norm_trt.x"),
                weights=weight_payloads["post_norm"],
                cache_dir=str(cache_dir / "post_norm"),
                input_shapes=[[tokens, 2048]],
                subgraph="text_post_attention_rms_norm",
            ),
            "mlp": self.garnet.load_model(
                str(SCRIPT_DIR / "text_mlp_trt.x"),
                weights=weight_payloads["mlp"],
                cache_dir=str(cache_dir / "mlp"),
                input_shapes=[[tokens, 2048]],
                subgraph="qwen3_text_mlp",
            ),
        }
        if any(model is None for model in models.values()):
            raise AssertionError(f"failed to load GPU tensor-chain models for text layer {layer_idx}")
        models["_weight_payloads"] = weight_payloads
        models["_layer_weights"] = layer_weights
        self.gpu_layer_models[key] = models
        return models

    def _gpu_layer_forward(self, layer_idx, hidden, cos_np, sin_np, kv_handle, start_position, sequence_length, prefill):
        tokens = int(cos_np.shape[0])
        models = self._gpu_models_for_layer(layer_idx, tokens)
        normed = models["input_norm"].forward(hidden)
        qkv = models["qkv"].forward(normed)
        rope_result = models["rope"].forward(qkv, cos_np, sin_np, int(kv_handle), tokens, int(start_position))
        if rope_result is None or str(rope_result["status"]) != "ok":
            raise AssertionError(f"GPU tensor-chain RoPE/KV write failed at layer {layer_idx}")
        rope = rope_result["output_tensor"]
        if prefill:
            attention = models["attention"].forward(rope)
        else:
            attention_result = models["attention"].forward(rope, int(kv_handle), int(sequence_length), 2048)
            if attention_result is None or str(attention_result["status"]) != "ok":
                raise AssertionError(f"GPU tensor-chain KV attention failed at layer {layer_idx}")
            attention = attention_result["output_tensor"]
        projected = models["o_proj"].forward(attention)
        residual = self.garnet.tensor_add(hidden, projected)
        post_norm = models["post_norm"].forward(residual)
        mlp = models["mlp"].forward(post_norm)
        return self.garnet.tensor_add(residual, mlp)

    def _cpp_text_runner(self, tokens):
        tokens = int(tokens)
        cached = self.cpp_text_runners.get(tokens)
        if cached is not None:
            return cached
        bundle_keys = ["input_norm", "qkv", "rope", "attention", "o_proj", "post_norm", "mlp"]
        bundles = []
        for layer_idx in range(self.cached_text_layer_count):
            models = self._gpu_models_for_layer(layer_idx, tokens)
            bundles.append({name: models[name] for name in bundle_keys})
        runner = self.garnet.QwenTextRunner(bundles)
        if runner is None:
            raise AssertionError("failed to create C++ QwenTextRunner")
        stats = runner.stats()
        if int(stats["layer_count"]) != self.cached_text_layer_count:
            raise AssertionError("C++ QwenTextRunner layer count mismatch")
        self.cpp_text_runner_bundles[tokens] = bundles
        self.cpp_text_runners[tokens] = runner
        return runner

    def _gpu_sample_hidden(self, hidden, tokens):
        hidden = self.garnet.tensor_last_row(hidden)
        tokens = 1
        if self.gpu_embed_tokens is None:
            self.gpu_embed_tokens = load_bfloat16_gpu_tensor(
                self.garnet, self.key_to_file, "language_model.embed_tokens.weight"
            )
        cached = self.gpu_output_models.get(tokens)
        if cached is None:
            cache_dir = self.cache_root / "gpu_tensor_chain" / f"tokens_{tokens}"
            output_weight_payloads = {
                "norm": {"language_model.layers.0.input_layernorm.weight": self.final_norm},
                "lm_head": {"language_model.embed_tokens.weight": self.gpu_embed_tokens},
            }
            cached = {
                "norm": self.garnet.load_model(
                    str(SCRIPT_DIR / "rms_norm_trt.x"),
                    weights=output_weight_payloads["norm"],
                    cache_dir=str(cache_dir / "final_norm"),
                    input_shapes=[[tokens, 2048]],
                    subgraph="rms_norm",
                ),
                "lm_head": self.garnet.load_model(
                    str(SCRIPT_DIR / "text_lm_head_trt.x"),
                    weights=output_weight_payloads["lm_head"],
                    cache_dir=str(cache_dir / "lm_head"),
                    input_shapes=[[tokens, 2048]],
                    subgraph="text_lm_head",
                ),
                "_weight_payloads": output_weight_payloads,
            }
            self.gpu_output_models[tokens] = cached
        normed = cached["norm"].forward(hidden)
        logits = cached["lm_head"].forward(normed)
        probe = cached["lm_head"].debug_probe("logits_top1", logits)
        if probe is None or str(probe["status"]) != "ok":
            raise AssertionError("GPU tensor-chain logits sampling failed")
        return int(probe["token_id"])

    def _cos_sin_for(self, input_ids, mm_token_type_ids, image_grid_thw):
        if self.config is None:
            raise AssertionError("native MRoPE cos/sin requires model config")
        position_ids, _ = qwen3vl_mrope_position_ids_numpy(
            input_ids[None, :],
            mm_token_type_ids[None, :],
            image_grid_thw=image_grid_thw,
            attention_mask=np.ones((1, input_ids.shape[0]), dtype=np.int64),
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
        )
        cos, sin = qwen3vl_text_mrope_cos_sin_numpy(
            position_ids,
            head_dim=self.config.text_config.head_dim,
            rope_theta=self.config.text_config.rope_parameters["rope_theta"],
            mrope_section=self.config.text_config.rope_parameters.get("mrope_section", [24, 20, 20]),
        )
        return cos[0].astype(np.float32), sin[0].astype(np.float32)

    def _logits_from_hidden(self, hidden, cache_dir):
        final_norm_model = self.garnet.load_model(
            str(SCRIPT_DIR / "rms_norm_trt.x"),
            weights={"language_model.layers.0.input_layernorm.weight": self.final_norm},
            cache_dir=str(cache_dir / "final_norm"),
            input_shapes=[list(hidden.shape)],
            subgraph="rms_norm",
        )
        normed = to_numpy(final_norm_model.forward(hidden)).reshape(hidden.shape)
        lm_head_model = self.garnet.load_model(
            str(SCRIPT_DIR / "text_lm_head_trt.x"),
            weights={"language_model.embed_tokens.weight": self.embed_tokens},
            cache_dir=str(cache_dir / "lm_head"),
            input_shapes=[list(normed.shape)],
            subgraph="text_lm_head",
        )
        lm_head_output = lm_head_model.forward(normed)
        return logits_to_numpy_or_cpu_copy(self.garnet, lm_head_output, normed.shape[0], self.embed_tokens.shape[0])

    def prefill(self, input_ids, pixel_values, image_grid_thw, mm_token_type_ids, visual_tokens_override=None):
        input_ids = np.asarray(input_ids, dtype=np.int64)
        mm_token_type_ids = np.asarray(mm_token_type_ids, dtype=np.int64)
        image_grid_thw = np.asarray(image_grid_thw, dtype=np.int64)
        self.prefill_length = int(input_ids.shape[0])
        self.logical_length = self.prefill_length
        self.image_grid_thw = image_grid_thw.copy()
        self.mm_token_type_ids = mm_token_type_ids.copy()

        total_capacity = self.prefill_length + self.max_new_tokens
        logical_pages = (total_capacity + self.page_size - 1) // self.page_size
        self.physical_pages = logical_pages + 2
        self.page_table = np.arange(logical_pages, dtype=np.int32)

        cos_np, sin_np = self._cos_sin_for(input_ids, mm_token_type_ids, image_grid_thw)
        visual_positions = np.flatnonzero(mm_token_type_ids == 1)
        if visual_positions.size:
            if visual_tokens_override is not None:
                if type(visual_tokens_override).__name__ == "XlangObject":
                    visual_tokens = visual_tokens_override
                else:
                    visual_tokens = np.asarray(visual_tokens_override, dtype=np.float32)
            else:
                visual_tokens = self._visual_tokens(pixel_values, image_grid_thw, int(visual_positions.size))
            if type(visual_tokens).__name__ != "XlangObject" and visual_tokens.shape[0] != visual_positions.size:
                raise AssertionError(
                    f"visual token cache has {visual_tokens.shape[0]} tokens, "
                    f"but prompt has {visual_positions.size} visual placeholders"
                )
        else:
            visual_tokens = np.zeros((0, self.embed_tokens.shape[1]), dtype=np.float32)

        if self.gpu_tensor_chain:
            if self.gpu_embed_tokens is None:
                self.gpu_embed_tokens = load_bfloat16_gpu_tensor(
                    self.garnet, self.key_to_file, "language_model.embed_tokens.weight"
                )
            hidden = self.garnet.embedding(self.gpu_embed_tokens, input_ids)
            if visual_positions.size:
                hidden = self.garnet.replace_rows_by_mask(hidden, mm_token_type_ids, visual_tokens, 1)
        else:
            hidden = self.embed_tokens[input_ids].astype(np.float32)
            if visual_positions.size:
                hidden[visual_positions] = visual_tokens

        cache_dir = self.cache_root / f"cached_text_prefill_tokens_{self.prefill_length}_layers_{self.cached_text_layer_count}"
        self.layer_caches = []
        self.close()
        for layer_idx in range(self.cached_text_layer_count):
            handle = create_device_paged_kv_handle(
                self.dll,
                self.physical_pages,
                self.page_size,
                int(self.page_table.shape[0]),
                16,
                8,
                128,
                self.page_table,
            )
            self.device_kv_handles.append(handle)
            self.layer_caches.append(handle)

        if self.gpu_tensor_chain and self.cpp_text_runner_enabled:
            hidden = self._cpp_text_runner(self.prefill_length).prefill(
                hidden, cos_np, sin_np, self.layer_caches, 0, self.prefill_length,
                self.deepstack_visual_tokens, mm_token_type_ids
            )
            if hidden is None:
                raise AssertionError("C++ QwenTextRunner prefill failed")
        else:
            for layer_idx, handle in enumerate(self.layer_caches):
                if self.gpu_tensor_chain:
                    hidden = self._gpu_layer_forward(
                        layer_idx, hidden, cos_np, sin_np, handle, 0, self.prefill_length, True
                    )
                    continue
                layer_weights = load_text_layer_weights_as_layer0_aliases(self.key_to_file, layer_idx)
                hidden = run_garnet_text_decoder_layer_prefill_with_device_paged_kv(
                    self.garnet,
                    self.dll,
                    layer_weights,
                    hidden,
                    cos_np,
                    sin_np,
                    cache_dir / f"layer_{layer_idx}",
                    handle,
                )

        if self.gpu_tensor_chain:
            next_token_id = self._gpu_sample_hidden(hidden, self.prefill_length)
            logits = None
        else:
            next_token_id = None
            logits = self._logits_from_hidden(hidden[-1:], cache_dir)
        return {
            "logits": logits,
            "next_token_id": next_token_id,
            "hidden_states": hidden,
            "visual_positions": visual_positions,
            "visual_tokens": visual_tokens,
            "text_layer_count": self.cached_text_layer_count,
            "full_text_layer_count": self.text_layer_count,
            "vision_block_count": self.vision_block_count if self.use_all_vision_blocks else 0,
            "kv_page_size": self.page_size,
            "kv_logical_pages": int(self.page_table.shape[0]),
            "kv_physical_pages": int(self.physical_pages),
            "cached_decode": True,
        }

    def decode_one(self, token_id, prefix_input_ids, prefix_mm_token_type_ids):
        if self.page_table is None or not self.layer_caches:
            raise AssertionError("cached decode called before prefill")
        token_id = int(token_id)
        offset = int(self.logical_length)
        full_ids = np.concatenate([np.asarray(prefix_input_ids, dtype=np.int64), np.asarray([token_id], dtype=np.int64)])
        full_mm = np.concatenate([np.asarray(prefix_mm_token_type_ids, dtype=np.int64), np.asarray([0], dtype=np.int64)])
        cos_full, sin_full = self._cos_sin_for(full_ids, full_mm, self.image_grid_thw)
        cos_decode = cos_full[-1:]
        sin_decode = sin_full[-1:]
        if self.gpu_tensor_chain:
            hidden = self.garnet.embedding(self.gpu_embed_tokens, np.asarray([token_id], dtype=np.int64))
        else:
            hidden = self.embed_tokens[np.asarray([token_id], dtype=np.int64)].astype(np.float32)
        cache_dir = self.cache_root / f"cached_text_decode_one_layers_{self.cached_text_layer_count}"

        if self.gpu_tensor_chain and self.cpp_text_runner_enabled:
            hidden = self._cpp_text_runner(1).decode(
                hidden, cos_decode, sin_decode, self.layer_caches, offset, offset + 1
            )
            if hidden is None:
                raise AssertionError("C++ QwenTextRunner decode failed")
        else:
            for layer_idx, handle in enumerate(self.layer_caches):
                if self.gpu_tensor_chain:
                    hidden = self._gpu_layer_forward(
                        layer_idx, hidden, cos_decode, sin_decode, handle, offset, offset + 1, False
                    )
                    continue
                layer_weights = load_text_layer_weights_as_layer0_aliases(self.key_to_file, layer_idx)
                hidden = run_garnet_text_decoder_layer_decode_with_device_paged_kv(
                    self.garnet,
                    self.dll,
                    layer_weights,
                    hidden,
                    cos_decode,
                    sin_decode,
                    cache_dir / f"layer_{layer_idx}",
                    handle,
                    offset,
                )

        self.logical_length += 1
        if self.gpu_tensor_chain:
            next_token_id = self._gpu_sample_hidden(hidden, 1)
            logits = None
        else:
            next_token_id = None
            logits = self._logits_from_hidden(hidden, cache_dir)
        return {
            "logits": logits,
            "next_token_id": next_token_id,
            "hidden_states": hidden,
            "text_layer_count": self.cached_text_layer_count,
            "full_text_layer_count": self.text_layer_count,
            "cached_decode": True,
            "kv_logical_length": int(self.logical_length),
        }


def run_model_forward_facade_against_hf_modules(garnet, model_dir, key_to_file):
    torch, config, _, rotary, create_causal_mask = load_hf_text_layer(model_dir, key_to_file, 0)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception:
        tokenizer = None

    token_count = int(os.environ.get("GARNET_QWEN_MODEL_FORWARD_TOKEN_WINDOW", "12"))
    processor_npz = latest_processor_npz()
    arrays = np.load(processor_npz)
    input_ids = arrays["input_ids"][0, :token_count].astype(np.int64)
    mm_types = arrays["mm_token_type_ids"][0, :token_count].astype(np.int64)
    image_grid_thw = arrays["image_grid_thw"].astype(np.int64)
    pixel_values = arrays["pixel_values"].astype(np.float32)

    window = make_processor_window_embeddings(garnet, model_dir, key_to_file, token_count, True)
    expected_x = window["expected_x"]
    embed_tokens = window["embed_tokens"]
    final_norm = load_tensor(key_to_file, "language_model.norm.weight")
    layer_count = text_layer_count_from_weights(key_to_file)

    with torch.inference_mode():
        hidden = torch.from_numpy(expected_x).unsqueeze(0)
        position_ids = torch.stack([
            torch.arange(expected_x.shape[0], dtype=torch.long),
            torch.arange(expected_x.shape[0], dtype=torch.long) + 17,
            torch.arange(expected_x.shape[0], dtype=torch.long) + 31,
        ], dim=0).unsqueeze(1)
        text_position_ids = torch.arange(expected_x.shape[0], dtype=torch.long).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=config,
            inputs_embeds=hidden,
            attention_mask=torch.ones(1, expected_x.shape[0], dtype=torch.long),
            past_key_values=None,
            position_ids=text_position_ids,
        )
        cos, sin = rotary(hidden, position_ids)
        cos_np = cos.squeeze(0).detach().cpu().numpy().astype(np.float32)
        sin_np = sin.squeeze(0).detach().cpu().numpy().astype(np.float32)
        for layer_idx in range(layer_count):
            _, _, layer, _, _ = load_hf_text_layer(model_dir, key_to_file, layer_idx)
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=(cos, sin),
            )
            del layer
        expected_hidden = hidden.squeeze(0).detach().cpu().numpy().astype(np.float32)

    expected_norm = expected_hidden / np.sqrt(np.mean(np.square(expected_hidden), axis=-1, keepdims=True) + 1.0e-6)
    expected_norm = expected_norm * final_norm
    expected_logits = expected_norm @ embed_tokens.T

    facade = GarnetQwen3VLForwardFacade(
        garnet,
        model_dir,
        key_to_file,
        SCRIPT_DIR / "cache_qwen3vl_model_forward_facade" / f"tokens_{token_count}",
        use_all_vision_blocks=True,
    )
    actual = facade.forward(input_ids, pixel_values, image_grid_thw, mm_types, cos_np, sin_np)
    actual_logits = actual["logits"]

    expected_top = np.argsort(expected_logits[-1])[-10:][::-1]
    actual_top = np.argsort(actual_logits[-1])[-10:][::-1]
    overlap = len(set(int(x) for x in expected_top[:10]) & set(int(x) for x in actual_top[:10]))
    expected_next_id = int(expected_top[0])
    actual_next_id = int(actual_top[0])
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=3e-1, atol=8.0)
    if overlap < 8:
        raise AssertionError(f"model forward facade top-10 token overlap too low: {overlap}/10")
    return {
        "processor_npz": str(processor_npz),
        "input_ids": [int(x) for x in input_ids.tolist()],
        "visual_positions": [int(x) for x in actual["visual_positions"].tolist()],
        "grid_thw": image_grid_thw.tolist(),
        "vision_block_count": int(actual["vision_block_count"]),
        "text_layer_count": int(actual["text_layer_count"]),
        "logits_shape": list(actual_logits.shape),
        "max_error": float(np.max(np.abs(actual_logits - expected_logits))),
        "mean_error": float(np.mean(np.abs(actual_logits - expected_logits))),
        "last_token_top10_overlap": overlap,
        "expected_next_token_id": expected_next_id,
        "actual_next_token_id": actual_next_id,
        "expected_next_token_text": tokenizer.decode([expected_next_id]) if tokenizer is not None else "",
        "actual_next_token_text": tokenizer.decode([actual_next_id]) if tokenizer is not None else "",
        "expected_top10_ids": [int(x) for x in expected_top.tolist()],
        "actual_top10_ids": [int(x) for x in actual_top.tolist()],
        "facade_inputs": ["input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids", "cos", "sin"],
    }


def run_model_forward_facade_native_rope_against_hf_modules(garnet, model_dir, key_to_file):
    torch, config, _, _, create_causal_mask = load_hf_text_layer(model_dir, key_to_file, 0)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception:
        tokenizer = None

    processor_npz = latest_processor_npz()
    arrays = np.load(processor_npz)
    input_ids = np.asarray([151644, 872, 198, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151645, 198], dtype=np.int64)
    mm_types = np.asarray([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=np.int64)
    image_grid_thw = np.asarray([[1, 4, 8]], dtype=np.int64)
    pixel_values = arrays["pixel_values"][:32].astype(np.float32)

    facade = GarnetQwen3VLForwardFacade(
        garnet,
        model_dir,
        key_to_file,
        SCRIPT_DIR / "cache_qwen3vl_model_forward_native_rope_facade",
        use_all_vision_blocks=True,
    )

    position_ids, _ = qwen3vl_mrope_position_ids_numpy(
        input_ids[None, :],
        mm_types[None, :],
        image_grid_thw=image_grid_thw,
        attention_mask=np.ones((1, input_ids.shape[0]), dtype=np.int64),
        spatial_merge_size=facade.config.vision_config.spatial_merge_size,
    )
    cos, sin = qwen3vl_text_mrope_cos_sin_numpy(
        position_ids,
        head_dim=facade.config.text_config.head_dim,
        rope_theta=facade.config.text_config.rope_parameters["rope_theta"],
        mrope_section=facade.config.text_config.rope_parameters.get("mrope_section", [24, 20, 20]),
    )

    visual_positions = np.flatnonzero(mm_types == 1)
    visual_tokens = facade._visual_tokens(pixel_values, image_grid_thw, int(visual_positions.size))
    expected_x = facade.embed_tokens[input_ids].astype(np.float32)
    expected_x[visual_positions] = visual_tokens

    with torch.inference_mode():
        hidden = torch.from_numpy(expected_x).unsqueeze(0)
        text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=config,
            inputs_embeds=hidden,
            attention_mask=torch.ones(1, input_ids.shape[0], dtype=torch.long),
            past_key_values=None,
            position_ids=text_position_ids,
        )
        cos_t = torch.from_numpy(cos.astype(np.float32))
        sin_t = torch.from_numpy(sin.astype(np.float32))
        for layer_idx in range(facade.text_layer_count):
            _, _, layer, _, _ = load_hf_text_layer(model_dir, key_to_file, layer_idx)
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=(cos_t, sin_t),
            )
            del layer
        expected_hidden = hidden.squeeze(0).detach().cpu().numpy().astype(np.float32)

    expected_norm = expected_hidden / np.sqrt(np.mean(np.square(expected_hidden), axis=-1, keepdims=True) + 1.0e-6)
    expected_norm = expected_norm * facade.final_norm
    expected_logits = expected_norm @ facade.embed_tokens.T

    actual = facade.forward(input_ids, pixel_values, image_grid_thw, mm_types)
    actual_logits = actual["logits"]
    expected_top = np.argsort(expected_logits[-1])[-10:][::-1]
    actual_top = np.argsort(actual_logits[-1])[-10:][::-1]
    overlap = len(set(int(x) for x in expected_top[:10]) & set(int(x) for x in actual_top[:10]))
    expected_next_id = int(expected_top[0])
    actual_next_id = int(actual_top[0])
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=3e-1, atol=8.0)
    if overlap < 8:
        raise AssertionError(f"native-rope facade top-10 token overlap too low: {overlap}/10")
    return {
        "processor_npz": str(processor_npz),
        "input_ids": [int(x) for x in input_ids.tolist()],
        "visual_positions": [int(x) for x in visual_positions.tolist()],
        "image_grid_thw": image_grid_thw.tolist(),
        "position_ids_shape": list(position_ids.shape),
        "cos_shape": list(cos.shape),
        "sin_shape": list(sin.shape),
        "vision_block_count": int(facade.vision_block_count),
        "text_layer_count": int(facade.text_layer_count),
        "logits_shape": list(actual_logits.shape),
        "max_error": float(np.max(np.abs(actual_logits - expected_logits))),
        "mean_error": float(np.mean(np.abs(actual_logits - expected_logits))),
        "last_token_top10_overlap": overlap,
        "expected_next_token_id": expected_next_id,
        "actual_next_token_id": actual_next_id,
        "expected_next_token_text": tokenizer.decode([expected_next_id]) if tokenizer is not None else "",
        "actual_next_token_text": tokenizer.decode([actual_next_id]) if tokenizer is not None else "",
        "expected_top10_ids": [int(x) for x in expected_top.tolist()],
        "actual_top10_ids": [int(x) for x in actual_top.tolist()],
        "native_rope": True,
    }


def run_model_forward_facade_native_rope_decode(garnet, model_dir, key_to_file, serving_state=None):
    frontend_start = time.perf_counter()
    use_garnet_tokenizer = env_flag("GARNET_QWEN_NATIVE_ROPE_DECODE_GARNET_TOKENIZER")
    tokenizer = None
    if not use_garnet_tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        except Exception as exc:
            skip(f"AutoTokenizer is required for native-rope decode smoke test: {exc}")

    user_prompt = os.environ.get("GARNET_QWEN_NATIVE_ROPE_DECODE_PROMPT", "Detect visible objects. Return short JSON.")
    processor_npz = latest_processor_npz()
    image_path = None
    processor_pixels = None
    native_decode = None
    native_token_id = None
    if env_flag("GARNET_QWEN_NATIVE_ROPE_DECODE_ONE_CALL_PREPARE"):
        image_path = Path(os.environ.get(
            "GARNET_QWEN_NATIVE_ROPE_DECODE_IMAGE",
            REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg",
        ))
        processor_pixels_value = os.environ.get("GARNET_QWEN_NATIVE_ROPE_DECODE_PROCESSOR_PIXELS")
        if processor_pixels_value is None or processor_pixels_value.lower() in {"", "default", "original"}:
            processor_pixels = 65536
        elif processor_pixels_value.lower().startswith("max:"):
            processor_pixels = int(processor_pixels_value.split(":", 1)[1])
        else:
            processor_pixels = int(processor_pixels_value)
        prepared = prepare_qwen_vl_request_one_call(model_dir, image_path, user_prompt, processor_pixels)
        input_ids = prepared["input_ids"]
        mm_types = prepared["mm_types"]
        image_grid_thw = prepared["image_grid_thw"]
        pixel_values = prepared["pixel_values"]
        native_token_id = prepared["native_token_id"]
        native_decode = prepared["native_decode"]
        processor_npz = prepared["processor_npz"]
    elif env_flag("GARNET_QWEN_NATIVE_ROPE_DECODE_PROCESSOR_IMAGE"):
        use_garnet_image_preprocess = env_flag("GARNET_QWEN_NATIVE_ROPE_DECODE_GARNET_IMAGE_PREPROCESS")
        try:
            from PIL import Image
        except Exception as exc:
            skip(f"whole-image decode requires PIL: {exc}")
        if not use_garnet_image_preprocess:
            try:
                import torch
                from transformers import AutoProcessor
            except Exception as exc:
                skip(f"HF processor decode requires torch/AutoProcessor: {exc}")
        image_path = Path(os.environ.get(
            "GARNET_QWEN_NATIVE_ROPE_DECODE_IMAGE",
            REPO_ROOT / "data" / "Dataset.1980Love" / "imgs" / "frame_0.jpg",
        ))
        processor_pixels_value = os.environ.get("GARNET_QWEN_NATIVE_ROPE_DECODE_PROCESSOR_PIXELS")
        if processor_pixels_value is None or processor_pixels_value.lower() in {"", "default", "original"}:
            processor = None
            if not use_garnet_image_preprocess:
                processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
            processor_pixels = None
        elif processor_pixels_value.lower().startswith("max:"):
            processor_pixels = int(processor_pixels_value.split(":", 1)[1])
            processor = None
            if not use_garnet_image_preprocess:
                processor = AutoProcessor.from_pretrained(
                    str(model_dir),
                    trust_remote_code=True,
                    max_pixels=processor_pixels,
                )
        else:
            processor_pixels = int(processor_pixels_value)
            processor = None
            if not use_garnet_image_preprocess:
                processor = AutoProcessor.from_pretrained(
                    str(model_dir),
                    trust_remote_code=True,
                    min_pixels=processor_pixels,
                    max_pixels=processor_pixels,
                )
        image = Image.open(image_path).convert("RGB")
        def tensor_to_numpy(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)
        if use_garnet_image_preprocess:
            if processor_pixels is None:
                processor_pixels = 65536
            import ctypes
            import math
            use_garnet_jpeg_preprocess = env_flag("GARNET_QWEN_NATIVE_ROPE_DECODE_GARNET_JPEG_PREPROCESS")
            factor = 32
            resized_h = round(image.height / factor) * factor
            resized_w = round(image.width / factor) * factor
            if resized_h * resized_w > processor_pixels:
                beta = math.sqrt((image.height * image.width) / processor_pixels)
                resized_h = max(factor, math.floor(image.height / beta / factor) * factor)
                resized_w = max(factor, math.floor(image.width / beta / factor) * factor)
            elif resized_h * resized_w < processor_pixels:
                beta = math.sqrt(processor_pixels / (image.height * image.width))
                resized_h = math.ceil(image.height * beta / factor) * factor
                resized_w = math.ceil(image.width * beta / factor) * factor
            garnet_dll = Path(os.environ.get(
                "GARNET_DLL_PATH",
                REPO_ROOT / "out" / "build" / "x64-Debug" / "bin" / "garnet.dll",
            ))
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                for dll_dir in [
                    garnet_dll.parent,
                    REPO_ROOT.parent / "xlang" / "out" / "build" / "x64-Debug" / "bin",
                    REPO_ROOT.parent / "out" / "build" / "x64-Debug" / "bin",
                    REPO_ROOT.parent / "out" / "build" / "x64-debug" / "bin",
                    Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin"),
                    Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64"),
                    REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "bin",
                    REPO_ROOT.parent / "ThirdPartySDK" / "TensorRT" / "lib",
                    Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Redist/MSVC/14.51.36231/debug_nonredist/x64/Microsoft.VC145.DebugCRT"),
                ]:
                    if dll_dir.exists():
                        os.add_dll_directory(str(dll_dir))
            dll = ctypes.CDLL(str(garnet_dll))
            if use_garnet_tokenizer:
                dll.GarnetQwenVLBuildSingleImagePromptIds.argtypes = [
                    ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int,
                    ctypes.POINTER(ctypes.c_longlong), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                    ctypes.c_char_p, ctypes.c_int,
                ]
                dll.GarnetQwenVLBuildSingleImagePromptIds.restype = ctypes.c_int
                dll.GarnetQwenTokenizerDecode.argtypes = [
                    ctypes.c_char_p, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int, ctypes.c_int,
                    ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int,
                ]
                dll.GarnetQwenTokenizerDecode.restype = ctypes.c_int
                dll.GarnetQwenTokenizerTokenId.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
                dll.GarnetQwenTokenizerTokenId.restype = ctypes.c_longlong

                def _native_token_id(token):
                    return int(dll.GarnetQwenTokenizerTokenId(str(model_dir).encode("utf-8"), token.encode("utf-8")))

                def _native_decode(ids, skip_special=True):
                    arr = (ctypes.c_longlong * len(ids))(*[int(x) for x in ids])
                    byte_count = ctypes.c_int(0)
                    decode_error = ctypes.create_string_buffer(512)
                    rc_decode = dll.GarnetQwenTokenizerDecode(
                        str(model_dir).encode("utf-8"), arr, len(ids), 1 if skip_special else 0,
                        None, 0, ctypes.byref(byte_count), decode_error, len(decode_error)
                    )
                    if rc_decode not in (0, 3):
                        raise AssertionError(f"GarnetQwenTokenizerDecode count failed rc={rc_decode}: {decode_error.value.decode(errors='ignore')}")
                    text_buf = ctypes.create_string_buffer(byte_count.value + 1)
                    rc_decode = dll.GarnetQwenTokenizerDecode(
                        str(model_dir).encode("utf-8"), arr, len(ids), 1 if skip_special else 0,
                        text_buf, len(text_buf), ctypes.byref(byte_count), decode_error, len(decode_error)
                    )
                    if rc_decode != 0:
                        raise AssertionError(f"GarnetQwenTokenizerDecode failed rc={rc_decode}: {decode_error.value.decode(errors='ignore')}")
                    return text_buf.value.decode("utf-8", errors="replace")

                native_token_id = _native_token_id
                native_decode = _native_decode
            patch_count = (resized_h // 16) * (resized_w // 16)
            feature_dim = 3 * 2 * 16 * 16
            pixel_values = np.empty((patch_count, feature_dim), dtype=np.float32)
            image_grid_thw = np.zeros((1, 3), dtype=np.int64)
            source_h = ctypes.c_int(0)
            source_w = ctypes.c_int(0)
            out_h = ctypes.c_int(0)
            out_w = ctypes.c_int(0)
            error = ctypes.create_string_buffer(512)
            if use_garnet_jpeg_preprocess:
                dll.GarnetQwenVLPreprocessJpegFile.argtypes = [
                    ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_longlong),
                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                    ctypes.c_char_p, ctypes.c_int,
                ]
                dll.GarnetQwenVLPreprocessJpegFile.restype = ctypes.c_int
                rc = dll.GarnetQwenVLPreprocessJpegFile(
                    str(image_path).encode("utf-8"),
                    processor_pixels,
                    processor_pixels,
                    pixel_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    image_grid_thw.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
                    ctypes.byref(source_h),
                    ctypes.byref(source_w),
                    ctypes.byref(out_h),
                    ctypes.byref(out_w),
                    error,
                    len(error),
                )
            else:
                raw_rgb = np.ascontiguousarray(np.asarray(image, dtype=np.float32))
                dll.GarnetQwenVLResizePreprocessRGBF32.argtypes = [
                    ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_longlong),
                    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                    ctypes.c_char_p, ctypes.c_int,
                ]
                dll.GarnetQwenVLResizePreprocessRGBF32.restype = ctypes.c_int
                rc = dll.GarnetQwenVLResizePreprocessRGBF32(
                    raw_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    image.height,
                    image.width,
                    3,
                    0,
                    processor_pixels,
                    processor_pixels,
                    ctypes.c_float(255.0),
                    pixel_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    image_grid_thw.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
                    ctypes.byref(out_h),
                    ctypes.byref(out_w),
                    error,
                    len(error),
                )
            if rc != 0:
                api_name = "GarnetQwenVLPreprocessJpegFile" if use_garnet_jpeg_preprocess else "GarnetQwenVLResizePreprocessRGBF32"
                raise AssertionError(f"{api_name} failed rc={rc}: {error.value.decode(errors='ignore')}")
            expected_visual_count = int(np.prod(image_grid_thw[0]) // 4)
            if use_garnet_tokenizer:
                prompt_count = ctypes.c_int(0)
                prompt_error = ctypes.create_string_buffer(512)
                grid_arr = (ctypes.c_longlong * 3)(*image_grid_thw[0].astype(np.int64).tolist())
                rc_prompt = dll.GarnetQwenVLBuildSingleImagePromptIds(
                    str(model_dir).encode("utf-8"), user_prompt.encode("utf-8"), grid_arr, 2,
                    None, 0, ctypes.byref(prompt_count), prompt_error, len(prompt_error)
                )
                if rc_prompt not in (0, 3):
                    raise AssertionError(f"GarnetQwenVLBuildSingleImagePromptIds count failed rc={rc_prompt}: {prompt_error.value.decode(errors='ignore')}")
                prompt_ids = (ctypes.c_longlong * prompt_count.value)()
                rc_prompt = dll.GarnetQwenVLBuildSingleImagePromptIds(
                    str(model_dir).encode("utf-8"), user_prompt.encode("utf-8"), grid_arr, 2,
                    prompt_ids, prompt_count.value, ctypes.byref(prompt_count), prompt_error, len(prompt_error)
                )
                if rc_prompt != 0:
                    raise AssertionError(f"GarnetQwenVLBuildSingleImagePromptIds failed rc={rc_prompt}: {prompt_error.value.decode(errors='ignore')}")
                input_ids = np.asarray([int(prompt_ids[i]) for i in range(prompt_count.value)], dtype=np.int64)
                prompt_text = "garnet_native_qwen_vl_prompt"
                image_pad_id = native_token_id("<|image_pad|>")
            else:
                prompt_text = (
                    "<|im_start|>user\n"
                    "<|vision_start|>"
                    + "<|image_pad|>" * expected_visual_count
                    + "<|vision_end|>\n"
                    + user_prompt
                    + "\n"
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                input_ids = np.asarray(tokenizer.encode(prompt_text, add_special_tokens=False), dtype=np.int64)
                image_pad_id = tokenizer.encode("<|image_pad|>", add_special_tokens=False)[0]
            mm_types = np.zeros_like(input_ids, dtype=np.int64)
            mm_types[input_ids == image_pad_id] = 1
            processor_npz = "garnet_nvjpeg_cuda_image_preprocess" if use_garnet_jpeg_preprocess else "garnet_cuda_image_preprocess"
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            }]
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
            input_ids = tensor_to_numpy(inputs["input_ids"])[0].astype(np.int64)
            mm_types = tensor_to_numpy(inputs["mm_token_type_ids"])[0].astype(np.int64)
            image_grid_thw = tensor_to_numpy(inputs["image_grid_thw"]).astype(np.int64)
            pixel_values = tensor_to_numpy(inputs["pixel_values"]).astype(np.float32)
    else:
        arrays = np.load(processor_npz)
        image_grid_thw = np.asarray([[1, 4, 8]], dtype=np.int64)
        pixel_values = arrays["pixel_values"][:32].astype(np.float32)
        prompt_text = (
            "<|im_start|>user\n"
            "<|vision_start|>"
            + "<|image_pad|>" * 8
            + "<|vision_end|>\n"
            + user_prompt
            + "\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        input_ids = np.asarray(tokenizer.encode(prompt_text, add_special_tokens=False), dtype=np.int64)
        mm_types = np.zeros_like(input_ids, dtype=np.int64)
        image_pad_id = tokenizer.encode("<|image_pad|>", add_special_tokens=False)[0]
        mm_types[input_ids == image_pad_id] = 1
    visual_count = int(np.sum(mm_types == 1))
    expected_visual_count = int(np.prod(image_grid_thw[0]) // 4)
    if visual_count != expected_visual_count:
        raise AssertionError(f"visual placeholder count {visual_count} != reduced-grid token count {expected_visual_count}")
    frontend_prepare_ms = (time.perf_counter() - frontend_start) * 1000.0

    max_new_tokens = int(os.environ.get("GARNET_QWEN_NATIVE_ROPE_DECODE_TOKENS", "2"))
    use_cached_text_decode = env_flag("GARNET_QWEN_NATIVE_ROPE_DECODE_USE_CACHED_TEXT")
    cached_text_layer_count_value = os.environ.get("GARNET_QWEN_NATIVE_ROPE_DECODE_TEXT_LAYER_COUNT", "").strip().lower()
    cached_text_layer_count = None if cached_text_layer_count_value in {"", "all", "full"} else int(cached_text_layer_count_value)
    facade_cls = GarnetQwen3VLCachedForwardFacade if use_cached_text_decode else GarnetQwen3VLForwardFacade
    facade_kwargs = {
        "use_all_vision_blocks": True,
    }
    if use_cached_text_decode:
        facade_kwargs.update({
            "max_new_tokens": max_new_tokens,
            "page_size": int(os.environ.get("GARNET_QWEN_NATIVE_ROPE_DECODE_TEXT_KV_PAGE_SIZE", "16")),
            "text_layer_count": cached_text_layer_count,
        })
    facade = serving_state.get("facade") if serving_state is not None else None
    if facade is None:
        facade = facade_cls(
            garnet,
            model_dir,
            key_to_file,
            SCRIPT_DIR / "cache_qwen3vl_model_forward_native_rope_decode",
            **facade_kwargs,
        )
        if serving_state is not None:
            serving_state["facade"] = facade
    generated = []
    step_summaries = []
    cur_ids = input_ids.copy()
    cur_mm_types = mm_types.copy()
    eos_id = native_token_id("<|im_end|>") if use_garnet_tokenizer else tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    kv_manager = None
    kv_prefill_pages = []
    kv_stats_before_free = None
    kv_stats_after_free = None
    kv_free_result = None
    kv_sequence_id = int(os.environ.get("GARNET_QWEN_NATIVE_ROPE_DECODE_SEQUENCE_ID", "1"))
    if env_flag("GARNET_QWEN_NATIVE_ROPE_DECODE_USE_KV_MANAGER"):
        manager_page_size = int(os.environ.get("GARNET_QWEN_KV_PAGE_SIZE", "16"))
        request_pages = (int(cur_ids.shape[0]) + max_new_tokens + manager_page_size - 1) // manager_page_size
        kv_manager = garnet.KVCacheManager(
            max_num_pages=int(os.environ.get("GARNET_QWEN_KV_MAX_PAGES", str(request_pages + 2))),
            page_size=manager_page_size,
            head_dim=int(os.environ.get("GARNET_QWEN_KV_HEAD_DIM", "128")),
            num_kv_heads=int(os.environ.get("GARNET_QWEN_KV_HEADS", "8")),
            num_layers=int(facade.text_layer_count),
            dtype_bytes=int(os.environ.get("GARNET_QWEN_KV_DTYPE_BYTES", "2")),
            device_id=int(os.environ.get("GARNET_QWEN_KV_DEVICE_ID", "0")),
        )
        kv_prefill_pages = [int(x) for x in kv_manager.allocate(kv_sequence_id, sequence_length=int(cur_ids.shape[0]))]
    timing = {}
    timing["frontend_prepare_ms"] = frontend_prepare_ms
    visual_positions = np.flatnonzero(mm_types == 1)
    visual_token_start = time.perf_counter()
    visual_tokens = facade._visual_tokens(pixel_values, image_grid_thw, int(visual_positions.size)) if visual_positions.size else None
    timing["visual_tokens_once_ms"] = (time.perf_counter() - visual_token_start) * 1000.0
    if getattr(facade, "cpp_vision_runner_enabled", False):
        timing["vision_runner_prepare_ms"] = float(getattr(facade, "vision_runner_prepare_ms", 0.0))
        timing["vision_runner_warmup_ms"] = float(getattr(facade, "vision_runner_warmup_ms", 0.0))
        timing["vision_patch_ms"] = float(getattr(facade, "vision_patch_ms", 0.0))
        timing["vision_execute_ms"] = float(getattr(facade, "vision_execute_ms", 0.0))
    # Build the prefill runner lazily inside prefill, after the shared embedding
    # table is resident. Eager runner construction can consume the remaining
    # device memory and make the embedding upload fail on 16 GB cards.
    timing["text_runner_prepare_ms"] = 0.0
    decode_start = time.perf_counter()
    if use_cached_text_decode:
        try:
            prefill_start = time.perf_counter()
            prefill_out = facade.prefill(
                cur_ids,
                pixel_values,
                image_grid_thw,
                cur_mm_types,
                visual_tokens_override=visual_tokens,
            )
            timing["text_prefill_ms"] = (time.perf_counter() - prefill_start) * 1000.0
            logits = prefill_out["logits"]
            sampled_token_id = prefill_out.get("next_token_id")
            for step in range(max_new_tokens):
                step_start = time.perf_counter()
                if step > 0:
                    decode_out = facade.decode_one(int(cur_ids[-1]), cur_ids[:-1], cur_mm_types[:-1])
                    logits = decode_out["logits"]
                    sampled_token_id = decode_out.get("next_token_id")
                if sampled_token_id is not None:
                    next_id = int(sampled_token_id)
                    top5 = np.asarray([next_id], dtype=np.int64)
                else:
                    next_id = int(np.argmax(logits[-1]))
                    top5 = np.argsort(logits[-1])[-5:][::-1]
                generated.append(next_id)
                step_summaries.append({
                    "step": int(step),
                    "input_tokens": int(cur_ids.shape[0]),
                    "next_token_id": next_id,
                    "next_token_text": native_decode([next_id], False) if use_garnet_tokenizer else tokenizer.decode([next_id]),
                    "top5_ids": [int(x) for x in top5.tolist()],
                    "step_ms": (time.perf_counter() - step_start) * 1000.0,
                    "mode": "prefill_logits" if step == 0 else "cached_decode_one",
                })
                cur_ids = np.concatenate([cur_ids, np.asarray([next_id], dtype=np.int64)])
                cur_mm_types = np.concatenate([cur_mm_types, np.asarray([0], dtype=np.int64)])
                if kv_manager is not None:
                    kv_manager.append(kv_sequence_id, tokens=1)
                if next_id == eos_id:
                    break
                if "json" in user_prompt.lower():
                    partial_text = native_decode(generated, False) if use_garnet_tokenizer else tokenizer.decode(generated)
                    json_start = partial_text.find("{")
                    if json_start >= 0:
                        try:
                            json.JSONDecoder().raw_decode(partial_text[json_start:])
                            break
                        except json.JSONDecodeError:
                            pass
        finally:
            if serving_state is None:
                facade.close()
    else:
        for step in range(max_new_tokens):
            step_start = time.perf_counter()
            out = facade.forward(
                cur_ids,
                pixel_values,
                image_grid_thw,
                cur_mm_types,
                last_token_logits_only=True,
                visual_tokens_override=visual_tokens,
            )
            logits = out["logits"]
            next_id = int(np.argmax(logits[-1]))
            top5 = np.argsort(logits[-1])[-5:][::-1]
            generated.append(next_id)
            step_summaries.append({
                "step": int(step),
                "input_tokens": int(cur_ids.shape[0]),
                "next_token_id": next_id,
                "next_token_text": native_decode([next_id], False) if use_garnet_tokenizer else tokenizer.decode([next_id]),
                "top5_ids": [int(x) for x in top5.tolist()],
                "step_ms": (time.perf_counter() - step_start) * 1000.0,
                "mode": "full_sequence_recompute",
            })
            cur_ids = np.concatenate([cur_ids, np.asarray([next_id], dtype=np.int64)])
            cur_mm_types = np.concatenate([cur_mm_types, np.asarray([0], dtype=np.int64)])
            if kv_manager is not None:
                kv_manager.append(kv_sequence_id, tokens=1)
            if next_id == eos_id:
                break
    timing["decode_loop_ms"] = (time.perf_counter() - decode_start) * 1000.0
    if use_cached_text_decode:
        timing["decode_token_loop_ms"] = timing["decode_loop_ms"] - timing.get("text_prefill_ms", 0.0)
    timing["total_after_frontend_ms"] = (
        timing["visual_tokens_once_ms"]
        + timing.get("text_runner_prepare_ms", 0.0)
        + timing["decode_loop_ms"]
    )
    timing["steady_pipeline_ms"] = (
        timing.get("vision_patch_ms", timing["visual_tokens_once_ms"])
        + timing.get("vision_execute_ms", 0.0)
        + timing.get("text_prefill_ms", 0.0)
        + timing.get("decode_token_loop_ms", 0.0)
    )
    if kv_manager is not None:
        kv_stats_before_free = kv_manager.stats()
        kv_free_result = bool(kv_manager.free(kv_sequence_id))
        kv_stats_after_free = kv_manager.stats()

    generated_text = (native_decode(generated, True) if use_garnet_tokenizer else tokenizer.decode(generated, skip_special_tokens=True)).strip()
    raw_generated_text = native_decode(generated, False) if use_garnet_tokenizer else tokenizer.decode(generated, skip_special_tokens=False)
    if not raw_generated_text.strip():
        raise AssertionError("native-rope decode produced empty raw text")
    return {
        "processor_npz": str(processor_npz),
        "image": str(image_path) if image_path is not None else "",
        "processor_pixels": processor_pixels,
        "user_prompt": user_prompt,
        "prompt_token_count": int(input_ids.shape[0]),
        "pixel_values_shape": list(pixel_values.shape),
        "visual_token_count": visual_count,
        "image_grid_thw": image_grid_thw.tolist(),
        "max_new_tokens": max_new_tokens,
        "generated_token_ids": [int(x) for x in generated],
        "generated_text": generated_text,
        "raw_generated_text": raw_generated_text,
        "steps": step_summaries,
        "native_rope": True,
        "native_tokenizer": bool(use_garnet_tokenizer),
        "visual_tokens_cached_once": True,
        "text_decode_mode": "cached_prefill_decode" if use_cached_text_decode else "full_sequence_recompute",
        "gpu_tensor_chain": bool(use_cached_text_decode and getattr(facade, "gpu_tensor_chain", False)),
        "cpp_text_runner": bool(use_cached_text_decode and getattr(facade, "cpp_text_runner_enabled", False)),
        "cpp_vision_runner": bool(use_cached_text_decode and getattr(facade, "cpp_vision_runner_enabled", False)),
        "kv_cache_manager": {
            "enabled": kv_manager is not None,
            "sequence_id": kv_sequence_id if kv_manager is not None else None,
            "prefill_pages": kv_prefill_pages,
            "stats_before_free": kv_stats_before_free,
            "free_result": kv_free_result,
            "stats_after_free": kv_stats_after_free,
        },
        "timing_ms": timing,
        "vision_block_count": int(facade.vision_block_count),
        "text_layer_count": int(facade.cached_text_layer_count if use_cached_text_decode else facade.text_layer_count),
        "full_text_layer_count": int(facade.text_layer_count),
    }


def run_two_layer_logits_against_hf_modules(garnet, model_dir, key_to_file):
    torch, config, layer0, rotary, create_causal_mask = load_hf_text_layer(model_dir, key_to_file, 0)
    _, _, layer1, _, _ = load_hf_text_layer(model_dir, key_to_file, 1)
    rng = np.random.default_rng(20260718)
    x = rng.normal(0.0, 0.02, size=(4, config.hidden_size)).astype(np.float32)
    final_norm = load_tensor(key_to_file, "language_model.norm.weight")
    embed_tokens = load_tensor(key_to_file, "language_model.embed_tokens.weight")

    with torch.inference_mode():
        hx = torch.from_numpy(x).unsqueeze(0)
        position_ids = torch.stack([
            torch.arange(x.shape[0], dtype=torch.long),
            torch.arange(x.shape[0], dtype=torch.long) + 17,
            torch.arange(x.shape[0], dtype=torch.long) + 31,
        ], dim=0).unsqueeze(1)
        text_position_ids = torch.arange(x.shape[0], dtype=torch.long).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=config,
            inputs_embeds=hx,
            attention_mask=torch.ones(1, x.shape[0], dtype=torch.long),
            past_key_values=None,
            position_ids=text_position_ids,
        )
        cos, sin = rotary(hx, position_ids)
        expected_hidden = layer0(
            hx,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        )
        expected_hidden = layer1(
            expected_hidden,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        ).squeeze(0).detach().cpu().numpy().astype(np.float32)
        cos_np = cos.squeeze(0).detach().cpu().numpy().astype(np.float32)
        sin_np = sin.squeeze(0).detach().cpu().numpy().astype(np.float32)

    expected_norm = expected_hidden / np.sqrt(np.mean(np.square(expected_hidden), axis=-1, keepdims=True) + 1.0e-6)
    expected_norm = expected_norm * final_norm
    expected_logits = expected_norm @ embed_tokens.T

    actual = x
    cache_dir = SCRIPT_DIR / "cache_two_layer_logits_hf"
    for layer_idx in range(2):
        layer_weights = load_text_layer_weights_as_layer0_aliases(key_to_file, layer_idx)
        actual = run_garnet_text_decoder_layer(garnet, layer_weights, actual, cos_np, sin_np, cache_dir)

    final_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={"language_model.layers.0.input_layernorm.weight": final_norm},
        cache_dir=str(cache_dir),
        input_shapes=[list(actual.shape)],
        subgraph="rms_norm",
    )
    actual_norm = to_numpy(final_norm_model.forward(actual)).reshape(actual.shape)

    lm_head_model = garnet.load_model(
        str(SCRIPT_DIR / "text_lm_head_trt.x"),
        weights={"language_model.embed_tokens.weight": embed_tokens},
        cache_dir=str(cache_dir),
        input_shapes=[list(actual_norm.shape)],
        subgraph="text_lm_head",
    )
    actual_logits = to_numpy(lm_head_model.forward(actual_norm)).reshape(expected_logits.shape)

    expected_top = np.argsort(expected_logits[-1])[-10:][::-1]
    actual_top = np.argsort(actual_logits[-1])[-10:][::-1]
    overlap = len(set(int(x) for x in expected_top[:10]) & set(int(x) for x in actual_top[:10]))
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=8e-2, atol=8e-1)
    if overlap < 8:
        raise AssertionError(f"top-10 token overlap too low: {overlap}/10")
    return {
        "layers": 2,
        "input_shape": list(x.shape),
        "hidden_shape": list(actual.shape),
        "logits_shape": list(actual_logits.shape),
        "max_error": float(np.max(np.abs(actual_logits - expected_logits))),
        "mean_error": float(np.mean(np.abs(actual_logits - expected_logits))),
        "last_token_top10_overlap": overlap,
    }


def latest_processor_npz():
    artifact_dir = REPO_ROOT / "test2026" / "artifacts" / "qwen_vl_reference"
    explicit = os.environ.get("GARNET_QWEN_VL_PROCESSOR_DUMP_NPZ", "").strip()
    if explicit:
        return Path(explicit)
    candidates = sorted(artifact_dir.glob("processor_*_objects_json.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(artifact_dir.glob("processor_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        skip(f"no processor dump NPZ found in {artifact_dir}")
    return candidates[0]


def run_processor_token_logits_against_hf_modules(garnet, model_dir, key_to_file):
    torch, config, layer0, rotary, create_causal_mask = load_hf_text_layer(model_dir, key_to_file, 0)
    _, _, layer1, _, _ = load_hf_text_layer(model_dir, key_to_file, 1)
    processor_npz = latest_processor_npz()
    arrays = np.load(processor_npz)
    if "input_ids" not in arrays:
        raise AssertionError(f"processor dump missing input_ids: {processor_npz}")
    token_count = int(os.environ.get("GARNET_QWEN_TOKEN_WINDOW", "8"))
    input_ids = arrays["input_ids"][0, :token_count].astype(np.int64)
    embed_tokens = load_tensor(key_to_file, "language_model.embed_tokens.weight")
    final_norm = load_tensor(key_to_file, "language_model.norm.weight")
    x = embed_tokens[input_ids].astype(np.float32)

    with torch.inference_mode():
        hx = torch.from_numpy(x).unsqueeze(0)
        position_ids = torch.stack([
            torch.arange(x.shape[0], dtype=torch.long),
            torch.arange(x.shape[0], dtype=torch.long) + 17,
            torch.arange(x.shape[0], dtype=torch.long) + 31,
        ], dim=0).unsqueeze(1)
        text_position_ids = torch.arange(x.shape[0], dtype=torch.long).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=config,
            inputs_embeds=hx,
            attention_mask=torch.ones(1, x.shape[0], dtype=torch.long),
            past_key_values=None,
            position_ids=text_position_ids,
        )
        cos, sin = rotary(hx, position_ids)
        expected_hidden = layer0(
            hx,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        )
        expected_hidden = layer1(
            expected_hidden,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        ).squeeze(0).detach().cpu().numpy().astype(np.float32)
        cos_np = cos.squeeze(0).detach().cpu().numpy().astype(np.float32)
        sin_np = sin.squeeze(0).detach().cpu().numpy().astype(np.float32)

    expected_norm = expected_hidden / np.sqrt(np.mean(np.square(expected_hidden), axis=-1, keepdims=True) + 1.0e-6)
    expected_norm = expected_norm * final_norm
    expected_logits = expected_norm @ embed_tokens.T

    actual = x
    cache_dir = SCRIPT_DIR / "cache_processor_token_logits_hf"
    for layer_idx in range(2):
        layer_weights = load_text_layer_weights_as_layer0_aliases(key_to_file, layer_idx)
        actual = run_garnet_text_decoder_layer(garnet, layer_weights, actual, cos_np, sin_np, cache_dir)

    final_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={"language_model.layers.0.input_layernorm.weight": final_norm},
        cache_dir=str(cache_dir),
        input_shapes=[list(actual.shape)],
        subgraph="rms_norm",
    )
    actual_norm = to_numpy(final_norm_model.forward(actual)).reshape(actual.shape)

    lm_head_model = garnet.load_model(
        str(SCRIPT_DIR / "text_lm_head_trt.x"),
        weights={"language_model.embed_tokens.weight": embed_tokens},
        cache_dir=str(cache_dir),
        input_shapes=[list(actual_norm.shape)],
        subgraph="text_lm_head",
    )
    actual_logits = to_numpy(lm_head_model.forward(actual_norm)).reshape(expected_logits.shape)

    expected_top = np.argsort(expected_logits[-1])[-10:][::-1]
    actual_top = np.argsort(actual_logits[-1])[-10:][::-1]
    overlap = len(set(int(x) for x in expected_top[:10]) & set(int(x) for x in actual_top[:10]))
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-1, atol=1.0)
    if overlap < 8:
        raise AssertionError(f"processor token top-10 token overlap too low: {overlap}/10")
    return {
        "processor_npz": str(processor_npz),
        "layers": 2,
        "input_ids": [int(x) for x in input_ids.tolist()],
        "hidden_shape": list(actual.shape),
        "logits_shape": list(actual_logits.shape),
        "max_error": float(np.max(np.abs(actual_logits - expected_logits))),
        "mean_error": float(np.mean(np.abs(actual_logits - expected_logits))),
        "last_token_top10_overlap": overlap,
    }


def vision_patch_pos_merger_tokens_numpy(key_to_file, visual_token_count):
    processor_npz = latest_processor_npz()
    arrays = np.load(processor_npz)
    if "pixel_values" not in arrays or "image_grid_thw" not in arrays:
        raise AssertionError(f"processor dump missing pixel_values/image_grid_thw: {processor_npz}")
    patch_count = visual_token_count * 4
    pixel_values = arrays["pixel_values"][:patch_count].astype(np.float32)
    grid_thw = arrays["image_grid_thw"].astype(np.int64)
    patch_weight_5d = load_tensor(key_to_file, "visual.patch_embed.proj.weight")
    patch_weight = patch_weight_5d.reshape(patch_weight_5d.shape[0], -1).astype(np.float32)
    patch_bias = load_tensor(key_to_file, "visual.patch_embed.proj.bias")
    pos_weight = load_tensor(key_to_file, "visual.pos_embed.weight")
    norm_weight = load_tensor(key_to_file, "visual.merger.norm.weight")
    norm_bias = load_tensor(key_to_file, "visual.merger.norm.bias")
    fc1_weight = load_tensor(key_to_file, "visual.merger.linear_fc1.weight")
    fc1_bias = load_tensor(key_to_file, "visual.merger.linear_fc1.bias")
    fc2_weight = load_tensor(key_to_file, "visual.merger.linear_fc2.weight")
    fc2_bias = load_tensor(key_to_file, "visual.merger.linear_fc2.bias")

    bilinear_indices, bilinear_weights = qwen3_vl_vision_bilinear_numpy(grid_thw)
    pos = np.sum(pos_weight[bilinear_indices[:, :patch_count]] * bilinear_weights[:, :patch_count, None], axis=0).astype(np.float32)
    patch = pixel_values @ patch_weight.T + patch_bias + pos
    normed = layer_norm_numpy(patch, norm_weight, norm_bias)
    merged_input = normed.reshape(visual_token_count, 4096)
    hidden = gelu_tanh(merged_input @ fc1_weight.T + fc1_bias)
    return (hidden @ fc2_weight.T + fc2_bias).astype(np.float32), processor_npz, grid_thw


def vision_block_reference(x, grid_thw, model_dir, key_to_file, layer_idx):
    patch_count = x.shape[0]
    prefix = f"visual.blocks.{layer_idx}"
    norm1_w = load_tensor(key_to_file, prefix + ".norm1.weight")
    norm1_b = load_tensor(key_to_file, prefix + ".norm1.bias")
    norm2_w = load_tensor(key_to_file, prefix + ".norm2.weight")
    norm2_b = load_tensor(key_to_file, prefix + ".norm2.bias")
    qkv_w = load_tensor(key_to_file, prefix + ".attn.qkv.weight")
    qkv_b = load_tensor(key_to_file, prefix + ".attn.qkv.bias")
    proj_w = load_tensor(key_to_file, prefix + ".attn.proj.weight")
    proj_b = load_tensor(key_to_file, prefix + ".attn.proj.bias")
    mlp_fc1_w = load_tensor(key_to_file, prefix + ".mlp.linear_fc1.weight")
    mlp_fc1_b = load_tensor(key_to_file, prefix + ".mlp.linear_fc1.bias")
    mlp_fc2_w = load_tensor(key_to_file, prefix + ".mlp.linear_fc2.weight")
    mlp_fc2_b = load_tensor(key_to_file, prefix + ".mlp.linear_fc2.bias")

    norm1 = layer_norm_numpy(x, norm1_w, norm1_b)
    qkv = norm1 @ qkv_w.T + qkv_b
    qkv_rope, config = apply_vision_rope_numpy(qkv, grid_thw, model_dir, patch_count)
    attention = vision_attention_reference(qkv_rope, config)
    attn_out = attention @ proj_w.T + proj_b
    hidden = x + attn_out
    norm2 = layer_norm_numpy(hidden, norm2_w, norm2_b)
    mlp_hidden = gelu_tanh(norm2 @ mlp_fc1_w.T + mlp_fc1_b)
    mlp_out = mlp_hidden @ mlp_fc2_w.T + mlp_fc2_b
    return (hidden + mlp_out).astype(np.float32)


def run_garnet_vision_block(garnet, x, grid_thw, model_dir, key_to_file, layer_idx, cache_dir):
    patch_count = x.shape[0]
    prefix = f"visual.blocks.{layer_idx}"
    norm1_w = load_tensor(key_to_file, prefix + ".norm1.weight")
    norm1_b = load_tensor(key_to_file, prefix + ".norm1.bias")
    norm2_w = load_tensor(key_to_file, prefix + ".norm2.weight")
    norm2_b = load_tensor(key_to_file, prefix + ".norm2.bias")
    qkv_w = load_tensor(key_to_file, prefix + ".attn.qkv.weight")
    qkv_b = load_tensor(key_to_file, prefix + ".attn.qkv.bias")
    proj_w = load_tensor(key_to_file, prefix + ".attn.proj.weight")
    proj_b = load_tensor(key_to_file, prefix + ".attn.proj.bias")
    mlp_fc1_w = load_tensor(key_to_file, prefix + ".mlp.linear_fc1.weight")
    mlp_fc1_b = load_tensor(key_to_file, prefix + ".mlp.linear_fc1.bias")
    mlp_fc2_w = load_tensor(key_to_file, prefix + ".mlp.linear_fc2.weight")
    mlp_fc2_b = load_tensor(key_to_file, prefix + ".mlp.linear_fc2.bias")

    norm1_model = garnet.load_model(
        str(SCRIPT_DIR / "layer_norm_trt.x"),
        weights={
            "visual.blocks.0.norm1.weight": norm1_w,
            "visual.blocks.0.norm1.bias": norm1_b,
        },
        cache_dir=str(cache_dir / "norm1"),
        input_shapes=[list(x.shape)],
        subgraph="layer_norm",
    )
    gnorm1 = to_numpy(norm1_model.forward(x)).reshape(x.shape)

    qkv_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": qkv_w, "B": qkv_b},
        cache_dir=str(cache_dir / "qkv"),
        input_shapes=[list(gnorm1.shape)],
        subgraph="linear_bias",
    )
    gqkv = to_numpy(qkv_model.forward(gnorm1)).reshape(patch_count, 3072)
    gqkv_rope, _ = apply_vision_rope_numpy(gqkv, grid_thw, model_dir, patch_count)

    attention_model = garnet.load_model(
        str(SCRIPT_DIR / "vision_attention_core_trt.x"),
        weights={},
        cache_dir=str(cache_dir / "attention"),
        input_shapes=[list(gqkv_rope.shape)],
        subgraph="vision_attention_core",
    )
    gattention = to_numpy(attention_model.forward(gqkv_rope)).reshape(patch_count, 1024)

    proj_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": proj_w, "B": proj_b},
        cache_dir=str(cache_dir / "proj"),
        input_shapes=[list(gattention.shape)],
        subgraph="linear_bias",
    )
    gattn_out = to_numpy(proj_model.forward(gattention)).reshape(x.shape)
    ghidden = x + gattn_out

    norm2_model = garnet.load_model(
        str(SCRIPT_DIR / "layer_norm_trt.x"),
        weights={
            "visual.blocks.0.norm1.weight": norm2_w,
            "visual.blocks.0.norm1.bias": norm2_b,
        },
        cache_dir=str(cache_dir / "norm2"),
        input_shapes=[list(ghidden.shape)],
        subgraph="layer_norm",
    )
    gnorm2 = to_numpy(norm2_model.forward(ghidden)).reshape(x.shape)

    mlp_fc1_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": mlp_fc1_w, "B": mlp_fc1_b},
        cache_dir=str(cache_dir / "mlp_fc1"),
        input_shapes=[list(gnorm2.shape)],
        subgraph="linear_bias",
    )
    gmlp_hidden_linear = to_numpy(mlp_fc1_model.forward(gnorm2)).reshape(patch_count, 4096)
    gmlp_hidden = gelu_tanh(gmlp_hidden_linear).astype(np.float32)

    mlp_fc2_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": mlp_fc2_w, "B": mlp_fc2_b},
        cache_dir=str(cache_dir / "mlp_fc2"),
        input_shapes=[list(gmlp_hidden.shape)],
        subgraph="linear_bias",
    )
    gmlp = to_numpy(mlp_fc2_model.forward(gmlp_hidden)).reshape(x.shape)
    return (ghidden + gmlp).astype(np.float32)


def vision_merger_reference(x, key_to_file):
    norm_weight = load_tensor(key_to_file, "visual.merger.norm.weight")
    norm_bias = load_tensor(key_to_file, "visual.merger.norm.bias")
    fc1_weight = load_tensor(key_to_file, "visual.merger.linear_fc1.weight")
    fc1_bias = load_tensor(key_to_file, "visual.merger.linear_fc1.bias")
    fc2_weight = load_tensor(key_to_file, "visual.merger.linear_fc2.weight")
    fc2_bias = load_tensor(key_to_file, "visual.merger.linear_fc2.bias")
    normed = layer_norm_numpy(x, norm_weight, norm_bias)
    merged_input = normed.reshape(-1, 4096)
    hidden = gelu_tanh(merged_input @ fc1_weight.T + fc1_bias)
    return (hidden @ fc2_weight.T + fc2_bias).astype(np.float32)


def run_garnet_vision_merger(garnet, x, key_to_file, cache_dir):
    norm_weight = load_tensor(key_to_file, "visual.merger.norm.weight")
    norm_bias = load_tensor(key_to_file, "visual.merger.norm.bias")
    fc1_weight = load_tensor(key_to_file, "visual.merger.linear_fc1.weight")
    fc1_bias = load_tensor(key_to_file, "visual.merger.linear_fc1.bias")
    fc2_weight = load_tensor(key_to_file, "visual.merger.linear_fc2.weight")
    fc2_bias = load_tensor(key_to_file, "visual.merger.linear_fc2.bias")

    norm_model = garnet.load_model(
        str(SCRIPT_DIR / "layer_norm_trt.x"),
        weights={
            "visual.blocks.0.norm1.weight": norm_weight,
            "visual.blocks.0.norm1.bias": norm_bias,
        },
        cache_dir=str(cache_dir / "norm"),
        input_shapes=[list(x.shape)],
        subgraph="layer_norm",
    )
    gnormed = to_numpy(norm_model.forward(x)).reshape(x.shape)
    gmerged_input = gnormed.reshape(-1, 4096).astype(np.float32)

    fc1_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": fc1_weight, "B": fc1_bias},
        cache_dir=str(cache_dir / "fc1"),
        input_shapes=[list(gmerged_input.shape)],
        subgraph="linear_bias",
    )
    ghidden_linear = to_numpy(fc1_model.forward(gmerged_input)).reshape(gmerged_input.shape[0], 4096)
    ghidden = gelu_tanh(ghidden_linear).astype(np.float32)

    fc2_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": fc2_weight, "B": fc2_bias},
        cache_dir=str(cache_dir / "fc2"),
        input_shapes=[list(ghidden.shape)],
        subgraph="linear_bias",
    )
    return to_numpy(fc2_model.forward(ghidden)).reshape(gmerged_input.shape[0], 2048).astype(np.float32)


def vision_block_count_from_weights(key_to_file):
    indices = []
    for key in key_to_file:
        match = re.match(r"visual\.blocks\.(\d+)\.norm1\.weight$", key)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        raise AssertionError("no visual.blocks.* weights found")
    return max(indices) + 1


def run_vision_all_blocks_merger_from_processor(garnet, model_dir, key_to_file):
    patch_count = int(os.environ.get("GARNET_QWEN_VISION_ALL_BLOCKS_PATCH_WINDOW", "16"))
    if patch_count % 4 != 0:
        raise AssertionError("GARNET_QWEN_VISION_ALL_BLOCKS_PATCH_WINDOW must be a multiple of 4")
    processor_npz, grid_thw, x0 = vision_patch_with_pos_from_processor(key_to_file, patch_count)
    block_count = vision_block_count_from_weights(key_to_file)
    expected = x0.astype(np.float32)
    actual = x0.astype(np.float32)
    cache_dir = SCRIPT_DIR / "cache_vision_all_blocks_merger" / f"patch_{patch_count}"

    per_block = []
    for layer_idx in range(block_count):
        expected = vision_block_reference(expected, grid_thw, model_dir, key_to_file, layer_idx)
        actual = run_garnet_vision_block(garnet, actual, grid_thw, model_dir, key_to_file, layer_idx, cache_dir / f"block_{layer_idx}")
        block_error = np.abs(actual - expected)
        per_block.append({
            "layer": layer_idx,
            "max_error": float(np.max(block_error)),
            "mean_error": float(np.mean(block_error)),
        })

    expected_tokens = vision_merger_reference(expected, key_to_file)
    actual_tokens = run_garnet_vision_merger(garnet, actual, key_to_file, cache_dir / "merger")
    np.testing.assert_allclose(actual_tokens, expected_tokens, rtol=5e-2, atol=5e-1)
    return {
        "processor_npz": str(processor_npz),
        "grid_thw": grid_thw.tolist(),
        "patch_input_shape": list(x0.shape),
        "block_count": int(block_count),
        "visual_tokens_shape": list(actual_tokens.shape),
        "max_error": float(np.max(np.abs(actual_tokens - expected_tokens))),
        "mean_error": float(np.mean(np.abs(actual_tokens - expected_tokens))),
        "last_block": per_block[-1],
    }


def vision_all_blocks_merger_tokens_pair(garnet, model_dir, key_to_file, visual_token_count):
    patch_count = int(visual_token_count) * 4
    processor_npz, grid_thw, x0 = vision_patch_with_pos_from_processor(key_to_file, patch_count)
    block_count = vision_block_count_from_weights(key_to_file)
    expected = x0.astype(np.float32)
    actual = x0.astype(np.float32)
    cache_dir = SCRIPT_DIR / "cache_visual_splice_all_vision" / f"patch_{patch_count}"
    for layer_idx in range(block_count):
        expected = vision_block_reference(expected, grid_thw, model_dir, key_to_file, layer_idx)
        actual = run_garnet_vision_block(garnet, actual, grid_thw, model_dir, key_to_file, layer_idx, cache_dir / f"block_{layer_idx}")
    expected_tokens = vision_merger_reference(expected, key_to_file)
    actual_tokens = run_garnet_vision_merger(garnet, actual, key_to_file, cache_dir / "merger")
    return expected_tokens.astype(np.float32), actual_tokens.astype(np.float32), processor_npz, grid_thw, block_count


def run_visual_splice_logits_against_hf_modules(garnet, model_dir, key_to_file):
    torch, config, layer0, rotary, create_causal_mask = load_hf_text_layer(model_dir, key_to_file, 0)
    _, _, layer1, _, _ = load_hf_text_layer(model_dir, key_to_file, 1)
    processor_npz = latest_processor_npz()
    arrays = np.load(processor_npz)
    if "input_ids" not in arrays or "mm_token_type_ids" not in arrays:
        raise AssertionError(f"processor dump missing input_ids/mm_token_type_ids: {processor_npz}")
    token_count = int(os.environ.get("GARNET_QWEN_VISUAL_SPLICE_TOKEN_WINDOW", "8"))
    input_ids = arrays["input_ids"][0, :token_count].astype(np.int64)
    mm_types = arrays["mm_token_type_ids"][0, :token_count].astype(np.int64)
    visual_positions = np.flatnonzero(mm_types == 1)
    if visual_positions.size == 0:
        raise AssertionError("selected token window has no visual placeholder positions")

    embed_tokens = load_tensor(key_to_file, "language_model.embed_tokens.weight")
    final_norm = load_tensor(key_to_file, "language_model.norm.weight")
    use_all_vision = env_flag("GARNET_QWEN_VISUAL_SPLICE_USE_ALL_VISION_BLOCKS")
    if use_all_vision:
        expected_visual_tokens, actual_visual_tokens, visual_npz, grid_thw, vision_block_count = vision_all_blocks_merger_tokens_pair(
            garnet, model_dir, key_to_file, int(visual_positions.size)
        )
    else:
        expected_visual_tokens, visual_npz, grid_thw = vision_patch_pos_merger_tokens_numpy(key_to_file, int(visual_positions.size))
        actual_visual_tokens = expected_visual_tokens
        vision_block_count = 0
    expected_x = embed_tokens[input_ids].astype(np.float32)
    actual_x = expected_x.copy()
    expected_x[visual_positions] = expected_visual_tokens
    actual_x[visual_positions] = actual_visual_tokens

    with torch.inference_mode():
        hx = torch.from_numpy(expected_x).unsqueeze(0)
        position_ids = torch.stack([
            torch.arange(expected_x.shape[0], dtype=torch.long),
            torch.arange(expected_x.shape[0], dtype=torch.long) + 17,
            torch.arange(expected_x.shape[0], dtype=torch.long) + 31,
        ], dim=0).unsqueeze(1)
        text_position_ids = torch.arange(expected_x.shape[0], dtype=torch.long).unsqueeze(0)
        attention_mask = create_causal_mask(
            config=config,
            inputs_embeds=hx,
            attention_mask=torch.ones(1, expected_x.shape[0], dtype=torch.long),
            past_key_values=None,
            position_ids=text_position_ids,
        )
        cos, sin = rotary(hx, position_ids)
        expected_hidden = layer0(
            hx,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        )
        expected_hidden = layer1(
            expected_hidden,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=(cos, sin),
        ).squeeze(0).detach().cpu().numpy().astype(np.float32)
        cos_np = cos.squeeze(0).detach().cpu().numpy().astype(np.float32)
        sin_np = sin.squeeze(0).detach().cpu().numpy().astype(np.float32)

    expected_norm = expected_hidden / np.sqrt(np.mean(np.square(expected_hidden), axis=-1, keepdims=True) + 1.0e-6)
    expected_norm = expected_norm * final_norm
    expected_logits = expected_norm @ embed_tokens.T

    actual = actual_x
    cache_base = "cache_visual_splice_all_vision_logits_hf" if use_all_vision else "cache_visual_splice_logits_hf"
    cache_dir = SCRIPT_DIR / cache_base / f"tokens_{token_count}"
    for layer_idx in range(2):
        layer_weights = load_text_layer_weights_as_layer0_aliases(key_to_file, layer_idx)
        actual = run_garnet_text_decoder_layer(garnet, layer_weights, actual, cos_np, sin_np, cache_dir)

    final_norm_model = garnet.load_model(
        str(SCRIPT_DIR / "rms_norm_trt.x"),
        weights={"language_model.layers.0.input_layernorm.weight": final_norm},
        cache_dir=str(cache_dir),
        input_shapes=[list(actual.shape)],
        subgraph="rms_norm",
    )
    actual_norm = to_numpy(final_norm_model.forward(actual)).reshape(actual.shape)

    lm_head_model = garnet.load_model(
        str(SCRIPT_DIR / "text_lm_head_trt.x"),
        weights={"language_model.embed_tokens.weight": embed_tokens},
        cache_dir=str(cache_dir),
        input_shapes=[list(actual_norm.shape)],
        subgraph="text_lm_head",
    )
    actual_logits = to_numpy(lm_head_model.forward(actual_norm)).reshape(expected_logits.shape)

    expected_top = np.argsort(expected_logits[-1])[-10:][::-1]
    actual_top = np.argsort(actual_logits[-1])[-10:][::-1]
    overlap = len(set(int(x) for x in expected_top[:10]) & set(int(x) for x in actual_top[:10]))
    np.testing.assert_allclose(actual_logits, expected_logits, rtol=1e-1, atol=1.0)
    if overlap < 8:
        raise AssertionError(f"visual splice top-10 token overlap too low: {overlap}/10")
    return {
        "processor_npz": str(processor_npz),
        "visual_npz": str(visual_npz),
        "grid_thw": grid_thw.tolist(),
        "input_ids": [int(x) for x in input_ids.tolist()],
        "visual_positions": [int(x) for x in visual_positions.tolist()],
        "use_all_vision_blocks": bool(use_all_vision),
        "vision_block_count": int(vision_block_count),
        "visual_tokens_shape": list(actual_visual_tokens.shape),
        "visual_token_max_error": float(np.max(np.abs(actual_visual_tokens - expected_visual_tokens))),
        "visual_token_mean_error": float(np.mean(np.abs(actual_visual_tokens - expected_visual_tokens))),
        "logits_shape": list(actual_logits.shape),
        "max_error": float(np.max(np.abs(actual_logits - expected_logits))),
        "mean_error": float(np.mean(np.abs(actual_logits - expected_logits))),
        "last_token_top10_overlap": overlap,
    }


def run_vision_patch_embed_from_processor(garnet, key_to_file):
    processor_npz = latest_processor_npz()
    arrays = np.load(processor_npz)
    if "pixel_values" not in arrays:
        raise AssertionError(f"processor dump missing pixel_values: {processor_npz}")
    patch_count = int(os.environ.get("GARNET_QWEN_VISION_PATCH_WINDOW", "16"))
    pixel_values = arrays["pixel_values"][:patch_count].astype(np.float32)
    weight_5d = load_tensor(key_to_file, "visual.patch_embed.proj.weight")
    weight = weight_5d.reshape(weight_5d.shape[0], -1).astype(np.float32)
    bias = load_tensor(key_to_file, "visual.patch_embed.proj.bias")
    expected = pixel_values @ weight.T + bias

    engine = garnet.load_model(
        str(SCRIPT_DIR / "vision_patch_embed_trt.x"),
        weights={
            "visual.patch_embed.proj.weight": weight,
            "visual.patch_embed.proj.bias": bias,
        },
        cache_dir=str(SCRIPT_DIR / "cache_vision_patch_embed"),
        input_shapes=[list(pixel_values.shape)],
        subgraph="vision_patch_embed",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for vision patch embed")

    actual = to_numpy(engine.forward(pixel_values)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=8e-3, atol=8e-3)
    return {
        "processor_npz": str(processor_npz),
        "input_shape": list(pixel_values.shape),
        "weight_shape": list(weight.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_vision_patch_merger_from_processor(garnet, key_to_file):
    processor_npz = latest_processor_npz()
    arrays = np.load(processor_npz)
    if "pixel_values" not in arrays:
        raise AssertionError(f"processor dump missing pixel_values: {processor_npz}")
    patch_count = int(os.environ.get("GARNET_QWEN_VISION_MERGER_PATCH_WINDOW", "16"))
    if patch_count % 4 != 0:
        raise AssertionError("GARNET_QWEN_VISION_MERGER_PATCH_WINDOW must be a multiple of 4")
    pixel_values = arrays["pixel_values"][:patch_count].astype(np.float32)
    patch_weight_5d = load_tensor(key_to_file, "visual.patch_embed.proj.weight")
    patch_weight = patch_weight_5d.reshape(patch_weight_5d.shape[0], -1).astype(np.float32)
    patch_bias = load_tensor(key_to_file, "visual.patch_embed.proj.bias")
    patch = pixel_values @ patch_weight.T + patch_bias

    norm_weight = load_tensor(key_to_file, "visual.merger.norm.weight")
    norm_bias = load_tensor(key_to_file, "visual.merger.norm.bias")
    fc1_weight = load_tensor(key_to_file, "visual.merger.linear_fc1.weight")
    fc1_bias = load_tensor(key_to_file, "visual.merger.linear_fc1.bias")
    fc2_weight = load_tensor(key_to_file, "visual.merger.linear_fc2.weight")
    fc2_bias = load_tensor(key_to_file, "visual.merger.linear_fc2.bias")

    normed = (patch - np.mean(patch, axis=-1, keepdims=True)) / np.sqrt(np.var(patch, axis=-1, keepdims=True) + 1.0e-6)
    normed = normed * norm_weight + norm_bias
    merged_input = normed.reshape(-1, 4096)
    hidden = gelu_tanh(merged_input @ fc1_weight.T + fc1_bias)
    expected = hidden @ fc2_weight.T + fc2_bias

    cache_dir = SCRIPT_DIR / "cache_vision_patch_merger"
    norm_model = garnet.load_model(
        str(SCRIPT_DIR / "layer_norm_trt.x"),
        weights={
            "visual.blocks.0.norm1.weight": norm_weight,
            "visual.blocks.0.norm1.bias": norm_bias,
        },
        cache_dir=str(cache_dir),
        input_shapes=[list(patch.shape)],
        subgraph="layer_norm",
    )
    gnormed = to_numpy(norm_model.forward(patch)).reshape(patch.shape)
    gmerged_input = gnormed.reshape(-1, 4096).astype(np.float32)

    fc1_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": fc1_weight, "B": fc1_bias},
        cache_dir=str(cache_dir / "fc1"),
        input_shapes=[list(gmerged_input.shape)],
        subgraph="linear_bias",
    )
    ghidden_linear = to_numpy(fc1_model.forward(gmerged_input)).reshape(hidden.shape)
    ghidden = gelu_tanh(ghidden_linear).astype(np.float32)

    fc2_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": fc2_weight, "B": fc2_bias},
        cache_dir=str(cache_dir / "fc2"),
        input_shapes=[list(ghidden.shape)],
        subgraph="linear_bias",
    )
    actual = to_numpy(fc2_model.forward(ghidden)).reshape(expected.shape)

    np.testing.assert_allclose(actual, expected, rtol=8e-3, atol=8e-3)
    return {
        "processor_npz": str(processor_npz),
        "patch_input_shape": list(pixel_values.shape),
        "merged_input_shape": list(gmerged_input.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def qwen3_vl_vision_bilinear_numpy(grid_thw, num_grid_per_side=48, spatial_merge_size=2):
    idx_parts = [[] for _ in range(4)]
    weight_parts = [[] for _ in range(4)]
    side = int(num_grid_per_side)
    merge = int(spatial_merge_size)
    for t, h, w in grid_thw.astype(np.int64).tolist():
        h_grid = np.linspace(0, side - 1, int(h), dtype=np.float32)
        w_grid = np.linspace(0, side - 1, int(w), dtype=np.float32)
        h_floor = h_grid.astype(np.int64)
        w_floor = w_grid.astype(np.int64)
        h_ceil = np.clip(h_floor + 1, 0, side - 1)
        w_ceil = np.clip(w_floor + 1, 0, side - 1)
        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor
        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side

        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).reshape(-1),
            (h_floor_offset[:, None] + w_ceil[None, :]).reshape(-1),
            (h_ceil_offset[:, None] + w_floor[None, :]).reshape(-1),
            (h_ceil_offset[:, None] + w_ceil[None, :]).reshape(-1),
        ]
        corner_weights = [
            ((1.0 - h_frac)[:, None] * (1.0 - w_frac)[None, :]).reshape(-1),
            ((1.0 - h_frac)[:, None] * w_frac[None, :]).reshape(-1),
            (h_frac[:, None] * (1.0 - w_frac)[None, :]).reshape(-1),
            (h_frac[:, None] * w_frac[None, :]).reshape(-1),
        ]

        h_idx = np.arange(int(h), dtype=np.int64).reshape(int(h) // merge, merge)
        w_idx = np.arange(int(w), dtype=np.int64).reshape(int(w) // merge, merge)
        reorder = (h_idx[:, :, None, None] * int(w) + w_idx[None, None, :, :]).transpose(0, 2, 1, 3).reshape(-1)
        reorder = np.tile(reorder, int(t))
        for i in range(4):
            idx_parts[i].append(corner_indices[i][reorder])
            weight_parts[i].append(corner_weights[i][reorder])

    return np.stack([np.concatenate(p) for p in idx_parts]), np.stack([np.concatenate(p) for p in weight_parts]).astype(np.float32)


def run_vision_patch_pos_merger_from_processor(garnet, key_to_file):
    processor_npz = latest_processor_npz()
    arrays = np.load(processor_npz)
    if "pixel_values" not in arrays or "image_grid_thw" not in arrays:
        raise AssertionError(f"processor dump missing pixel_values/image_grid_thw: {processor_npz}")
    patch_count = int(os.environ.get("GARNET_QWEN_VISION_POS_MERGER_PATCH_WINDOW", "16"))
    if patch_count % 4 != 0:
        raise AssertionError("GARNET_QWEN_VISION_POS_MERGER_PATCH_WINDOW must be a multiple of 4")
    pixel_values = arrays["pixel_values"][:patch_count].astype(np.float32)
    grid_thw = arrays["image_grid_thw"].astype(np.int64)

    patch_weight_5d = load_tensor(key_to_file, "visual.patch_embed.proj.weight")
    patch_weight = patch_weight_5d.reshape(patch_weight_5d.shape[0], -1).astype(np.float32)
    patch_bias = load_tensor(key_to_file, "visual.patch_embed.proj.bias")
    pos_weight = load_tensor(key_to_file, "visual.pos_embed.weight")
    norm_weight = load_tensor(key_to_file, "visual.merger.norm.weight")
    norm_bias = load_tensor(key_to_file, "visual.merger.norm.bias")
    fc1_weight = load_tensor(key_to_file, "visual.merger.linear_fc1.weight")
    fc1_bias = load_tensor(key_to_file, "visual.merger.linear_fc1.bias")
    fc2_weight = load_tensor(key_to_file, "visual.merger.linear_fc2.weight")
    fc2_bias = load_tensor(key_to_file, "visual.merger.linear_fc2.bias")

    bilinear_indices, bilinear_weights = qwen3_vl_vision_bilinear_numpy(grid_thw)
    pos = np.sum(pos_weight[bilinear_indices[:, :patch_count]] * bilinear_weights[:, :patch_count, None], axis=0).astype(np.float32)
    patch = pixel_values @ patch_weight.T + patch_bias
    patch_with_pos = patch + pos

    normed = (patch_with_pos - np.mean(patch_with_pos, axis=-1, keepdims=True)) / np.sqrt(np.var(patch_with_pos, axis=-1, keepdims=True) + 1.0e-6)
    normed = normed * norm_weight + norm_bias
    merged_input = normed.reshape(-1, 4096)
    hidden = gelu_tanh(merged_input @ fc1_weight.T + fc1_bias)
    expected = hidden @ fc2_weight.T + fc2_bias

    cache_dir = SCRIPT_DIR / "cache_vision_patch_pos_merger"
    patch_model = garnet.load_model(
        str(SCRIPT_DIR / "vision_patch_embed_trt.x"),
        weights={
            "visual.patch_embed.proj.weight": patch_weight,
            "visual.patch_embed.proj.bias": patch_bias,
        },
        cache_dir=str(cache_dir / "patch"),
        input_shapes=[list(pixel_values.shape)],
        subgraph="vision_patch_embed",
    )
    gpatch = to_numpy(patch_model.forward(pixel_values)).reshape(patch.shape)
    gpatch_with_pos = (gpatch + pos).astype(np.float32)

    norm_model = garnet.load_model(
        str(SCRIPT_DIR / "layer_norm_trt.x"),
        weights={
            "visual.blocks.0.norm1.weight": norm_weight,
            "visual.blocks.0.norm1.bias": norm_bias,
        },
        cache_dir=str(cache_dir / "norm"),
        input_shapes=[list(gpatch_with_pos.shape)],
        subgraph="layer_norm",
    )
    gnormed = to_numpy(norm_model.forward(gpatch_with_pos)).reshape(gpatch_with_pos.shape)
    gmerged_input = gnormed.reshape(-1, 4096).astype(np.float32)

    fc1_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": fc1_weight, "B": fc1_bias},
        cache_dir=str(cache_dir / "fc1"),
        input_shapes=[list(gmerged_input.shape)],
        subgraph="linear_bias",
    )
    ghidden_linear = to_numpy(fc1_model.forward(gmerged_input)).reshape(hidden.shape)
    ghidden = gelu_tanh(ghidden_linear).astype(np.float32)

    fc2_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": fc2_weight, "B": fc2_bias},
        cache_dir=str(cache_dir / "fc2"),
        input_shapes=[list(ghidden.shape)],
        subgraph="linear_bias",
    )
    actual = to_numpy(fc2_model.forward(ghidden)).reshape(expected.shape)

    np.testing.assert_allclose(actual, expected, rtol=8e-3, atol=8e-3)
    return {
        "processor_npz": str(processor_npz),
        "grid_thw": grid_thw.tolist(),
        "patch_input_shape": list(pixel_values.shape),
        "pos_shape": list(pos.shape),
        "merged_input_shape": list(gmerged_input.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def vision_patch_with_pos_from_processor(key_to_file, patch_count):
    processor_npz = latest_processor_npz()
    arrays = np.load(processor_npz)
    if "pixel_values" not in arrays or "image_grid_thw" not in arrays:
        raise AssertionError(f"processor dump missing pixel_values/image_grid_thw: {processor_npz}")
    grid_thw = arrays["image_grid_thw"].astype(np.int64)
    return processor_npz, grid_thw, vision_patch_with_pos_from_arrays(key_to_file, arrays["pixel_values"], grid_thw, patch_count)


def vision_patch_with_pos_from_arrays(key_to_file, pixel_values_all, grid_thw, patch_count):
    pixel_values = pixel_values_all[:patch_count].astype(np.float32)
    patch_weight_5d = load_tensor(key_to_file, "visual.patch_embed.proj.weight")
    patch_weight = patch_weight_5d.reshape(patch_weight_5d.shape[0], -1).astype(np.float32)
    patch_bias = load_tensor(key_to_file, "visual.patch_embed.proj.bias")
    pos_weight = load_tensor(key_to_file, "visual.pos_embed.weight")
    bilinear_indices, bilinear_weights = qwen3_vl_vision_bilinear_numpy(grid_thw)
    pos = np.sum(pos_weight[bilinear_indices[:, :patch_count]] * bilinear_weights[:, :patch_count, None], axis=0).astype(np.float32)
    patch = pixel_values @ patch_weight.T + patch_bias
    return patch + pos


def run_vision_qkv_proj_from_patch_pos(garnet, key_to_file):
    patch_count = int(os.environ.get("GARNET_QWEN_VISION_QKV_PATCH_WINDOW", "16"))
    processor_npz, grid_thw, x = vision_patch_with_pos_from_processor(key_to_file, patch_count)
    weight = load_tensor(key_to_file, "visual.blocks.0.attn.qkv.weight")
    bias = load_tensor(key_to_file, "visual.blocks.0.attn.qkv.bias")
    expected = x @ weight.T + bias

    qkv_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": weight, "B": bias},
        cache_dir=str(SCRIPT_DIR / "cache_vision_qkv_proj"),
        input_shapes=[list(x.shape)],
        subgraph="linear_bias",
    )
    actual = to_numpy(qkv_model.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=8e-3, atol=8e-3)
    return {
        "processor_npz": str(processor_npz),
        "grid_thw": grid_thw.tolist(),
        "input_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_vision_attention_core_from_patch_pos(garnet, model_dir, key_to_file):
    try:
        import torch
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionRotaryEmbedding,
            get_vision_position_ids,
        )
    except Exception as exc:
        skip(f"HF vision attention helpers are not available: {exc}")

    patch_count = int(os.environ.get("GARNET_QWEN_VISION_ATTENTION_PATCH_WINDOW", "16"))
    processor_npz, grid_thw, x = vision_patch_with_pos_from_processor(key_to_file, patch_count)
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True).vision_config
    weight = load_tensor(key_to_file, "visual.blocks.0.attn.qkv.weight")
    bias = load_tensor(key_to_file, "visual.blocks.0.attn.qkv.bias")
    qkv = x @ weight.T + bias
    q, k, v = qkv.reshape(patch_count, 3, config.num_heads, -1).transpose(1, 0, 2, 3)

    with torch.inference_mode():
        position_ids = get_vision_position_ids(
            torch.from_numpy(grid_thw),
            config.spatial_merge_size,
        )[:patch_count]
        rotary = Qwen3VLVisionRotaryEmbedding((config.hidden_size // config.num_heads) // 2)
        rotary_pos = rotary(position_ids).detach().cpu().numpy().astype(np.float32)
    emb = np.concatenate([rotary_pos, rotary_pos], axis=-1)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    q = (q * cos[:, None, :]) + (rotate_half(q) * sin[:, None, :])
    k = (k * cos[:, None, :]) + (rotate_half(k) * sin[:, None, :])

    q_htd = np.transpose(q, (1, 0, 2))
    k_htd = np.transpose(k, (1, 0, 2))
    v_htd = np.transpose(v, (1, 0, 2))
    scores = np.matmul(q_htd, np.swapaxes(k_htd, 1, 2)) * ((config.hidden_size // config.num_heads) ** -0.5)
    weights = softmax(scores, axis=-1).astype(np.float32)
    context = np.matmul(weights, v_htd)
    expected = np.transpose(context, (1, 0, 2)).reshape(patch_count, config.hidden_size)
    qkv_rope = np.concatenate([
        q.reshape(patch_count, config.hidden_size),
        k.reshape(patch_count, config.hidden_size),
        v.reshape(patch_count, config.hidden_size),
    ], axis=-1).astype(np.float32)

    attention_model = garnet.load_model(
        str(SCRIPT_DIR / "vision_attention_core_trt.x"),
        weights={},
        cache_dir=str(SCRIPT_DIR / "cache_vision_attention_core"),
        input_shapes=[list(qkv_rope.shape)],
        subgraph="vision_attention_core",
    )
    actual = to_numpy(attention_model.forward(qkv_rope)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=8e-3, atol=8e-3)
    return {
        "processor_npz": str(processor_npz),
        "grid_thw": grid_thw.tolist(),
        "input_shape": list(qkv_rope.shape),
        "num_heads": int(config.num_heads),
        "head_dim": int(config.hidden_size // config.num_heads),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def vision_rope_cos_sin_numpy(grid_thw, model_dir, patch_count):
    try:
        import torch
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionRotaryEmbedding,
            get_vision_position_ids,
        )
    except Exception as exc:
        skip(f"HF vision RoPE helpers are not available: {exc}")
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True).vision_config
    with torch.inference_mode():
        position_ids = get_vision_position_ids(
            torch.from_numpy(grid_thw),
            config.spatial_merge_size,
        )[:patch_count]
        rotary = Qwen3VLVisionRotaryEmbedding((config.hidden_size // config.num_heads) // 2)
        rotary_pos = rotary(position_ids).detach().cpu().numpy().astype(np.float32)
    emb = np.concatenate([rotary_pos, rotary_pos], axis=-1)
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    return cos, sin


def apply_vision_rope_numpy(qkv, grid_thw, model_dir, patch_count):
    try:
        from transformers import AutoConfig
    except Exception as exc:
        skip(f"HF vision config is not available: {exc}")
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True).vision_config
    q, k, v = qkv.reshape(patch_count, 3, config.num_heads, -1).transpose(1, 0, 2, 3)
    cos, sin = vision_rope_cos_sin_numpy(grid_thw, model_dir, patch_count)
    q = (q * cos[:, None, :]) + (rotate_half(q) * sin[:, None, :])
    k = (k * cos[:, None, :]) + (rotate_half(k) * sin[:, None, :])
    qkv_rope = np.concatenate([
        q.reshape(patch_count, config.hidden_size),
        k.reshape(patch_count, config.hidden_size),
        v.reshape(patch_count, config.hidden_size),
    ], axis=-1).astype(np.float32)
    return qkv_rope, config


def vision_attention_reference(qkv_rope, config):
    patch_count = qkv_rope.shape[0]
    q, k, v = qkv_rope.reshape(patch_count, 3, config.num_heads, -1).transpose(1, 0, 2, 3)
    q_htd = np.transpose(q, (1, 0, 2))
    k_htd = np.transpose(k, (1, 0, 2))
    v_htd = np.transpose(v, (1, 0, 2))
    scores = np.matmul(q_htd, np.swapaxes(k_htd, 1, 2)) * ((config.hidden_size // config.num_heads) ** -0.5)
    weights = softmax(scores, axis=-1).astype(np.float32)
    context = np.matmul(weights, v_htd)
    return np.transpose(context, (1, 0, 2)).reshape(patch_count, config.hidden_size)


def layer_norm_numpy(x, weight, bias, eps=1.0e-6):
    y = (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(np.var(x, axis=-1, keepdims=True) + eps)
    return y * weight + bias


def run_vision_block0_from_patch_pos(garnet, model_dir, key_to_file):
    patch_count = int(os.environ.get("GARNET_QWEN_VISION_BLOCK_PATCH_WINDOW", "16"))
    processor_npz, grid_thw, x = vision_patch_with_pos_from_processor(key_to_file, patch_count)
    prefix = "visual.blocks.0"
    norm1_w = load_tensor(key_to_file, prefix + ".norm1.weight")
    norm1_b = load_tensor(key_to_file, prefix + ".norm1.bias")
    norm2_w = load_tensor(key_to_file, prefix + ".norm2.weight")
    norm2_b = load_tensor(key_to_file, prefix + ".norm2.bias")
    qkv_w = load_tensor(key_to_file, prefix + ".attn.qkv.weight")
    qkv_b = load_tensor(key_to_file, prefix + ".attn.qkv.bias")
    proj_w = load_tensor(key_to_file, prefix + ".attn.proj.weight")
    proj_b = load_tensor(key_to_file, prefix + ".attn.proj.bias")
    mlp_fc1_w = load_tensor(key_to_file, prefix + ".mlp.linear_fc1.weight")
    mlp_fc1_b = load_tensor(key_to_file, prefix + ".mlp.linear_fc1.bias")
    mlp_fc2_w = load_tensor(key_to_file, prefix + ".mlp.linear_fc2.weight")
    mlp_fc2_b = load_tensor(key_to_file, prefix + ".mlp.linear_fc2.bias")

    norm1 = layer_norm_numpy(x, norm1_w, norm1_b)
    qkv = norm1 @ qkv_w.T + qkv_b
    qkv_rope, config = apply_vision_rope_numpy(qkv, grid_thw, model_dir, patch_count)
    attention = vision_attention_reference(qkv_rope, config)
    attn_out = attention @ proj_w.T + proj_b
    hidden = x + attn_out
    norm2 = layer_norm_numpy(hidden, norm2_w, norm2_b)
    mlp_hidden = gelu_tanh(norm2 @ mlp_fc1_w.T + mlp_fc1_b)
    mlp_out = mlp_hidden @ mlp_fc2_w.T + mlp_fc2_b
    expected = hidden + mlp_out

    cache_dir = SCRIPT_DIR / "cache_vision_block0"
    norm1_model = garnet.load_model(
        str(SCRIPT_DIR / "layer_norm_trt.x"),
        weights={
            "visual.blocks.0.norm1.weight": norm1_w,
            "visual.blocks.0.norm1.bias": norm1_b,
        },
        cache_dir=str(cache_dir / "norm1"),
        input_shapes=[list(x.shape)],
        subgraph="layer_norm",
    )
    gnorm1 = to_numpy(norm1_model.forward(x)).reshape(x.shape)

    qkv_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": qkv_w, "B": qkv_b},
        cache_dir=str(cache_dir / "qkv"),
        input_shapes=[list(gnorm1.shape)],
        subgraph="linear_bias",
    )
    gqkv = to_numpy(qkv_model.forward(gnorm1)).reshape(qkv.shape)
    gqkv_rope, _ = apply_vision_rope_numpy(gqkv, grid_thw, model_dir, patch_count)

    attention_model = garnet.load_model(
        str(SCRIPT_DIR / "vision_attention_core_trt.x"),
        weights={},
        cache_dir=str(cache_dir / "attention"),
        input_shapes=[list(gqkv_rope.shape)],
        subgraph="vision_attention_core",
    )
    gattention = to_numpy(attention_model.forward(gqkv_rope)).reshape(attention.shape)

    proj_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": proj_w, "B": proj_b},
        cache_dir=str(cache_dir / "proj"),
        input_shapes=[list(gattention.shape)],
        subgraph="linear_bias",
    )
    gattn_out = to_numpy(proj_model.forward(gattention)).reshape(attn_out.shape)
    ghidden = x + gattn_out

    norm2_model = garnet.load_model(
        str(SCRIPT_DIR / "layer_norm_trt.x"),
        weights={
            "visual.blocks.0.norm1.weight": norm2_w,
            "visual.blocks.0.norm1.bias": norm2_b,
        },
        cache_dir=str(cache_dir / "norm2"),
        input_shapes=[list(ghidden.shape)],
        subgraph="layer_norm",
    )
    gnorm2 = to_numpy(norm2_model.forward(ghidden)).reshape(norm2.shape)

    mlp_fc1_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": mlp_fc1_w, "B": mlp_fc1_b},
        cache_dir=str(cache_dir / "mlp_fc1"),
        input_shapes=[list(gnorm2.shape)],
        subgraph="linear_bias",
    )
    gmlp_hidden_linear = to_numpy(mlp_fc1_model.forward(gnorm2)).reshape(mlp_hidden.shape)
    gmlp_hidden = gelu_tanh(gmlp_hidden_linear).astype(np.float32)

    mlp_fc2_model = garnet.load_model(
        str(SCRIPT_DIR / "linear_bias_trt.x"),
        weights={"W": mlp_fc2_w, "B": mlp_fc2_b},
        cache_dir=str(cache_dir / "mlp_fc2"),
        input_shapes=[list(gmlp_hidden.shape)],
        subgraph="linear_bias",
    )
    gmlp = to_numpy(mlp_fc2_model.forward(gmlp_hidden)).reshape(mlp_out.shape)
    actual = ghidden + gmlp

    np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-1)
    return {
        "processor_npz": str(processor_npz),
        "grid_thw": grid_thw.tolist(),
        "input_shape": list(x.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_text_mlp(garnet, key_to_file):
    prefix = "language_model.layers.0.mlp"
    weights = {
        prefix + ".gate_proj.weight": load_tensor(key_to_file, prefix + ".gate_proj.weight"),
        prefix + ".up_proj.weight": load_tensor(key_to_file, prefix + ".up_proj.weight"),
        prefix + ".down_proj.weight": load_tensor(key_to_file, prefix + ".down_proj.weight"),
    }

    rng = np.random.default_rng(20260706)
    x = rng.normal(0.0, 0.02, size=(3, weights[prefix + ".gate_proj.weight"].shape[1])).astype(np.float32)
    gate = x @ weights[prefix + ".gate_proj.weight"].T
    up = x @ weights[prefix + ".up_proj.weight"].T
    expected = (silu(gate) * up) @ weights[prefix + ".down_proj.weight"].T

    engine = garnet.load_model(
        str(SCRIPT_DIR / "text_mlp_trt.x"),
        weights=weights,
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="qwen3_text_mlp",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real Qwen3TextMLP")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=7e-3, atol=7e-3)
    return {
        "input_shape": list(x.shape),
        "gate_shape": list(weights[prefix + ".gate_proj.weight"].shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_text_mlp_bfloat16_weights(garnet, key_to_file):
    import torch

    prefix = "language_model.layers.0.mlp"
    keys = [prefix + suffix for suffix in (
        ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")]
    cpu_weights = {
        key: torch.from_numpy(load_tensor(key_to_file, key)).to(torch.bfloat16).float().numpy()
        for key in keys
    }
    gpu_weights = {key: load_bfloat16_gpu_tensor(garnet, key_to_file, key) for key in keys}
    rng = np.random.default_rng(20260720)
    x = rng.normal(0.0, 0.02, size=(3, cpu_weights[keys[0]].shape[1])).astype(np.float32)
    gate = x @ cpu_weights[keys[0]].T
    up = x @ cpu_weights[keys[1]].T
    expected = (silu(gate) * up) @ cpu_weights[keys[2]].T
    engine = garnet.load_model(
        str(SCRIPT_DIR / "text_mlp_trt.x"),
        weights=gpu_weights,
        cache_dir=str(SCRIPT_DIR / "cache_real_bfloat16"),
        input_shapes=[list(x.shape)],
        subgraph="qwen3_text_mlp",
    )
    actual_value = engine.forward(x)
    actual = logits_to_numpy_or_cpu_copy(garnet, actual_value, x.shape[0], x.shape[1])
    np.testing.assert_allclose(actual, expected, rtol=1.5e-2, atol=2e-2)
    return {
        "weight_dtype": "bfloat16",
        "activation_dtype": "float32",
        "input_shape": list(x.shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


def run_vision_mlp(garnet, key_to_file):
    prefix = "visual.blocks.0.mlp"
    weights = {
        prefix + ".linear_fc1.weight": load_tensor(key_to_file, prefix + ".linear_fc1.weight"),
        prefix + ".linear_fc1.bias": load_tensor(key_to_file, prefix + ".linear_fc1.bias"),
        prefix + ".linear_fc2.weight": load_tensor(key_to_file, prefix + ".linear_fc2.weight"),
        prefix + ".linear_fc2.bias": load_tensor(key_to_file, prefix + ".linear_fc2.bias"),
    }

    rng = np.random.default_rng(20260707)
    x = rng.normal(0.0, 0.02, size=(5, weights[prefix + ".linear_fc1.weight"].shape[1])).astype(np.float32)
    hidden = gelu_tanh(x @ weights[prefix + ".linear_fc1.weight"].T + weights[prefix + ".linear_fc1.bias"])
    expected = hidden @ weights[prefix + ".linear_fc2.weight"].T + weights[prefix + ".linear_fc2.bias"]

    engine = garnet.load_model(
        str(SCRIPT_DIR / "vision_mlp_trt.x"),
        weights=weights,
        cache_dir=str(SCRIPT_DIR / "cache_real"),
        input_shapes=[list(x.shape)],
        subgraph="vision_mlp",
    )
    if engine is None:
        raise AssertionError("Garnet load_model returned None for real VisionMLP")

    actual = to_numpy(engine.forward(x)).reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=7e-3, atol=7e-3)
    return {
        "input_shape": list(x.shape),
        "fc1_shape": list(weights[prefix + ".linear_fc1.weight"].shape),
        "output_shape": list(actual.shape),
        "max_error": float(np.max(np.abs(actual - expected))),
        "mean_error": float(np.mean(np.abs(actual - expected))),
    }


print("Phase 05: real Qwen3-VL MLP subgraph parity")

if not env_flag("RUN_GARNET_REAL_QWEN_MLP_PARITY"):
    skip("set RUN_GARNET_REAL_QWEN_MLP_PARITY=1 to run real Qwen3-VL MLP subgraph parity")

model_dir = default_model_dir()
if model_dir is None or not Path(model_dir).exists():
    skip("Qwen3-VL model snapshot not found; set HF_QWEN_VL_MODEL_DIR")

garnet, garnet_dll = import_garnet()
key_to_file = load_weight_map(model_dir)

if env_flag("RUN_GARNET_REAL_QWEN_BFLOAT16_MLP_ONLY"):
    result = run_text_mlp_bfloat16_weights(garnet, key_to_file)
    print(__import__("json").dumps(result, indent=2))
    raise SystemExit(0)

if env_flag("RUN_GARNET_REAL_QWEN_MODEL_FORWARD_NATIVE_ROPE_DECODE") and env_flag("GARNET_QWEN_NATIVE_ROPE_DECODE_ONLY"):
    import json
    image_list_value = os.environ.get("GARNET_QWEN_NATIVE_ROPE_DECODE_IMAGES", "").strip()
    if image_list_value:
        image_paths = [Path(value) for value in image_list_value.split(";") if value.strip()]
        if not image_paths:
            raise AssertionError("GARNET_QWEN_NATIVE_ROPE_DECODE_IMAGES did not contain an image")
        serving_state = {}
        requests = []
        batch_start = time.perf_counter()
        try:
            for request_index, image_path in enumerate(image_paths):
                if not image_path.exists():
                    raise AssertionError(f"multi-image test input does not exist: {image_path}")
                os.environ["GARNET_QWEN_NATIVE_ROPE_DECODE_IMAGE"] = str(image_path)
                os.environ["GARNET_QWEN_NATIVE_ROPE_DECODE_SEQUENCE_ID"] = str(request_index + 1)
                request_start = time.perf_counter()
                request_result = run_model_forward_facade_native_rope_decode(
                    garnet, model_dir, key_to_file, serving_state=serving_state
                )
                request_result["request_index"] = request_index
                request_result["request_wall_ms"] = (time.perf_counter() - request_start) * 1000.0
                requests.append(request_result)
        finally:
            facade = serving_state.get("facade")
            if facade is not None:
                facade.close()
        steady_requests = requests[1:] if len(requests) > 1 else requests
        steady_values = [float(item["timing_ms"]["steady_pipeline_ms"]) for item in steady_requests]
        wall_values = [float(item["request_wall_ms"]) for item in steady_requests]
        decode_values = [
            float(step["step_ms"])
            for item in steady_requests
            for step in item["steps"]
            if step["mode"] == "cached_decode_one"
        ]
        model_forward_native_rope_decode_result = {
            "mode": "persistent_facade_multi_image",
            "request_count": len(requests),
            "first_request_is_warmup": len(requests) > 1,
            "batch_wall_ms": (time.perf_counter() - batch_start) * 1000.0,
            "requests": requests,
            "aggregate": {
                "steady_request_count": len(steady_requests),
                "steady_pipeline_avg_ms": float(np.mean(steady_values)),
                "steady_pipeline_min_ms": float(np.min(steady_values)),
                "steady_pipeline_max_ms": float(np.max(steady_values)),
                "full_request_avg_ms": float(np.mean(wall_values)),
                "full_request_min_ms": float(np.min(wall_values)),
                "full_request_max_ms": float(np.max(wall_values)),
                "decode_token_avg_ms": float(np.mean(decode_values)) if decode_values else 0.0,
                "decode_token_min_ms": float(np.min(decode_values)) if decode_values else 0.0,
                "decode_token_max_ms": float(np.max(decode_values)) if decode_values else 0.0,
            },
        }
    else:
        model_forward_native_rope_decode_result = run_model_forward_facade_native_rope_decode(garnet, model_dir, key_to_file)
    print(json.dumps(model_forward_native_rope_decode_result, indent=2))
    raise SystemExit(0)

if env_flag("RUN_GARNET_REAL_QWEN_CACHED_DECODER_LAYER_PARITY") and env_flag("GARNET_CACHED_DECODER_LAYER_ONLY"):
    cached_decoder_layer_result = run_garnet_text_decoder_layer_cached_last_token_parity(garnet, key_to_file)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cached_summary_path = ARTIFACT_DIR / "real_qwen_cached_decoder_layer_parity.json"
    cached_summary_path.write_text(
        __import__("json").dumps(
            {
                "model_dir": str(model_dir),
                "garnet": str(garnet_dll),
                "cached_decoder_layer": cached_decoder_layer_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(__import__("json").dumps(cached_decoder_layer_result, indent=2))
    print(f"wrote {cached_summary_path}")
    raise SystemExit(0)

if env_flag("RUN_GARNET_REAL_QWEN_CACHED_PREFILL_DECODE_PARITY") and env_flag("GARNET_CACHED_PREFILL_DECODE_ONLY"):
    cached_prefill_decode_result = run_garnet_text_prefill_decode_cached_parity(garnet, key_to_file)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cached_summary_path = ARTIFACT_DIR / "real_qwen_cached_prefill_decode_parity.json"
    cached_summary_path.write_text(
        __import__("json").dumps(
            {
                "model_dir": str(model_dir),
                "garnet": str(garnet_dll),
                "cached_prefill_decode": cached_prefill_decode_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(__import__("json").dumps(cached_prefill_decode_result, indent=2))
    print(f"wrote {cached_summary_path}")
    raise SystemExit(0)

if env_flag("RUN_GARNET_REAL_QWEN_DEVICE_KV_PREFILL_DECODE_PARITY") and env_flag("GARNET_DEVICE_KV_PREFILL_DECODE_ONLY"):
    device_kv_prefill_decode_result = run_garnet_text_prefill_decode_device_kv_parity(garnet, key_to_file)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cached_summary_path = ARTIFACT_DIR / "real_qwen_device_kv_prefill_decode_parity.json"
    cached_summary_path.write_text(
        __import__("json").dumps(
            {
                "model_dir": str(model_dir),
                "garnet": str(garnet_dll),
                "device_kv_prefill_decode": device_kv_prefill_decode_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(__import__("json").dumps(device_kv_prefill_decode_result, indent=2))
    print(f"wrote {cached_summary_path}")
    raise SystemExit(0)

text_norm_result = run_rms_norm(garnet, key_to_file)
vision_norm_result = run_layer_norm(garnet, key_to_file)
text_qkv_result = run_text_qkv_proj(garnet, key_to_file)
text_qkv_head_norm_result = run_text_qkv_head_norm(garnet, key_to_file)
text_o_proj_result = run_text_o_proj(garnet, key_to_file)
text_rope_result = run_text_rope_apply(garnet, key_to_file)
text_attention_result = run_text_attention_core(garnet, key_to_file)
text_decoder_layer_result = run_text_decoder_layer_chain(garnet, key_to_file)
cached_decoder_layer_result = None
if env_flag("RUN_GARNET_REAL_QWEN_CACHED_DECODER_LAYER_PARITY"):
    cached_decoder_layer_result = run_garnet_text_decoder_layer_cached_last_token_parity(garnet, key_to_file)
cached_prefill_decode_result = None
if env_flag("RUN_GARNET_REAL_QWEN_CACHED_PREFILL_DECODE_PARITY"):
    cached_prefill_decode_result = run_garnet_text_prefill_decode_cached_parity(garnet, key_to_file)
device_kv_prefill_decode_result = None
if env_flag("RUN_GARNET_REAL_QWEN_DEVICE_KV_PREFILL_DECODE_PARITY"):
    device_kv_prefill_decode_result = run_garnet_text_prefill_decode_device_kv_parity(garnet, key_to_file)
native_mrope_result = None
if env_flag("RUN_GARNET_REAL_QWEN_NATIVE_MROPE_PARITY"):
    native_mrope_result = run_native_mrope_positions_against_hf(model_dir)
native_mrope_cos_sin_result = None
if env_flag("RUN_GARNET_REAL_QWEN_NATIVE_MROPE_COS_SIN_PARITY"):
    native_mrope_cos_sin_result = run_native_mrope_cos_sin_against_hf(model_dir)
text_decoder_layer_hf_result = None
if env_flag("RUN_GARNET_REAL_QWEN_HF_LAYER_PARITY"):
    text_decoder_layer_hf_result = run_text_decoder_layer_against_hf_module(garnet, model_dir, key_to_file)
two_text_decoder_layers_hf_result = None
if env_flag("RUN_GARNET_REAL_QWEN_TWO_LAYER_PARITY"):
    two_text_decoder_layers_hf_result = run_two_text_decoder_layers_against_hf_modules(garnet, model_dir, key_to_file)
two_layer_logits_result = None
if env_flag("RUN_GARNET_REAL_QWEN_LOGITS_PARITY"):
    two_layer_logits_result = run_two_layer_logits_against_hf_modules(garnet, model_dir, key_to_file)
full_prompt_logits_result = None
if env_flag("RUN_GARNET_REAL_QWEN_FULL_PROMPT_LOGITS_PARITY"):
    full_prompt_logits_result = run_full_prompt_window_logits_against_hf_modules(garnet, model_dir, key_to_file)
model_forward_facade_result = None
if env_flag("RUN_GARNET_REAL_QWEN_MODEL_FORWARD_FACADE_PARITY"):
    model_forward_facade_result = run_model_forward_facade_against_hf_modules(garnet, model_dir, key_to_file)
model_forward_native_rope_facade_result = None
if env_flag("RUN_GARNET_REAL_QWEN_MODEL_FORWARD_NATIVE_ROPE_FACADE_PARITY"):
    model_forward_native_rope_facade_result = run_model_forward_facade_native_rope_against_hf_modules(
        garnet, model_dir, key_to_file
    )
model_forward_native_rope_decode_result = None
if env_flag("RUN_GARNET_REAL_QWEN_MODEL_FORWARD_NATIVE_ROPE_DECODE"):
    model_forward_native_rope_decode_result = run_model_forward_facade_native_rope_decode(garnet, model_dir, key_to_file)
    if env_flag("GARNET_QWEN_NATIVE_ROPE_DECODE_ONLY"):
        import json
        print(json.dumps(model_forward_native_rope_decode_result, indent=2))
        raise SystemExit(0)
processor_token_logits_result = None
if env_flag("RUN_GARNET_REAL_QWEN_PROCESSOR_TOKEN_LOGITS_PARITY"):
    processor_token_logits_result = run_processor_token_logits_against_hf_modules(garnet, model_dir, key_to_file)
visual_splice_logits_result = None
if env_flag("RUN_GARNET_REAL_QWEN_VISUAL_SPLICE_LOGITS_PARITY"):
    visual_splice_logits_result = run_visual_splice_logits_against_hf_modules(garnet, model_dir, key_to_file)
vision_patch_embed_result = None
if env_flag("RUN_GARNET_REAL_QWEN_VISION_PATCH_PARITY"):
    vision_patch_embed_result = run_vision_patch_embed_from_processor(garnet, key_to_file)
vision_patch_merger_result = None
if env_flag("RUN_GARNET_REAL_QWEN_VISION_MERGER_PARITY"):
    vision_patch_merger_result = run_vision_patch_merger_from_processor(garnet, key_to_file)
vision_patch_pos_merger_result = None
if env_flag("RUN_GARNET_REAL_QWEN_VISION_POS_MERGER_PARITY"):
    vision_patch_pos_merger_result = run_vision_patch_pos_merger_from_processor(garnet, key_to_file)
vision_qkv_proj_result = None
if env_flag("RUN_GARNET_REAL_QWEN_VISION_QKV_PARITY"):
    vision_qkv_proj_result = run_vision_qkv_proj_from_patch_pos(garnet, key_to_file)
vision_attention_core_result = None
if env_flag("RUN_GARNET_REAL_QWEN_VISION_ATTENTION_PARITY"):
    vision_attention_core_result = run_vision_attention_core_from_patch_pos(garnet, model_dir, key_to_file)
vision_block0_result = None
if env_flag("RUN_GARNET_REAL_QWEN_VISION_BLOCK_PARITY"):
    vision_block0_result = run_vision_block0_from_patch_pos(garnet, model_dir, key_to_file)
vision_all_blocks_merger_result = None
if env_flag("RUN_GARNET_REAL_QWEN_VISION_ALL_BLOCKS_MERGER_PARITY"):
    vision_all_blocks_merger_result = run_vision_all_blocks_merger_from_processor(garnet, model_dir, key_to_file)
text_result = run_text_mlp(garnet, key_to_file)
vision_result = run_vision_mlp(garnet, key_to_file)

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
summary_path = ARTIFACT_DIR / "real_qwen_mlp_subgraph_parity.json"
summary_path.write_text(
    __import__("json").dumps(
        {
            "model_dir": str(model_dir),
            "garnet": str(garnet_dll),
            "text_rms_norm": text_norm_result,
            "vision_layer_norm": vision_norm_result,
            "text_qkv_proj": text_qkv_result,
            "text_qkv_head_norm": text_qkv_head_norm_result,
            "text_o_proj": text_o_proj_result,
            "text_rope_apply": text_rope_result,
            "text_attention_core": text_attention_result,
            "text_decoder_layer_chain": text_decoder_layer_result,
            "cached_decoder_layer": cached_decoder_layer_result,
            "cached_prefill_decode": cached_prefill_decode_result,
            "device_kv_prefill_decode": device_kv_prefill_decode_result,
            "native_mrope_positions": native_mrope_result,
            "native_mrope_cos_sin": native_mrope_cos_sin_result,
            "text_decoder_layer_hf_module": text_decoder_layer_hf_result,
            "two_text_decoder_layers_hf_modules": two_text_decoder_layers_hf_result,
            "two_layer_logits": two_layer_logits_result,
            "full_prompt_logits": full_prompt_logits_result,
            "model_forward_facade": model_forward_facade_result,
            "model_forward_native_rope_facade": model_forward_native_rope_facade_result,
            "model_forward_native_rope_decode": model_forward_native_rope_decode_result,
            "processor_token_logits": processor_token_logits_result,
            "visual_splice_logits": visual_splice_logits_result,
            "vision_patch_embed": vision_patch_embed_result,
            "vision_patch_merger": vision_patch_merger_result,
            "vision_patch_pos_merger": vision_patch_pos_merger_result,
            "vision_qkv_proj": vision_qkv_proj_result,
            "vision_attention_core": vision_attention_core_result,
            "vision_block0": vision_block0_result,
            "vision_all_blocks_merger": vision_all_blocks_merger_result,
            "text_mlp": text_result,
            "vision_mlp": vision_result,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print(f"model_dir={model_dir}")
print(f"garnet={garnet_dll}")
print(f"text_rms_norm={text_norm_result}")
print(f"vision_layer_norm={vision_norm_result}")
print(f"text_qkv_proj={text_qkv_result}")
print(f"text_qkv_head_norm={text_qkv_head_norm_result}")
print(f"text_o_proj={text_o_proj_result}")
print(f"text_rope_apply={text_rope_result}")
print(f"text_attention_core={text_attention_result}")
print(f"text_decoder_layer_chain={text_decoder_layer_result}")
if native_mrope_result is not None:
    print(f"native_mrope_positions={native_mrope_result}")
if native_mrope_cos_sin_result is not None:
    print(f"native_mrope_cos_sin={native_mrope_cos_sin_result}")
if text_decoder_layer_hf_result is not None:
    print(f"text_decoder_layer_hf_module={text_decoder_layer_hf_result}")
if two_text_decoder_layers_hf_result is not None:
    print(f"two_text_decoder_layers_hf_modules={two_text_decoder_layers_hf_result}")
if two_layer_logits_result is not None:
    print(f"two_layer_logits={two_layer_logits_result}")
if full_prompt_logits_result is not None:
    print(f"full_prompt_logits={full_prompt_logits_result}")
if model_forward_facade_result is not None:
    print(f"model_forward_facade={model_forward_facade_result}")
if model_forward_native_rope_facade_result is not None:
    print(f"model_forward_native_rope_facade={model_forward_native_rope_facade_result}")
if model_forward_native_rope_decode_result is not None:
    print(f"model_forward_native_rope_decode={model_forward_native_rope_decode_result}")
if processor_token_logits_result is not None:
    print(f"processor_token_logits={processor_token_logits_result}")
if visual_splice_logits_result is not None:
    print(f"visual_splice_logits={visual_splice_logits_result}")
if vision_patch_embed_result is not None:
    print(f"vision_patch_embed={vision_patch_embed_result}")
if vision_patch_merger_result is not None:
    print(f"vision_patch_merger={vision_patch_merger_result}")
if vision_patch_pos_merger_result is not None:
    print(f"vision_patch_pos_merger={vision_patch_pos_merger_result}")
if vision_qkv_proj_result is not None:
    print(f"vision_qkv_proj={vision_qkv_proj_result}")
if vision_attention_core_result is not None:
    print(f"vision_attention_core={vision_attention_core_result}")
if vision_block0_result is not None:
    print(f"vision_block0={vision_block0_result}")
if vision_all_blocks_merger_result is not None:
    print(f"vision_all_blocks_merger={vision_all_blocks_merger_result}")
print(f"text_mlp={text_result}")
print(f"vision_mlp={vision_result}")
print(f"summary={summary_path}")
print("Phase 05 real Qwen3-VL MLP subgraph parity passed.")
