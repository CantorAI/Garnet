"""Run with xlang3.exe run_contracts.py BACKEND CACHE_DIRECTORY."""
import os
import sys

import garnet
import tensor


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    require(len(sys.argv) == 3, __doc__)
    backend = sys.argv[1]
    require(backend in ("tensorrt", "openvino"), "invalid backend")
    root = os.path.abspath(sys.argv[2])
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_registered.py")
    cases = [
        ("dtype_count", [[2, 3]], ["float32", "float32"], "invalid_symbolic_input_dtype"),
        ("negative_dimension", [[2, -1]], ["float32"], "invalid_symbolic_input_shape"),
        ("unsupported_dtype", [[2, 3]], ["uint8"], "invalid_symbolic_input_dtype"),
    ]
    for name, shapes, dtypes, error_code in cases:
        model = garnet.load_model(
            source, runtime_mode="compiled_xmodel", entry_function="forward",
            backend=backend, precision="bf16", input_shapes=shapes,
            input_dtypes=dtypes, cache_dir=os.path.join(root, name))
        status = model.runtime_status()
        require(not status["ready"] and status["error_code"] == error_code, str(status))
        if name == "dtype_count":
            # Tiny CPU storage: invalid capacity must be rejected before GPU work.
            qkv = garnet.tensor_from_bfloat16_bits(
                tensor.tensor([0, 0, 0], shape=[1, 3], dtype=tensor.uint16), device="cpu")
            pages = garnet.tensor_from_bfloat16_bits(
                tensor.tensor([0], dtype=tensor.uint16), device="cpu")
            for implementation in ("reference", "split", "flash"):
                probe = model.debug_probe("paged_kv_bf16", {
                    "qkv": qkv, "key_pages": pages, "value_pages": pages,
                    "page_table": tensor.tensor([0, 0], dtype=tensor.int32),
                    "token_count": 1, "start_position": 0, "sequence_length": 1,
                    "page_size": 1073741824, "q_heads": 1, "kv_heads": 1,
                    "head_dim": 1, "implementation": implementation})
                require(probe["status"] == "error" and
                        probe["error_code"] == "paged_kv_probe_capacity_out_of_range", str(probe))
            print("garnet-native-capacity-contract-passed", backend, flush=True)
        require(model.release_runtime(), "release failed after invalid input contract")
        print("garnet-native-contract-passed", backend, name, flush=True)

    # These frontends must reject absent tokenizer assets instead of pretending
    # initialization succeeded. Tiny one-input profiles avoid allocating KV caches.
    for frontend in ("qwen3_text", "qwen3_vl", "qwen3_asr", "qwen3_tts"):
        model = garnet.load_model(
            source, runtime_mode="compiled_xmodel", entry_function="forward",
            backend=backend, precision="bf16", input_shapes=[[2, 3]],
            input_dtypes=["float32"], frontend=frontend,
            cache_dir=os.path.join(root, frontend),
            compile={"builder_workspace_mb": 64, "builder_optimization_level": 0})
        status = model.runtime_status()
        require(not status["ready"] and status["error_code"] == "frontend_preparation_failed",
                str(status))
        require(bool(status["error_message"]), str(status))
        require(not status["frontend_prepared"], str(status))
        require(not status["engines_prepared"], "failed initialization retained prepared engine state: " + str(status))
        require(model.release_runtime(), "release failed after frontend initialization failure")
        print("garnet-native-frontend-contract-passed", backend, frontend, flush=True)
    print("garnet-native-contract-suite-passed", backend, flush=True)


if __name__ == "__main__":
    main()
