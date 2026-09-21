# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run with xlang3.exe run_models.py BACKEND CACHE_DIRECTORY [all|build|reload]."""
import os
import sys

import garnet
import tensor


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_output(result, shape, expected):
    require(result["status"] == "ok", str(result))
    cpu = garnet.tensor_to_cpu(result["output"])
    require(tuple(cpu.shape) == tuple(shape), str(cpu.shape))
    actual = cpu.tolist()
    require(len(actual) == len(expected), str(actual))
    for index in range(len(expected)):
        require(abs(actual[index] - expected[index]) <= 0.0001,
                "value " + str(index) + ": " + str(actual) + " != " + str(expected))
    return result["output"]


def load_case(source, shapes, cache, backend, **extra):
    return garnet.load_model(
        source, runtime_mode="compiled_xmodel", entry_function="forward",
        input_shapes=shapes, input_dtypes=["float32"] * len(shapes),
        cache_dir=cache, backend=backend, precision="bf16",
        compile={"builder_workspace_mb": 64, "builder_optimization_level": 0},
        **extra)


def exercise(case, backend, cache_root, reload):
    name, shapes, values, output_shape, expected, next_values, next_expected = case
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")
    cache = os.path.join(cache_root, name)
    graph_cache = os.path.join(cache, "runtime_graph.cache")
    require(os.path.isfile(graph_cache) == reload,
            "Use a fresh cache directory for build, and its existing cache for reload: " + cache)
    model = load_case(source, shapes, cache, backend)
    status = model.runtime_status()
    require(status["ready"], str(status))
    require(status["backend"] == backend, str(status))
    counters = status["forbidden_path_counters"]
    if reload:
        require(counters["graph_cache_hits"] == 1, str(status))
        require(counters["root_x_executions"] == 0, str(status))
        require(status["state"] == "engine_cache_loaded", str(status))
    else:
        require(counters["graph_cache_misses"] == 1, str(status))
        require(counters["root_x_executions"] == 1, str(status))
    require(counters["hardcoded_qwen_runner_calls"] == 0, str(status))
    require(counters["python_subgraph_calls"] == 0, str(status))
    require(counters["direct_internal_export_calls"] == 0, str(status))
    require(os.path.isfile(graph_cache), graph_cache)
    require(os.path.isfile(status["engine_path"]), str(status))
    plan = status["execution_plan"]
    require(len(plan["operations"]) > 0, str(plan))
    require(len(plan["regions"]) > 0, str(plan))

    inputs = [tensor.tensor(values[i], shape=shapes[i], dtype=tensor.float32)
              for i in range(len(shapes))]
    held_output = check_output(model.forward({"inputs": inputs}), output_shape, expected)
    next_inputs = [tensor.tensor(next_values[i], shape=shapes[i], dtype=tensor.float32)
                   for i in range(len(shapes))]
    check_output(model.forward({"inputs": next_inputs}), output_shape, next_expected)
    # An independently returned tensor must survive later calls and model release.
    check_output({"status": "ok", "output": held_output}, output_shape, expected)
    invalid = model.forward(None)
    require(invalid["status"] == "error" and
            invalid["error_code"] == "invalid_compiled_request", str(invalid))
    check_output(model.forward({"inputs": inputs}), output_shape, expected)
    malformed_requests = [
        ({"inputs": []}, "compiled_input_count_mismatch"),
        ({"inputs": inputs + [inputs[0]]}, "compiled_input_count_mismatch"),
        ({"inputs": [None] + inputs[1:]}, "compiled_input_not_tensor"),
        ({"inputs": [tensor.tensor([1], shape=[1], dtype=tensor.float32)] + inputs[1:]},
         "compiled_input_shape_mismatch"),
    ]
    for request, error_code in malformed_requests:
        invalid = model.forward(request)
        require(invalid["status"] == "error", "malformed request accepted: " + str(invalid))
        require(invalid["error_code"] == error_code, str(invalid))
        check_output(model.forward({"inputs": next_inputs}), output_shape, next_expected)
        check_output({"status": "ok", "output": held_output}, output_shape, expected)
    require(model.release_runtime(), "release_runtime failed")
    require(model.release_runtime(), "repeated release_runtime failed")
    status = model.runtime_status()
    require(status["mode"] == "compiled_xmodel" and status["state"] == "released"
            and not status["ready"], str(status))
    released = model.forward({"inputs": inputs})
    require(released["status"] == "error" and
            released["error_code"] == "compiled_graph_not_ready", str(released))
    check_output({"status": "ok", "output": held_output}, output_shape, expected)
    print("garnet-native-model-passed", backend, name, "reload" if reload else "build", flush=True)


CASES = [
    ("tiny_add", [[2, 3], [2, 3]],
     [[1, -2, 3, 4, 0, 6], [10, 20, -3, 0, 5, -6]], [2, 3],
     [11, 18, 0, 4, 5, 0],
     [[0, 0, 0, 0, 0, 0], [-1, 2, -3, 4, -5, 6]], [-1, 2, -3, 4, -5, 6]),
    ("tiny_matmul", [[2, 3], [3, 2]],
     [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], [2, 2],
     [58, 64, 139, 154],
     [[-1, 0, 2, 3, -2, 1], [1, 0, 0, 1, 2, -1]], [3, -2, 5, -3]),
    ("tiny_registered", [[2, 3]], [[-4, -0.5, 0, 1, 2, 8]], [2, 3],
     [0, 0, 0, 1, 2, 8], [[3, -2, 5, -8, 0, 0.25]], [3, 0, 5, 0, 0, 0.25]),
]


def main():
    require(len(sys.argv) in (3, 4), __doc__)
    backend = sys.argv[1]
    require(backend in ("tensorrt", "openvino"), "backend must be tensorrt or openvino")
    cache_root = os.path.abspath(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) == 4 else "all"
    require(mode in ("all", "build", "reload"), "mode must be all, build or reload")
    for case in CASES:
        if mode != "reload":
            exercise(case, backend, cache_root, False)
        if mode != "build":
            exercise(case, backend, cache_root, True)
    print("garnet-native-model-suite-passed", backend, mode, flush=True)


if __name__ == "__main__":
    main()
