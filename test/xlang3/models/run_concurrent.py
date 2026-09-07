"""Exercise native model calls sharing an engine across Python threads."""
import os
import sys
import threading
import time
import garnet
import tensor

source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_add.py")
assert len(sys.argv) == 2, "run_concurrent.py CACHE_DIRECTORY"
models = []
engine_path = None
for index in range(12):
    model = garnet.load_model(
        source, runtime_mode="compiled_xmodel", entry_function="forward",
        input_shapes=[[2, 3], [2, 3]], input_dtypes=["float32", "float32"],
        cache_dir=sys.argv[1], backend="tensorrt", precision="bf16",
        compile={"builder_workspace_mb": 64, "builder_optimization_level": 0})
    status = model.runtime_status()
    assert status["ready"], str(status)
    if engine_path is None:
        engine_path = status["engine_path"]
    else:
        assert status["engine_path"] == engine_path, "model instances did not share engine path"
        assert status["forbidden_path_counters"]["graph_cache_hits"] == 1, str(status)
        assert status["forbidden_path_counters"]["root_x_executions"] == 0, str(status)
    models.append(model)

start = threading.Event()
errors = []
completed = []
held_outputs = [None] * 12


def worker(index):
    try:
        inputs = []
        expected = []
        for variant in range(2):
            left = [index * 16 + variant * 7 + element for element in range(6)]
            right = [3 - element * 2 for element in range(6)]
            inputs.append([
                garnet.tensor_to_gpu(tensor.tensor(left, shape=[2, 3], dtype=tensor.float32)),
                garnet.tensor_to_gpu(tensor.tensor(right, shape=[2, 3], dtype=tensor.float32))])
            expected.append([left[element] + right[element] for element in range(6)])
        assert start.wait(30), "concurrent start timed out"
        retained = models[index].forward({"inputs": inputs[0]})
        assert retained["status"] == "ok", str(retained)
        assert garnet.tensor_to_cpu(retained["output"]).tolist() == expected[0]
        first = None
        for iteration in range(100):
            variant = iteration % 2
            result = models[index].forward({"inputs": inputs[variant], "reuse_output": True})
            assert result["status"] == "ok", str(result)
            output = result["output"]
            if first is None:
                first = output
            else:
                assert output is first, "reusable output identity changed"
            actual = garnet.tensor_to_cpu(output).tolist()
            assert actual == expected[variant], str((index, iteration, actual, expected[variant]))
        assert garnet.tensor_to_cpu(retained["output"]).tolist() == expected[0], "retained output overwritten"
        held_outputs[index] = (retained["output"], expected[0], first, expected[1])
        completed.append(index)
    except BaseException as error:
        errors.append(str(index) + ": " + str(error))


threads = []
for index in range(12):
    thread = threading.Thread(target=worker, args=(index,), daemon=True)
    threads.append(thread)
    thread.start()
start.set()
deadline = time.monotonic() + 60
for thread in threads:
    thread.join(max(0, deadline - time.monotonic()))
assert not any(thread.is_alive() for thread in threads), "native concurrent forward timed out"
assert not errors, str(errors)
assert len(completed) == 12, str(completed)
for model in models:
    assert model.release_runtime()
for retained, expected_retained, reused, expected_reused in held_outputs:
    assert garnet.tensor_to_cpu(retained).tolist() == expected_retained, "retained output invalid after release"
    assert garnet.tensor_to_cpu(reused).tolist() == expected_reused, "reusable output invalid after release"
print("garnet-trt-native-concurrency-passed: 1212 calls, 12 model instances", flush=True)
