"""Exercise persistent GPU inputs and reusable TensorRT output ownership."""
import os
import sys
import garnet
import tensor

source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_add.py")
model = garnet.load_model(
    source, runtime_mode="compiled_xmodel", entry_function="forward",
    input_shapes=[[2, 3], [2, 3]], input_dtypes=["float32", "float32"],
    cache_dir=sys.argv[1], backend="tensorrt", precision="bf16",
    compile={"builder_workspace_mb": 64, "builder_optimization_level": 0})
assert model.runtime_status()["ready"]
pairs = []
for marker in [2, 7]:
    pairs.append([
        garnet.tensor_to_gpu(tensor.tensor([marker] * 6, shape=[2, 3], dtype=tensor.float32)),
        garnet.tensor_to_gpu(tensor.tensor([3] * 6, shape=[2, 3], dtype=tensor.float32))])
first = None
for iteration in range(1000):
    index = iteration % 2
    result = model.forward({"inputs": pairs[index], "reuse_output": True})
    assert result["status"] == "ok", str(result)
    output = result["output"]
    if first is None:
        first = output
    else:
        assert output is first, "reusable output acquired another owner wrapper"
    if iteration % 100 == 0 or iteration == 999:
        expected = 5 if index == 0 else 10
        assert garnet.tensor_to_cpu(output).tolist() == [expected] * 6
assert model.release_runtime()
assert garnet.tensor_to_cpu(first).tolist() == [10] * 6
print("garnet-trt-bounded-reuse-passed: 1000 calls, one output identity", flush=True)
