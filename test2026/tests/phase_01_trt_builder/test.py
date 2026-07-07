import os
import sys
import numpy as np

import xlang

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../../.."))
garnet_dll_path = os.environ.get(
    "GARNET_DLL_PATH",
    os.path.join(repo_root, "out", "build", "x64-Debug", "bin", "garnet.dll"),
)
if hasattr(os, "add_dll_directory"):
    for dll_dir in [
        os.path.dirname(garnet_dll_path),
        os.path.join(os.path.dirname(repo_root), "xlang", "out", "build", "x64-Debug", "bin"),
    ]:
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)

try:
    garnet = xlang.importModule("garnet", fromPath=garnet_dll_path)
except Exception as e:
    print(f"Failed to import Garnet via xlang: {e}")
    sys.exit(1)

xmodel_path = os.path.join(script_dir, "simple_matmul.x")

# Mock weights
weights = {"W": np.random.rand(128, 128).astype(np.float32)}

print("Testing TRT Engine compilation and execution...")
try:
    # If the C++ bindings are not implemented yet, this will throw an error and FAIL the test.
    engine = garnet.load_model(
        xmodel_path, 
        weights=weights,
        cache_dir=os.path.join(script_dir, "cache"),
        input_shapes=[[1, 128]],
        weight_shape=[128, 128],
    )
    
    # Prepare inputs
    a_np = np.random.rand(1, 128).astype(np.float32)
    
    # Run engine. Weights are owned by the loaded model, so forward only takes input.
    output = engine.forward(a_np)
    
    # Ground truth validation
    expected = np.matmul(a_np, weights["W"])
    
    to_numpy = getattr(output, "numpy", None)
    to_list = getattr(output, "tolist", None)
    if callable(to_numpy):
        out_np = to_numpy()
    elif callable(to_list):
        out_np = np.array(to_list(), dtype=np.float32)
    else:
        raise AssertionError("TRT output has no numpy/tolist conversion")
    out_np = out_np.reshape(expected.shape)
    np.testing.assert_allclose(out_np, expected, rtol=1e-3, atol=1e-3)
    
    print("Phase 01: TRT Builder test passed! (dummy execution)")
    sys.exit(0)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"TRT Builder failed: {e}")
    sys.exit(1)
