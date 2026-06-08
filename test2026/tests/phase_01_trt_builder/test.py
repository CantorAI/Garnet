import os
import sys
import numpy as np

import xlang

script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to the freshly built garnet.dll
garnet_dll_path = os.path.abspath(os.path.join(script_dir, "../../../../out/build/x64-Debug/bin/garnet.dll"))
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
        input_shapes=[[1, 128], [128, 128]]
    )
    
    # Prepare inputs
    a_np = np.random.rand(1, 128).astype(np.float32)
    b_np = weights["W"]
    a = garnet.tensor(a_np)
    b = garnet.tensor(b_np)
    
    # Run engine
    output = engine.forward(a, b)
    
    # Ground truth validation
    expected = np.matmul(a_np, b_np)
    
    out_np = output.numpy()
    np.testing.assert_allclose(out_np, expected, rtol=1e-3, atol=1e-3)
    
    print("Phase 01: TRT Builder test passed! (dummy execution)")
    sys.exit(0)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"TRT Builder failed: {e}")
    sys.exit(1)
