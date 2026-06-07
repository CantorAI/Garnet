import os
import sys
import numpy as np

import xlang

# We expect Garnet to be importable via xlang
try:
    garnet = xlang.importModule("garnet", fromPath="garnet")
except Exception as e:
    print(f"Failed to import Garnet via xlang: {e}")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
xmodel_path = os.path.join(script_dir, "simple_matmul.x")

# Create a simple .x file on the fly
with open(xmodel_path, "w") as f:
    f.write('''
from garnet import garnet
T = garnet.tensor()

def forward(a, b):
    return a * T.matmul() * b
''')

# Mock weights
weights = {"W": np.random.rand(128, 128).astype(np.float32)}

print("Testing TRT Engine compilation and execution...")
try:
    # If the C++ bindings are not implemented yet, this will throw an error and FAIL the test.
    engine = garnet.load_model(
        xmodel_path, 
        weights=weights,
        cache_dir=os.path.join(script_dir, "cache")
    )
    
    # Prepare inputs
    a = garnet.Tensor(np.random.rand(1, 128).astype(np.float32))
    b = garnet.Tensor(weights["W"])
    
    # Run engine
    output = engine.forward(a, b)
    
    # Ground truth validation
    expected = np.matmul(a.numpy(), b.numpy())
    np.testing.assert_allclose(output.numpy(), expected, rtol=1e-3, atol=1e-3)
    
    print("Phase 01: TRT Builder test passed!")
    sys.exit(0)
except Exception as e:
    print(f"TRT Builder failed: {e}")
    sys.exit(1)
