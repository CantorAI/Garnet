import os
import sys
import numpy as np

import xlang3

try:
    garnet = xlang3.importModule("garnet", fromPath="garnet")
except Exception as e:
    print(f"Failed to import Garnet via xlang: {e}")
    sys.exit(1)

image_dir = os.environ.get("TEST_DATA_IMAGES_DIR", "")
if not os.path.exists(image_dir):
    print(f"Warning: Image directory {image_dir} not found. Ensure config.yaml is correct.")

print("Testing QWen-VL-8B End-to-End inference loop...")

try:
    weights_path = os.path.join(os.path.dirname(__file__), "mock_qwen.safetensors")
    
    # If mock weights don't exist, we can't fully run the test yet, so we exit 0 (skip)
    if not os.path.exists(weights_path):
        print("Mock weights not found. Skipping E2E graph execution until weights are downloaded.")
        sys.exit(0)

    weights = garnet.load_weights(weights_path)

    # Initialize execution engine using the Python model graph.
    xmodel_path = os.path.abspath(os.path.join(__file__, "../../../../../xModel/qwen3/vl_2b_instruct/qwen_vl_model.py"))
    
    engine = garnet.load_model(
        xmodel_path, 
        weights=weights,
        cache_dir="./engine_cache"
    )

    # Prepare mock inputs
    image_tensor = garnet.Tensor(np.random.rand(1, 3, 448, 448).astype(np.float32))
    text_tensor = garnet.Tensor(np.array([[151644, 872, 198]]).astype(np.int32)) # Mock token IDs

    # Forward pass
    logits = engine.forward(image_tensor, text_tensor)
    
    assert logits is not None, "Output logits cannot be null!"
    assert len(logits.shape) == 3, f"Expected 3D logits tensor, got {len(logits.shape)}"

    print("Phase 03: QWen-VL End-to-End test passed!")
    sys.exit(0)
    
except Exception as e:
    print(f"End-to-End execution failed: {e}")
    sys.exit(1)
