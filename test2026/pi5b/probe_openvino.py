"""ARM64 OpenVINO prerequisite probe; not Garnet or model inference."""
import json
import time
import numpy as np
import openvino as ov
from openvino import opset13 as ops

core = ov.Core()
print(json.dumps({'openvino': ov.__version__, 'devices': core.available_devices}))
for batch in (1, 2):
    x = ops.parameter([batch, 64], np.float32, name='input')
    rng = np.random.default_rng(42)
    weights = rng.standard_normal((64, 32)).astype(np.float32)
    data = rng.standard_normal((batch, 64)).astype(np.float32)
    model = ov.Model([ops.relu(ops.matmul(x, ops.constant(weights), False, False))], [x])
    start = time.perf_counter()
    compiled = core.compile_model(model, 'CPU', {'INFERENCE_NUM_THREADS': 2, 'INFERENCE_PRECISION_HINT': 'f32'})
    compile_ms = (time.perf_counter() - start) * 1000
    actual = compiled([data])[0]
    np.testing.assert_allclose(actual, np.maximum(data @ weights, 0), rtol=1e-4, atol=1e-4)
    blob = compiled.export_model()
    imported = core.import_model(blob, 'CPU')
    np.testing.assert_allclose(imported([data])[0], actual, rtol=1e-5, atol=1e-5)
    request = compiled.create_infer_request()
    start = time.perf_counter()
    for _ in range(100):
        request.infer([data])
    print(json.dumps({'batch': batch, 'compile_ms': compile_ms,
                      'mean_infer_ms': (time.perf_counter()-start)*10,
                      'max_abs_error': float(np.max(np.abs(actual-np.maximum(data@weights,0)))),
                      'cache_reload': 'passed'}))

