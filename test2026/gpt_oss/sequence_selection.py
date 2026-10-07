"""Compiled mixed-count selection, updates and cold/warm reload parity."""
from pathlib import Path
import sys
import garnet as G

source = Path(__file__).with_name('sequence_selection_model.py')
cache = Path(sys.argv[1])
values = [float(i) for i in range(4 * 4 * 3)]
for reload in range(2):
    model = G.load_model(str(source), runtime_mode='compiled_xmodel',
        backend='tensorrt', precision='bf16', entry_function='SelectLastValid',
        cache_dir=str(cache), input_shapes=[[4, 4, 3], [4]],
        input_dtypes=['float32', 'int32'])
    if not model.runtime_status()['ready']:
        raise RuntimeError(str(model.runtime_status()))
    try:
        x = G.tensor_from_host(values, dtype='float32', shape=[4, 4, 3], device='cuda')
        counts = G.tensor_from_host([1, 4, 2, 3], dtype='int32', shape=[4], device='cuda')
        for wanted in ([1, 4, 2, 3], [4, 1, 3, 2]):
            G.tensor_update_from_host(counts, wanted)
            reply = model.forward({'inputs': [x, counts]})
            if reply['status'] != 'ok':
                raise RuntimeError(str(reply))
            actual = G.tensor_to_cpu(reply['output']).tolist()
            expected = [value for row, count in enumerate(wanted)
                for value in values[(row * 4 + count - 1) * 3:(row * 4 + count) * 3]]
            if actual != expected:
                raise AssertionError((actual, expected))
    finally:
        model.release_runtime()
print('Mixed last-valid sequence counts, updates, cold/warm reload: exact PASS')
