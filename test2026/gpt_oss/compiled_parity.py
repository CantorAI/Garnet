"""Run using XLang3: compiled_parity.py fixture-directory cache-directory.

Checks actual graph compilation, cold/warm engine loading, cached decode and
batch isolation against the independent reference made by make_fixture.py.
"""
import sys
import json
from pathlib import Path
import garnet as G

weights, cache = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
root = Path(__file__).resolve().parents[2] / 'xModel' / 'gpt_oss' / '120b'
expected = json.loads((weights / 'expected.json').read_text())
page_shape = [2, 2, 16, 2, 8]


def load(name, batch, tokens):
    functions = {'prefill': 'GptOssPrefill', 'decode': 'GptOssDecode', 'decode_batch': 'GptOssDecodeBatch'}
    model = G.load_model(str(root / (name + '.py')), runtime_mode='compiled_xmodel',
        backend='tensorrt', precision='bf16', entry_function=functions[name], weights=str(weights),
        input_shapes=[[batch, tokens], [batch, tokens], page_shape, page_shape,
                      [batch, 1], [batch], [batch], [batch]],
        input_dtypes=['int64', 'int64', 'bfloat16', 'bfloat16', 'int32', 'int32', 'int32', 'int32'],
        cache_dir=str(cache / name), compile={'builder_workspace_mb': 256, 'builder_optimization_level': 1})
    assert model.runtime_status()['ready'], model.runtime_status()
    return model


def tensor(values, dtype, shape):
    return G.tensor_from_host(values, dtype=dtype, shape=shape, device='cuda')


keys = G.tensor_to_bfloat16(tensor([0.] * 1024, 'float32', page_shape))
values = G.tensor_to_bfloat16(tensor([0.] * 1024, 'float32', page_shape))


def run(model, ids, positions, lengths, starts, active, pages):
    batch, tokens = len(ids), len(ids[0])
    inputs = [tensor([v for row in ids for v in row], 'int64', [batch, tokens]),
              tensor([v for row in positions for v in row], 'int64', [batch, tokens]), keys, values,
              tensor(pages, 'int32', [batch, 1]), tensor(lengths, 'int32', [batch]),
              tensor(starts, 'int32', [batch]), tensor(active, 'int32', [batch])]
    result = model.forward({'inputs': inputs, 'return_logits': 1})
    assert result['status'] == 'ok', result
    return G.tensor_to_cpu(result['output']).tolist()


def compare(actual, target, label):
    # BF16 reductions and tensor-core accumulation may differ by a few ULP.
    maximum = 0.
    reference = [value for row in target for value in row]
    assert len(actual) == len(reference), (len(actual), len(reference))
    for value, expected_value in zip(actual, reference):
        error = abs(value - expected_value)
        maximum = max(maximum, error)
        assert error <= .025 * (1 + abs(expected_value)), (label, value, expected_value)
    print(label, 'max_absolute_error', maximum, flush=True)


prefill = load('prefill', 2, 3)
actual = run(prefill, [row[:3] for row in expected['ids']], [[0, 1, 2], [0, 1, 2]], [3, 3], [0, 0], [1, 1], [0, 1])
compare(actual, [row for batch in expected['logits'] for row in batch[:3]], 'compiled prefill + batch isolation')
decode = load('decode_batch', 2, 1)
actual = run(decode, [[row[3]] for row in expected['ids']], [[3], [3]], [4, 4], [3, 3], [1, 1], [0, 1])
compare(actual, [batch[3] for batch in expected['logits']], 'compiled cached batch decode')
saved_keys = G.tensor_to_cpu(keys).tolist()
saved_values = G.tensor_to_cpu(values).tolist()
run(decode, [[4], [3]], [[3], [3]], [4, 4], [3, 3], [1, 0], [0, 1])
assert G.tensor_to_cpu(keys).tolist() == saved_keys
assert G.tensor_to_cpu(values).tolist() == saved_values
decode.release_runtime()
decode = load('decode_batch', 2, 1)
actual = run(decode, [[row[3]] for row in expected['ids']], [[3], [3]], [4, 4], [3, 3], [1, 1], [0, 1])
compare(actual, [batch[3] for batch in expected['logits']], 'reload/refit from cached engine')
print('gpt-oss-compiled-parity-passed', flush=True)
