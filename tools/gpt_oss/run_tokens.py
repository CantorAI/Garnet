"""XLang3 driver: run_tokens.py original-weights cache request.json result.json.

The host provides Harmony token IDs. This single-device correctness driver
releases the prefill engine before loading decode to avoid two resident copies
of the 120B expert weights. It does not start an HTTP server.
"""
import sys
import json
import time
from pathlib import Path
import garnet as G

assert len(sys.argv) == 5, 'expected original checkpoint directory, cache, request JSON, result JSON'
weights, cache = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
request_file, result_file = Path(sys.argv[3]), Path(sys.argv[4])
request = json.loads(request_file.read_text())
config = json.loads((weights / 'config.json').read_text())
assert 'num_experts' in config, 'Use the original/ checkpoint, not the Transformers layout'
ids = request['input_ids']
limit = int(request.get('max_new_tokens', 32))
assert ids and all(isinstance(token, int) and 0 <= token < config['vocab_size'] for token in ids)
assert 1 <= limit <= 256, 'correctness driver supports 1..256 generated tokens'
assert len(ids) + limit <= 4096, 'correctness driver limits total KV capacity to 4096 tokens'
stops = request.get('stop_token_ids', [])
root = Path(__file__).resolve().parents[2] / 'xModel' / 'gpt_oss' / '120b'
pages = (len(ids) + limit + 15) // 16
shape = [config['num_hidden_layers'], pages, 16, config['num_key_value_heads'], config['head_dim']]
count = 1
for dimension in shape:
    count *= dimension


def tensor(values, dtype, dimensions):
    return G.tensor_from_host(values, dtype=dtype, shape=dimensions, device='cuda')


keys = G.tensor_to_bfloat16(tensor([0.] * count, 'float32', shape))
values = G.tensor_to_bfloat16(tensor([0.] * count, 'float32', shape))
table = tensor(list(range(pages)), 'int32', [1, pages])
length = tensor([len(ids)], 'int32', [1])
slot = tensor([0], 'int32', [1])
active = tensor([1], 'int32', [1])


def load(name, tokens):
    model = G.load_model(str(root / (name + '.py')), runtime_mode='compiled_xmodel',
        backend='tensorrt', precision='bf16',
        entry_function='GptOssPrefill' if name == 'prefill' else 'GptOssDecode',
        weights=str(weights), cache_dir=str(cache / (name + '-' + str(tokens))),
        input_shapes=[[1, tokens], [1, tokens], shape, shape, [1, pages], [1], [1], [1]],
        input_dtypes=['int64', 'int64', 'bfloat16', 'bfloat16', 'int32', 'int32', 'int32', 'int32'],
        compile={'builder_workspace_mb': 1024, 'builder_optimization_level': 1})
    status = model.runtime_status()
    assert status['ready'], (status['error_code'], status['error_message'])
    return model


started = time.perf_counter()
model = load('prefill', len(ids))
load_seconds = time.perf_counter() - started
prefill_started = time.perf_counter()
result = model.forward({'inputs': [tensor(ids, 'int64', [1, len(ids)]),
    tensor(list(range(len(ids))), 'int64', [1, len(ids)]), keys, values, table, length, slot, active], 'sample': 'greedy'})
assert result['status'] == 'ok', result
generated = [int(result['token_id'])]
prefill_seconds = time.perf_counter() - prefill_started
assert model.release_runtime()
del result
decode_seconds = 0.
if limit > 1 and generated[-1] not in stops:
    model = load('decode', 1)
    token = tensor([generated[-1]], 'int64', [1, 1])
    position = tensor([len(ids)], 'int64', [1, 1])
    for offset in range(1, limit):
        index = len(ids) + offset - 1
        G.tensor_update_from_host(token, [generated[-1]])
        G.tensor_update_from_host(position, [index])
        G.tensor_update_from_host(length, [index + 1])
        G.tensor_update_from_host(slot, [index])
        started = time.perf_counter()
        result = model.forward({'inputs': [token, position, keys, values, table, length, slot, active], 'sample': 'greedy'})
        assert result['status'] == 'ok', result
        decode_seconds += time.perf_counter() - started
        generated.append(int(result['token_id']))
        if generated[-1] in stops:
            break
    model.release_runtime()
result_file.write_text(json.dumps({'token_ids': generated, 'prompt_tokens': len(ids),
    'load_seconds': load_seconds, 'prefill_seconds': prefill_seconds, 'decode_seconds': decode_seconds,
    'validation': 'experimental correctness baseline; pretrained parity and performance require GPU testing'}))
print('Generated', len(generated), 'tokens; result:', str(result_file), flush=True)
