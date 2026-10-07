"""Compare full MoE output with two concurrently executing expert-parallel ranks."""
import concurrent.futures
import json
import sys
from pathlib import Path
import garnet as G

fixture, cache = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
expert_weight_shards = len(sys.argv) > 3 and sys.argv[3] == '1'
intermediate_shards = len(sys.argv) > 3 and sys.argv[3] == 'intermediate'
batch = int(sys.argv[4]) if len(sys.argv)>4 else 1
if not 1<=batch<=512:
    raise ValueError('Invalid MoE graph batch')
config=json.loads((fixture/'config.json').read_text())
if expert_weight_shards:
    cache = cache / 'expert-weight-shards'
if intermediate_shards:
    if config['intermediate_size']%64:
        raise ValueError('Intermediate MoE fixture width must be divisible by64')
    cache = cache / 'intermediate-shards' / str(batch)
source = Path(__file__).resolve().parents[2] / 'xModel/gpt_oss/120b/tp_moe_test.py'
models = {}
for rank in (-1, 0, 1):
    root = cache / f'rank{rank}'
    root.mkdir(parents=True, exist_ok=True)
    text = source.read_text().replace('TP_RANK = -1', f'TP_RANK = {rank}')
    text = text.replace('EXPERT_WEIGHT_SHARD = 0',
                        'EXPERT_WEIGHT_SHARD = ' + str(int(expert_weight_shards and rank >= 0)))
    text = text.replace('MOE_INTERMEDIATE_SHARD = 0',
                        'MOE_INTERMEDIATE_SHARD = ' + str(int(intermediate_shards and rank>=0)))
    text = text.replace('MOE_INTERMEDIATE_SIZE = 32','MOE_INTERMEDIATE_SIZE = '+str(config['intermediate_size']))
    (root / 'tp_moe_test.py').write_text(text)
    G.cuda_set_device(0 if rank < 0 else rank)
    model = G.load_model(str(root / 'tp_moe_test.py'), runtime_mode='compiled_xmodel',
        backend='tensorrt', precision='bf16', entry_function='GptOssTpMoe',
        weights=str(fixture), input_shapes=[[batch, 1, 32]], input_dtypes=['float32'],
        cache_dir=str(root / 'engine'))
    assert model.runtime_status()['ready'], model.runtime_status()
    models[rank] = model

def forward(rank):
    G.cuda_set_device(0 if rank < 0 else rank)
    x = G.tensor_from_host([((i * 17) % 31 - 15) / 16 for i in range(batch*32)],
                           dtype='float32', shape=[batch, 1, 32], device='cuda')
    result = models[rank].forward({'inputs': [x]})
    assert result['status'] == 'ok', result
    return G.tensor_to_cpu(result['output']).tolist()

try:
    baseline = forward(-1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(forward, (0, 1)))
    for rank, output in enumerate(parallel):
        error = max(abs(a - b) for a, b in zip(output, baseline))
        tolerance=.025 if intermediate_shards else .002
        if intermediate_shards:
            assert all(abs(a-b)<=tolerance*(1+abs(b)) for a,b in zip(output,baseline)), (rank,error,output,baseline)
        else:
            assert error <= .002, (rank,error,output,baseline)
        print('TP MoE rank',rank,'max_abs',error,'intermediate_shards',intermediate_shards,flush=True)
    print('rank-paired GPT-OSS MoE graph parity passed', flush=True)
finally:
    for model in models.values():
        model.release_runtime()
