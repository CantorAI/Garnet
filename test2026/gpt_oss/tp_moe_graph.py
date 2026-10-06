"""Compare full MoE output with two concurrently executing expert-parallel ranks."""
import concurrent.futures
import sys
from pathlib import Path
import garnet as G

fixture, cache = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
source = Path(__file__).resolve().parents[2] / 'xModel/gpt_oss/120b/tp_moe_test.py'
models = {}
for rank in (-1, 0, 1):
    root = cache / f'rank{rank}'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'tp_moe_test.py').write_text(source.read_text().replace('TP_RANK = -1', f'TP_RANK = {rank}'))
    G.cuda_set_device(0 if rank < 0 else rank)
    model = G.load_model(str(root / 'tp_moe_test.py'), runtime_mode='compiled_xmodel',
        backend='tensorrt', precision='bf16', entry_function='GptOssTpMoe',
        weights=str(fixture), input_shapes=[[1, 1, 32]], input_dtypes=['float32'],
        cache_dir=str(root / 'engine'))
    assert model.runtime_status()['ready'], model.runtime_status()
    models[rank] = model

def forward(rank):
    G.cuda_set_device(0 if rank < 0 else rank)
    x = G.tensor_from_host([((i * 17) % 31 - 15) / 16 for i in range(32)],
                           dtype='float32', shape=[1, 1, 32], device='cuda')
    result = models[rank].forward({'inputs': [x]})
    assert result['status'] == 'ok', result
    return G.tensor_to_cpu(result['output']).tolist()

try:
    baseline = forward(-1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(forward, (0, 1)))
    for rank, output in enumerate(parallel):
        error = max(abs(a - b) for a, b in zip(output, baseline))
        assert error <= .002, (rank, error, output, baseline)
    print('rank-paired GPT-OSS MoE graph parity passed', flush=True)
finally:
    for model in models.values():
        model.release_runtime()
