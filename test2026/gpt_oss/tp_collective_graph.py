"""End-to-end TensorRT-plugin test for paired rank execution and NCCL reduction."""
import concurrent.futures
import shutil
import sys
from pathlib import Path
import garnet as G

cache = Path(sys.argv[1]).resolve()
source = Path(__file__).resolve().parents[2] / 'xModel/gpt_oss/120b/tp_all_reduce_test.py'
models = []
for rank in range(2):
    root = cache / f'rank{rank}'
    root.mkdir(parents=True, exist_ok=True)
    model_source = source.read_text().replace('TP_RANK = 0', f'TP_RANK = {rank}')
    (root / 'tp_all_reduce_test.py').write_text(model_source)
    (root / '__init__.py').write_text('')
    shutil.copy2(source.parent / 'tensor_compat.py', root / 'tensor_compat.py')
    G.cuda_set_device(rank)
    model = G.load_model(str(root / 'tp_all_reduce_test.py'),
        runtime_mode='compiled_xmodel', backend='tensorrt', precision='bf16',
        entry_function='GptOssTpCollective', input_shapes=[[1, 1, 8]],
        input_dtypes=['float32'], cache_dir=str(root / 'engine'))
    assert model.runtime_status()['ready'], model.runtime_status()
    models.append(model)


def forward(rank):
    G.cuda_set_device(rank)
    values = [1.25 if rank == 0 else -0.5] * 8
    x = G.tensor_from_host(values, dtype='float32', shape=[1, 1, 8], device='cuda')
    result = models[rank].forward({'inputs': [x]})
    assert result['status'] == 'ok', result
    return G.tensor_to_cpu(result['output']).tolist()


try:
    for iteration in range(5):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            outputs = list(pool.map(forward, range(2)))
        for rank, output in enumerate(outputs):
            assert output == [0.75] * 8, (rank, output)
        print('rank-paired graph all-reduce passed', iteration, flush=True)
finally:
    for model in models:
        model.release_runtime()
print('TensorRT GPT-OSS TP2 collective graph parity passed.', flush=True)
