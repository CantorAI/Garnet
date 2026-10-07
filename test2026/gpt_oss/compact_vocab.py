"""Real TP2 compiled greedy parity: full logits vs compact candidate pairs."""
import math
from pathlib import Path
import sys
import garnet as G
repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'python'))
from garnet_pipeline import TensorParallel
cache = Path(sys.argv[1]).resolve()
source = repo / 'xModel/gpt_oss/120b/tp_vocab_test.py'
compat = repo / 'xModel/gpt_oss/120b/tensor_compat.py'
for batch, width in ((3, 7), (128, 100544)):
    rows = []
    for row in range(batch):
        values = [((row * 17 + col * 13) % 997 - 498) * .03125 for col in range(width * 2)]
        if row == 0:
            values[width - 1] = values[width] = 32.
        elif row == 1:
            values[width] = 64.
        elif row == 2:
            values = [math.nan] * (width * 2)
        elif row == 3:
            values = [-math.inf] * (width * 2)
        elif row == 4:
            values = [-0.] * (width * 2)
        elif row == 5:
            values[1] = values[width] = math.inf
        rows.append(values)
    outputs = []
    for compact in (False, True):
        stages, prepared = [], []
        for rank in (0, 1):
            G.cuda_set_device(rank)
            root = cache / f'b{batch}-w{width}-compact{int(compact)}-rank{rank}'
            root.mkdir(parents=True, exist_ok=True)
            (root / 'tensor_compat.py').write_text(compat.read_text())
            text = source.read_text().replace('TP_RANK = 0', f'TP_RANK = {rank}')
            text = text.replace('LOCAL_WIDTH = 7', f'LOCAL_WIDTH = {width}')
            text = text.replace('COMPACT = 0', f'COMPACT = {int(compact)}')
            (root / 'tp_vocab_test.py').write_text(text)
            model = G.load_model(str(root / 'tp_vocab_test.py'), runtime_mode='compiled_xmodel',
                backend='tensorrt', precision='bf16', entry_function='GptOssVocabTest',
                input_shapes=[[batch, 1, width]], input_dtypes=['float32'],
                cache_dir=str(root / 'engine'))
            assert model.runtime_status()['ready'], model.runtime_status()
            x = G.tensor_from_host([value for row in rows for value in row[rank * width:(rank + 1) * width]],
                                   dtype='float32', shape=[batch, 1, width], device='cuda')
            stage = dict(rank=rank, device_id=rank, model=model)
            stages.append(stage)
            request = {'inputs': [x]}
            if rank == 0 and not compact:
                request['sample'] = 'greedy_batch'
            prepared.append((stage, request))
        pipeline = TensorParallel(stages, greedy_candidate_pairs=compact)
        try:
            # Run twice to cover cached graph/sampler state and changing output leases.
            for _ in range(2):
                result = pipeline._run_prepared(prepared, compact_sample=compact)
                ids, scores = list(result['token_ids']), list(result['token_values'])
                if outputs:
                    assert ids == outputs[0][0], (batch, width, compact, ids)
                    assert scores == outputs[0][1], (batch, width, compact, scores)
                    assert [math.copysign(1, x) for x in scores] == [math.copysign(1, x) for x in outputs[0][1]]
                outputs.append((ids, scores))
        finally:
            pipeline.release()
    print('TP2 full/compact exact IDs and values passed', batch, width, flush=True)
