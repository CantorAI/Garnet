"""Inspect TP2 numerical error with independent CPU-reference token prefixes.

This does not replace exact free-generation or compact/full equivalence gates.
"""
import json
import ctypes
import os
from pathlib import Path
import sys
import garnet as G
repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'tools/gpt_oss'))
from pipeline import make_tensor_parallel_plan, build_tensor_parallel
from garnet_pipeline import ResidentTensorParallel
weights, cache, output = map(Path, sys.argv[1:4])
batch = int(sys.argv[4]) if len(sys.argv)>4 else 1
if not 1<=batch<=512:
    raise ValueError('Teacher-forced batch must be from1 to512')
if output.exists():
    raise FileExistsError(output)
expected = json.loads((weights / 'expected.json').read_text())
profile_step = int(os.environ.get('GARNET_GPT_OSS_PROFILE_TEACHER_STEP', '-1'))
if not -1<=profile_step<len(expected['generation_logits']):
    raise ValueError('Invalid teacher-forced profiler step')
profiler = ctypes.CDLL('libcudart.so') if profile_step>=0 else None
ids = json.loads((weights / 'request.json').read_text())['input_ids']
padded_tokens = int(os.environ.get('GARNET_GPT_OSS_TEACHER_PADDED_TOKENS', '0'))
if padded_tokens and not len(ids) < padded_tokens <= 16:
    raise ValueError('Padded teacher prefill must cover real tokens within context16')
prefill_tokens = padded_tokens or len(ids)
os.environ.pop('GARNET_GPT_OSS_COMPACT_VOCAB_GREEDY', None)
plan = make_tensor_parallel_plan(weights, json.loads(G.cuda_devices_json())[:2],
                                 batch=batch, capacity=16, tokens=prefill_tokens, reserve_bytes=0)
def tensor(values, dtype, shape):
    return G.tensor_from_host(values, dtype=dtype, shape=shape, device='cuda')
G.cuda_set_device(0)
table = tensor(list(range(batch)), 'int32', [batch, 1])
active = tensor([1]*batch, 'int32', [batch])
model = build_tensor_parallel(weights, cache, plan, prefill_tokens, True, last_token_logits=True,
                              padded_prefill=bool(padded_tokens))
resident = os.environ.get('GARNET_GPT_OSS_TEACHER_RESIDENT') == '1'
resident_pair = None
if resident:
    if any(2 * plan['estimated_per_gpu_bytes'] > stage['budget_bytes'] for stage in plan['stages']):
        model.release()
        raise ValueError('Synthetic resident fixture exceeds conservative double-engine budget')
    from garnet_pipeline import ResidentTensorParallel
    try:
        prefill_model = model
        resident_pair = ResidentTensorParallel.build(lambda: prefill_model,
            lambda shared: build_tensor_parallel(weights, cache, plan, 1, False, shared))
    except Exception:
        # Factory owns cleanup, including the existing prefill model.
        raise
steps, kv = [], None
try:
    for step, wanted in enumerate(expected['generation_logits']):
        if step:
            if step == 1:
                if not resident:
                    kv = ResidentTensorParallel.shared_rank_resources(model)
                    model.release()
                    model = build_tensor_parallel(weights, cache, plan, 1, False, kv)
            tokens = [expected['generated'][step - 1]]
            positions = [len(ids) + step - 1]
            start = positions[0]
        else:
            tokens, positions, start = ids, list(range(len(ids))), 0
            if padded_tokens:
                tokens = tokens + [0] * (padded_tokens - len(ids))
                positions = list(range(padded_tokens))
        length = len(ids) + step
        activation=tensor(tokens*batch, 'int64', [batch, len(tokens)])
        controls=[tensor(positions*batch, 'int64', [batch, len(tokens)]), table,
                  tensor([length]*batch, 'int32', [batch]), tensor([start]*batch, 'int32', [batch]), active]
        if step==profile_step and profiler.cudaProfilerStart()!=0:
            raise RuntimeError('Could not start teacher-forced profiler range')
        try:
            if resident_pair is not None:
                # Rank-local preparation mirrors TensorParallel.forward, with
                # the resident owner serializing phase execution.
                stages = resident_pair.prefill_stages if step == 0 else resident_pair.decode_stages
                prepared = []
                for stage in stages:
                    previous = G.cuda_set_device(stage['device_id'])
                    try:
                        prepared.append((G.tensor_to_device(activation, stage['device_id']),
                            [G.tensor_to_device(t, stage['device_id']) for t in controls]))
                    finally:
                        G.cuda_set_device(previous)
                reply = (resident_pair.forward_prefill(prepared) if step == 0
                         else resident_pair.forward_decode(prepared))
            else:
                reply = model.forward(activation,controls)
        finally:
            if step==profile_step and profiler.cudaProfilerStop()!=0:
                raise RuntimeError('Could not stop teacher-forced profiler range')
        G.cuda_set_device(0)
        actual = G.tensor_to_cpu(reply['output']).tolist()
        assert len(actual) == batch*len(wanted)
        errors = [abs(a - b) for a, b in zip(actual, wanted*batch)]
        within = all(e <= .025 * (1 + abs(b)) for e, b in zip(errors, wanted*batch))
        actual_top = sorted(range(len(wanted)), key=lambda i: (-actual[i], i))[:2]
        cpu_top = sorted(range(len(wanted)), key=lambda i: (-wanted[i], i))[:2]
        row = dict(step=step, batch=batch, logits_per_request=len(wanted), cpu_top=cpu_top, tp_top=actual_top,
                   cpu_margin=wanted[cpu_top[0]] - wanted[cpu_top[1]],
                   tp_margin=actual[actual_top[0]] - actual[actual_top[1]],
                   maximum_absolute_error=max(errors), within_existing_tolerance=within,
                   cpu_logits=wanted, tp_logits=actual)
        steps.append(row)
        print({k: v for k, v in row.items() if not k.endswith('logits')}, flush=True)
finally:
    if resident_pair is not None:
        resident_pair.release()
    else:
        model.release()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(dict(measurement='teacher-forced CPU prefixes, diagnostic only', batch=batch,
                                 profile_step=profile_step, padded_prefill_tokens=padded_tokens,
                                 resident_phases=resident, steps=steps), indent=2))
if not all(s['within_existing_tolerance'] for s in steps):
    raise RuntimeError('TP teacher-forced logits exceed the established compiled parity tolerance')
