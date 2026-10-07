"""Target TP2 teacher-forced full input, padded tail and reused-request KV wraps.

Run original or hybrid layout separately, with identical CPU prefix logits.
This synthetic correctness diagnostic is never a throughput benchmark.
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
from kv_layout import kv_memory

weights, cache, output = map(Path, sys.argv[1:4])
batch, chunk = map(int, sys.argv[4:6])
profile_flag = os.environ.get('GARNET_GPT_OSS_PROFILE_HYBRID_KV', '0')
if profile_flag not in ('0','1'): raise ValueError('Invalid hybrid diagnostic profiler flag')
profiler = ctypes.CDLL('libcudart.so') if profile_flag == '1' else None
capturing = False
if output.exists(): raise FileExistsError(output)
expected = json.loads((weights / 'expected.json').read_text())
ids = json.loads((weights / 'request.json').read_text())['input_ids']
if not (1 <= batch <= 512 and chunk in (4,8,16,32) and 32 < len(ids) <= 256):
    raise ValueError('Expected wrap-covering input and bounded synthetic profile')
os.environ.pop('GARNET_GPT_OSS_COMPACT_VOCAB_GREEDY', None)
capacity = ((len(ids) + max(chunk, len(expected['generated'])) + 15) // 16) * 16
plan = make_tensor_parallel_plan(weights, json.loads(G.cuda_devices_json())[:2],
    batch=batch, capacity=capacity, tokens=chunk, reserve_bytes=0)
if any(2 * plan['estimated_per_gpu_bytes'] > s['budget_bytes'] for s in plan['stages']):
    raise ValueError('Synthetic resident fixture exceeds conservative double-engine allowance')
pair = ResidentTensorParallel.build(
    lambda: build_tensor_parallel(weights,cache,plan,chunk,True,last_token_logits=True,padded_prefill=True),
    lambda shared: build_tensor_parallel(weights,cache,plan,1,False,shared))

def run(stages, values, positions, length, slot, prefill):
    inputs = []
    tokens = len(values)
    pages = (capacity + 15) // 16
    for stage in stages:
        previous = G.cuda_set_device(stage['device_id'])
        try:
            def tensor(data,dtype,shape):
                return G.tensor_from_host(data,dtype=dtype,shape=shape,device='cuda')
            inputs.append((tensor(values*batch,'int64',[batch,tokens]),[
                tensor(positions*batch,'int64',[batch,tokens]),
                tensor([b*pages+p for b in range(batch) for p in range(pages)],'int32',[batch,pages]),
                tensor([length]*batch,'int32',[batch]),tensor([slot]*batch,'int32',[batch]),
                tensor([1]*batch,'int32',[batch])]))
        finally: G.cuda_set_device(previous)
    reply = pair.forward_prefill(inputs) if prefill else pair.forward_decode(inputs)
    previous = G.cuda_set_device(pair.prefill_stages[0]['device_id'])
    try: return G.tensor_to_cpu(reply['output']).tolist()
    finally: G.cuda_set_device(previous)

records = []
try:
    for generation in range(2):
        for offset in range(0,len(ids),chunk):
            real = min(chunk,len(ids)-offset)
            if profiler is not None and generation == 0 and offset + chunk >= len(ids):
                if profiler.cudaProfilerStart() != 0: raise RuntimeError('Could not start hybrid diagnostic capture')
                capturing = True
            actual = run(pair.prefill_stages,ids[offset:offset+real]+[0]*(chunk-real),
                list(range(offset,offset+chunk)),offset+real,offset,True)
        for step,wanted in enumerate(expected['generation_logits']):
            if step:
                position = len(ids)+step-1
                actual = run(pair.decode_stages,[expected['generated'][step-1]],
                    [position],position+1,position,False)
            if len(actual) != batch*len(wanted): raise ValueError('Incomplete TP logits')
            errors = [abs(a-b) for a,b in zip(actual,wanted*batch)]
            if not all(error <= .025*(1+abs(b)) for error,b in zip(errors,wanted*batch)):
                raise ValueError('Wrap/padded/phase output exceeds existing CPU-prefix bound')
            records.append(dict(generation=generation,step=step,cpu_logits=wanted,
                tp_logits=actual,maximum_absolute_error=max(errors),within_existing_tolerance=True))
            if capturing and step == 1:
                if profiler.cudaProfilerStop() != 0: raise RuntimeError('Could not stop hybrid diagnostic capture')
                capturing = False
finally:
    try:
        if capturing and profiler.cudaProfilerStop() != 0:
            raise RuntimeError('Could not abort hybrid diagnostic capture')
    finally: pair.release()
if any(records[i]['tp_logits'] != records[i+3]['tp_logits'] for i in range(3)):
    raise ValueError('Reused-request KV changes synthetic trajectories')
kv_bytes,table_bytes = kv_memory(plan)
output.write_text(json.dumps(dict(measurement='Synthetic CPU-prefix KV diagnostic, not performance',
    batch=batch,chunk=chunk,input_tokens=len(ids),padded_tail_tokens=(-len(ids))%chunk,
    profiled_diagnostic=profiler is not None,
    complete_generations=2,plan=plan,shared_kv_bytes=kv_bytes,shared_auxiliary_bytes=table_bytes,
    steps=records),indent=2))
print('Synthetic repeated-request full-input/wrap/padded-tail/shared-bank CPU prefix PASS')
