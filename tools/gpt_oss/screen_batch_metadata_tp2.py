"""Replay a recorded shape for metadata, batch-router or FlashInfer attention candidates.

Python: REQUEST EXPECTED GARNET_REFERENCE NEW_RESULT_DIR [router|flash-prefill|flash-decode|intermediate-tp|prefill-router|prefill-router-wire|prefill-wire|marlin-prepacked].
Uses the vLLM environment for validation; preserves all logs and token evidence.
"""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

if len(sys.argv) not in (5,6) or (len(sys.argv)==6 and sys.argv[5] not in ('router','flash-prefill','flash-decode','intermediate-tp','prefill-router','prefill-router-wire','prefill-wire','marlin-prepacked')):
    raise SystemExit('Expected REQUEST EXPECTED GARNET_REFERENCE NEW_RESULT_DIR [router|flash-prefill|flash-decode|intermediate-tp|prefill-router|prefill-router-wire|prefill-wire|marlin-prepacked]')
request_path, expected_path, reference_path, directory = map(Path, sys.argv[1:5])
mode=sys.argv[5] if len(sys.argv)==6 else 'metadata'
router = mode=='router'
flash = mode in ('flash-prefill','flash-decode')
inner = mode=='intermediate-tp'
prepacked = mode=='marlin-prepacked'
wire_only = mode=='prefill-wire'
prefill_wire = mode in ('prefill-router-wire','prefill-wire')
prefill_router = mode in ('prefill-router','prefill-router-wire','prefill-wire')
flag = ('GARNET_GPT_OSS_MARLIN_PREPACKED' if prepacked else
        'GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS' if inner else
        'GARNET_GPT_OSS_DECODE_FLASHINFER' if mode=='flash-decode' else
        'GARNET_GPT_OSS_PREFILL_FLASHINFER' if flash else
        'GARNET_GPT_OSS_ROUTER_QUERY_TILE' if router else 'GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA')
label = 'packed' if prepacked else 'prefill' if prefill_router else 'inner' if inner else 'flash' if flash else 'router' if router else 'parallel'
choices = (1,2) if wire_only else (0,1,2) if prefill_wire else (0,2,4) if router else (0,1)
reference = json.loads(reference_path.read_text())
request = json.loads(request_path.read_text())
if reference['input_token_ids'] != request['input_ids']:
    raise ValueError('Request must exactly match the recorded input IDs')
if reference.get('profile_decode_steps') or reference.get('profile_prefill'):
    raise ValueError('Reference must be unprofiled')
batch, output = reference['batch'], reference['output_tokens_per_request']
if not (16 <= batch <= 512 and 16 <= output <= 2048):
    raise ValueError('Invalid batch/output for exact-order candidate')
if not expected_path.is_file():
    raise FileNotFoundError(expected_path)
if directory.exists():
    raise FileExistsError(f'Refusing to overwrite evidence: {directory}')
directory.mkdir(parents=True)
env = os.environ.copy()
for key in list(env):
    if key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_')):
        del env[key]
for key, value in reference['optimization_environment'].items():
    if (not key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_')) or
            key.endswith(('_TOKEN', '_KEY', '_SECRET', '_PASSWORD')) or not isinstance(value, str)):
        raise ValueError(f'Invalid recorded optimization setting: {key}')
    env[key] = value
env.pop('GARNET_BENCH_NSYS_OUTPUT', None)
env.pop('GARNET_BENCH_REFERENCE_JSON', None)
env.update(GARNET_GPT_OSS_PROFILE_DECODE_STEPS='0',
           GARNET_GPT_OSS_PROFILE_PREFILL='0', GARNET_BATCH_DECODE_TRIALS='3')
if prefill_router and (env.get('GARNET_GPT_OSS_TP_MOE_INTERMEDIATE_SHARDS') != '1' or
                    env.get('GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS') != '0'):
    raise ValueError('Prefill router/wire screen requires a recorded intermediate-axis TP2 reference')
repo = Path(__file__).resolve().parents[2]
root = Path(env.get('CANTORAI_ROOT', repo.parent))
tokenizer = env.get('GARNET_GPT_OSS_TOKENIZER', str(root / 'models/gpt-oss-120b-hf'))
results = []
for value in choices:
    if prefill_router:
        env['GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE'] = str(int(value >= 1))
        env['GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE'] = str(int(value == 2))
    else:
        env[flag] = str(value)
    if inner:
        env['GARNET_GPT_OSS_TP_EXPERT_WEIGHT_SHARDS'] = str(1-value)
    prefix = directory / f'{label}{value}-b{batch}-o{output}'
    result, validation = Path(str(prefix) + '.json'), Path(str(prefix) + '.validation.json')
    settings = ({name: env[name] for name in ('GARNET_GPT_OSS_PREFILL_ROUTER_TENSORCORE',
                'GARNET_GPT_OSS_BF16_PREFILL_ALLREDUCE')} if prefill_router else {flag: env[flag]})
    print(f'Starting {settings}, batch={batch}, input={len(request["input_ids"])}, '
          f'output={output}, reference={reference_path}', flush=True)
    with Path(str(prefix) + '.log').open('x') as log:
        subprocess.run(['bash', str(repo / 'tools/gpt_oss/benchmark_batch_tp2.sh'),
                        str(request_path.resolve()), str(result.resolve()), str(batch), str(output)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(repo / 'tools/gpt_oss/validate_batch_results.py'),
                    tokenizer, str(result.resolve()), str(expected_path.resolve()), str(validation.resolve())],
                   env=env, check=True)
    measured = json.loads(result.read_text())
    if (measured['input_token_ids'] != request['input_ids'] or measured['batch'] != batch or
            measured['output_tokens_per_request'] != output or
            measured['max_context_tokens_per_request'] != reference['max_context_tokens_per_request']):
        raise AssertionError('Measured shape differs from the recorded reference')
    if len(measured['decode_trials']) != 3:
        raise AssertionError('Expected three complete decode trials')
    for trial in measured['decode_trials']:
        rows = trial['token_ids_by_request']
        if len(rows) != batch or any(len(row) != output for row in rows):
            raise AssertionError('Incomplete output token evidence')
    results.append(measured)
    print('prefill_s', measured['prefill_seconds'], 'decode_tok_s',
          [trial['decode_aggregate_output_tokens_per_second'] for trial in measured['decode_trials']],
          'full_tok_s', measured['full_request_output_tokens_per_second'], flush=True)
baseline = results[0]['decode_trials'][0]['token_ids_by_request']
comparison = {'reference_result': str(reference_path.resolve()),
              'flag': 'BF16 prefill wire (router fixed on)' if wire_only else 'prefill router then BF16 wire' if prefill_wire else 'prefill router' if prefill_router else flag,
              'require_exact_trajectories': wire_only or not (flash or inner or prefill_router),
              'batch': batch, 'input': len(request['input_ids']), 'output': output,
              'context': reference['max_context_tokens_per_request'], 'modes': []}
for value, measured in zip(choices,results):
    matches = [sum(a == b for a, b in zip(baseline, trial['token_ids_by_request']))
               for trial in measured['decode_trials']]
    comparison['modes'].append({label: value, 'exact_baseline_slots_per_trial': matches,
        'optimization_environment': measured['optimization_environment'],
        'exact_repeat_trials': all(trial['token_ids_by_request'] == measured['decode_trials'][0]['token_ids_by_request']
                                  for trial in measured['decode_trials']),
        'median_decode_tok_s': statistics.median(
            trial['decode_aggregate_output_tokens_per_second'] for trial in measured['decode_trials']),
        'prefill_s': measured['prefill_seconds'],
        'full_tok_s': measured['full_request_output_tokens_per_second'],
        'sampled_peak_gpu_memory_mib': measured['sampled_peak_gpu_memory_mib'],
        'kv_cache_allocated_bytes_per_gpu': measured['kv_cache_allocated_bytes_per_gpu'],
        'weight_storage_estimate': measured.get('weight_storage_estimate'),
        'prefill_build_seconds': measured['prefill_build_seconds'],
        'decode_build_seconds': measured['decode_build_seconds']})
if prefill_wire:
    reference_trials = results[-2]['decode_trials']
    wire_matches = [sum(a == b for a,b in zip(first['token_ids_by_request'],second['token_ids_by_request']))
                    for first,second in zip(reference_trials,results[-1]['decode_trials'])]
    comparison['wire_only_exact_slots_per_trial'] = wire_matches
(directory / 'comparison.json').write_text(json.dumps(comparison, indent=2))
if prefill_wire and any(count != batch for count in wire_matches):
    raise AssertionError('BF16 wire altered pretrained trajectories; evidence retained')
if not (flash or inner or prefill_router) and any(count != batch for mode in comparison['modes'] for count in mode['exact_baseline_slots_per_trial']):
    raise AssertionError('Exact-order candidate altered token trajectories; evidence retained')
print('All expected answers passed; numerical-order trajectory comparison retained' if flash or inner or prefill_router else
      'All baseline/candidate token trajectories match exactly', flush=True)
