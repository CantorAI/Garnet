"""Replay a recorded shape for exact-order metadata or batch-router candidates.

Python: REQUEST EXPECTED GARNET_REFERENCE NEW_RESULT_DIR [router].
Uses the vLLM environment for validation; preserves all logs and token evidence.
"""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

if len(sys.argv) not in (5,6) or (len(sys.argv)==6 and sys.argv[5]!='router'):
    raise SystemExit('Expected REQUEST EXPECTED GARNET_REFERENCE NEW_RESULT_DIR [router]')
request_path, expected_path, reference_path, directory = map(Path, sys.argv[1:5])
router = len(sys.argv)==6
flag = 'GARNET_GPT_OSS_ROUTER_QUERY_TILE' if router else 'GARNET_GPT_OSS_PARALLEL_MARLIN_METADATA'
label = 'router' if router else 'parallel'
choices = (0,2,4) if router else (0,1)
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
repo = Path(__file__).resolve().parents[2]
root = Path(env.get('CANTORAI_ROOT', repo.parent))
tokenizer = env.get('GARNET_GPT_OSS_TOKENIZER', str(root / 'models/gpt-oss-120b-hf'))
results = []
for value in choices:
    env[flag] = str(value)
    prefix = directory / f'{label}{value}-b{batch}-o{output}'
    result, validation = Path(str(prefix) + '.json'), Path(str(prefix) + '.validation.json')
    print(f'Starting {flag}={value}, batch={batch}, input={len(request["input_ids"])}, '
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
              'flag': flag,
              'batch': batch, 'input': len(request['input_ids']), 'output': output,
              'context': reference['max_context_tokens_per_request'], 'modes': []}
for value, measured in zip(choices,results):
    matches = [sum(a == b for a, b in zip(baseline, trial['token_ids_by_request']))
               for trial in measured['decode_trials']]
    comparison['modes'].append({label: value, 'exact_baseline_slots_per_trial': matches,
        'median_decode_tok_s': statistics.median(
            trial['decode_aggregate_output_tokens_per_second'] for trial in measured['decode_trials']),
        'prefill_s': measured['prefill_seconds'],
        'full_tok_s': measured['full_request_output_tokens_per_second']})
(directory / 'comparison.json').write_text(json.dumps(comparison, indent=2))
if any(count != batch for mode in comparison['modes'] for count in mode['exact_baseline_slots_per_trial']):
    raise AssertionError('Exact-order candidate altered token trajectories; evidence retained')
print('All baseline/candidate token trajectories match exactly', flush=True)
