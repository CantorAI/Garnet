"""Replay a recorded Garnet batch configuration with a bounded Nsight range.

Python: REQUEST EXPECTED UNPROFILED_RESULT NEW_OUTPUT_PREFIX START STEPS.
The profiled rates are diagnostic and must not replace unprofiled benchmarks.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

if len(sys.argv) != 7:
    raise SystemExit('Expected REQUEST EXPECTED UNPROFILED_RESULT NEW_OUTPUT_PREFIX START STEPS')
request_path, expected_path, reference_path, prefix = map(Path, sys.argv[1:5])
start, steps = map(int, sys.argv[5:7])
reference = json.loads(reference_path.read_text())
request = json.loads(request_path.read_text())
if reference['input_token_ids'] != request['input_ids']:
    raise ValueError('Profile request must exactly match recorded input IDs')
if reference.get('profiled_diagnostic') or reference.get('profile_decode_steps') or reference.get('profile_prefill'):
    raise ValueError('Reference must be an unprofiled benchmark')
batch, output = reference['batch'], reference['output_tokens_per_request']
if not (1 <= batch <= 512 and 16 <= output <= 2048 and
        1 <= start and 1 <= steps and start + steps <= output):
    raise ValueError('Invalid batch/output/range')
if not expected_path.is_file():
    raise FileNotFoundError(expected_path)
prefix = prefix.resolve()
result = Path(str(prefix) + '.json')
validation = Path(str(prefix) + '.validation.json')
for suffix in ('.json', '.validation.json', '.nsys-rep', '.sqlite', '.qdstrm'):
    if Path(str(prefix) + suffix).exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {prefix}{suffix}')
env = os.environ.copy()
for key in list(env):
    if key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_')):
        del env[key]
for key, value in reference['optimization_environment'].items():
    if (not key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_')) or
            key.endswith(('_TOKEN', '_KEY', '_SECRET', '_PASSWORD')) or
            not isinstance(value, str)):
        raise ValueError(f'Invalid recorded optimization setting: {key}')
    env[key] = value
env.update(GARNET_BENCH_NSYS_OUTPUT=str(prefix),
           GARNET_BENCH_REFERENCE_JSON=str(reference_path.resolve()),
           GARNET_GPT_OSS_PROFILE_DECODE_START=str(start),
           GARNET_GPT_OSS_PROFILE_DECODE_STEPS=str(steps),
           GARNET_GPT_OSS_PROFILE_PREFILL='0',
           GARNET_BATCH_DECODE_TRIALS='1')
repo = Path(__file__).resolve().parents[2]
root = Path(env.get('CANTORAI_ROOT', repo.parent))
tokenizer = env.get('GARNET_GPT_OSS_TOKENIZER', str(root / 'models/gpt-oss-120b-hf'))
prefix.parent.mkdir(parents=True, exist_ok=True)
print(f'Profiling batch={batch}, input={len(request["input_ids"])}, output={output}, '
      f'decode offsets={start}..{start + steps - 1}; report={prefix}', flush=True)
subprocess.run(['bash', str(repo / 'tools/gpt_oss/benchmark_batch_tp2.sh'),
                str(request_path.resolve()), str(result), str(batch), str(output)],
               env=env, check=True)
subprocess.run([sys.executable, str(repo / 'tools/gpt_oss/validate_batch_results.py'),
                tokenizer, str(result), str(expected_path.resolve()), str(validation)],
               env=env, check=True)
