"""Capture resident prefill/decode ranges once; timings are diagnostic only.

Host Python: REQUEST EXPECTED UNPROFILED_RESULT NEW_PREFIX [range options].
Keeps one complete warmup/three complete requests and exact token parity.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from resident_budget import file_sha256
from resident_capture import ResidentCapture, existing_capture_artifacts

parser = argparse.ArgumentParser(description=__doc__)
for name in ('request', 'expected', 'reference', 'prefix'):
    parser.add_argument(name, type=Path)
parser.add_argument('--prefill-chunks', help='comma-separated zero-based chunk indices')
parser.add_argument('--decode-ranges', help='comma-separated START:STEPS decode offsets')
args = parser.parse_args()
reference = json.loads(args.reference.read_text())
request = json.loads(args.request.read_text())
if (reference.get('profiled_diagnostic') or reference.get('profile_prefill') or
        reference.get('profile_decode_steps') or not reference.get('resident_profile') or
        reference.get('complete_warmup_count') != 1 or len(reference.get('decode_trials', [])) != 3):
    raise ValueError('Require an uninstrumented resident one-warmup/three-full-trial reference')
if reference['input_token_ids'] != request['input_ids'] or not args.expected.is_file():
    raise ValueError('Request tokens/expected answers do not match the recorded workload')
baseline = reference['decode_trials'][0]['token_ids_by_request']
if any(trial['token_ids_by_request'] != baseline for trial in reference['decode_trials']):
    raise ValueError('Reference complete token matrices must repeat exactly')
measured = Path(reference['resident_profile'])
if file_sha256(measured) != reference['resident_profile_sha256']:
    raise ValueError('Resident engine profile changed after the reference')
batch, output = reference['batch'], reference['output_tokens_per_request']
input_tokens, chunk = len(request['input_ids']), reference['prefill_chunk_tokens']
capacity = reference['max_context_tokens_per_request']
if not (1 <= batch <= 512 and 16 <= output <= 2048 and
        0 < chunk <= input_tokens and input_tokens + output <= capacity <= 4096):
    raise ValueError('Invalid recorded workload')
prefill_chunks = args.prefill_chunks
if prefill_chunks is None:
    prefill_chunks = ','.join(map(str, sorted({0, math.ceil(input_tokens / chunk) - 1})))
decode_ranges = args.decode_ranges
if decode_ranges is None:
    early = min(10, output // 4)
    late = max(early + 3, min(output - 3, output - 32))
    decode_ranges = f'{early}:3,{late}:3'
prefix = args.prefix.resolve()
if existing_capture_artifacts(prefix):
    raise FileExistsError('Refusing to overwrite resident diagnostic evidence')
env = os.environ.copy()
for key in list(env):
    if key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_',
                       'GARNET_RESIDENT_', 'GARNET_BENCH_')):
        del env[key]
for key, value in reference['optimization_environment'].items():
    if (not key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_', 'GARNET_RESIDENT_')) or
            key.endswith(('_TOKEN', '_KEY', '_SECRET', '_PASSWORD')) or not isinstance(value, str)):
        raise ValueError('Invalid recorded optimization environment')
    env[key] = value
env.update(GARNET_RESIDENT_PROFILE=str(measured.resolve()), GARNET_RESIDENT_WARMUPS='1',
    GARNET_BATCH_CONTEXT_CAPACITY=str(capacity), GARNET_BATCH_PREFILL_CHUNK=str(chunk),
    GARNET_BENCH_NSYS_OUTPUT=str(prefix), GARNET_BENCH_RESIDENT_DIAGNOSTIC='1',
    GARNET_GPT_OSS_PROFILE_PREFILL='0', GARNET_GPT_OSS_PROFILE_DECODE_STEPS='0',
    GARNET_GPT_OSS_PROFILE_PREFILL_CHUNKS=prefill_chunks,
    GARNET_GPT_OSS_PROFILE_DECODE_RANGES=decode_ranges)
range_count = (len(prefill_chunks.split(',')) if prefill_chunks else 0) + (
    len(decode_ranges.split(',')) if decode_ranges else 0)
env['GARNET_BENCH_CAPTURE_RANGES'] = str(range_count)
selection = ResidentCapture(env, input_tokens, chunk, output, profiler=object())
repo = Path(__file__).resolve().parents[2]
root = Path(env.get('CANTORAI_ROOT', repo.parent))
tokenizer = env.get('GARNET_GPT_OSS_TOKENIZER', str(root / 'models/gpt-oss-120b-hf'))
prefix.parent.mkdir(parents=True, exist_ok=True)
result = Path(str(prefix) + '.json')
validation = Path(str(prefix) + '.validation.json')
manifest_path = Path(str(prefix) + '.capture-manifest.json')
manifest = dict(measurement='Resident Nsight diagnostic; no throughput/speed-goal claims',
    source_reference=str(args.reference.resolve()), reference_sha256=file_sha256(args.reference),
    request=str(args.request.resolve()), request_sha256=file_sha256(args.request),
    expected=str(args.expected.resolve()), expected_sha256=file_sha256(args.expected),
    requested_capture_ranges=selection.requested, result=str(result), status='running')
manifest_path.write_text(json.dumps(manifest, indent=2))
print('Resident diagnostic capture ranges', selection.requested, flush=True)
try:
    subprocess.run(['bash', str(repo / 'tools/gpt_oss/benchmark_batch_tp2.sh'),
        str(args.request.resolve()), str(result), str(batch), str(output)], env=env, check=True)
    captured = json.loads(result.read_text())
    if not captured.get('profiled_diagnostic') or captured['completed_capture_ranges'] != [
            list(value) for value in selection.requested]:
        raise ValueError('Incomplete or unmarked resident diagnostic')
    for key in ('native_binaries', 'hardware', 'hardware_csv', 'resident_profile_sha256',
                'batch', 'input_token_ids', 'output_tokens_per_request', 'max_context_tokens_per_request',
                'kv_cache_dtype', 'kv_cache_allocated_bytes_per_gpu', 'padded_tail_tokens'):
        if captured[key] != reference[key]:
            raise ValueError('Diagnostic/reference identity mismatch: ' + key)
    if any(trial['token_ids_by_request'] != baseline for trial in captured['decode_trials']):
        raise ValueError('Instrumented complete trajectories differ from the reference')
    subprocess.run([sys.executable, str(repo / 'tools/gpt_oss/validate_batch_results.py'),
        tokenizer, str(result), str(args.expected.resolve()), str(validation)], env=env, check=True)
    reports = sorted(prefix.parent.glob(prefix.name + '*.nsys-rep'))
    if len(reports) != range_count or any(not path.stat().st_size for path in reports):
        raise ValueError('Expected one nonempty Nsight report per requested range')
    manifest.update(status='complete', exact_all_trial_trajectory_parity=True,
        reports=[dict(path=str(path), sha256=file_sha256(path), bytes=path.stat().st_size)
                 for path in reports], result_sha256=file_sha256(result),
        validation_sha256=file_sha256(validation))
except Exception as error:
    manifest.update(status='failed', failure=str(error))
    raise
finally:
    manifest_path.write_text(json.dumps(manifest, indent=2))
