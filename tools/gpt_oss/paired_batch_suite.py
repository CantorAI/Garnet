"""Run saved batch cases: all vLLM references first, then matched Garnet.

Python: UNPROFILED_GARNET_PROFILE NEW_RESULT_DIR [CASE ...].
Defaults to code-tracing, instruction-following and long-context-retrieval.
The existing arithmetic reference is not rerun unnecessarily. Uses vLLM Python.
"""
import importlib.metadata
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('profile', type=Path)
parser.add_argument('directory', type=Path)
parser.add_argument('cases', nargs='*')
parser.add_argument('--batch', type=int, help='explicit homogeneous batch override, up to512')
parser.add_argument('--output', type=int, help='explicit fixed output length, up to2048')
parser.add_argument('--prefill-chunk', type=int, help='explicit Garnet query chunk length')
parser.add_argument('--context', type=int, help='explicit context reservation; reject shorter requests')
args = parser.parse_args()
reference_path, directory = args.profile, args.directory
names = args.cases or ['code-tracing', 'instruction-following', 'long-context-retrieval']
allowed = {'arithmetic', 'code-tracing', 'instruction-following', 'long-context-retrieval'}
if len(names) != len(set(names)) or any(name not in allowed for name in names):
    raise ValueError('Cases must be unique saved benchmark names')
reference = json.loads(reference_path.read_text())
if reference.get('profile_decode_steps') or reference.get('profile_prefill'):
    raise ValueError('Profile must be an uninstrumented benchmark')
batch = reference['batch'] if args.batch is None else args.batch
output = reference['output_tokens_per_request'] if args.output is None else args.output
if not (1 <= batch <= 512 and 16 <= output <= 2048):
    raise ValueError('Invalid batch/output profile')
if args.prefill_chunk is not None and not 1 <= args.prefill_chunk <= 4096:
    raise ValueError('Invalid prefill chunk')
if args.context is not None and not 1 <= args.context <= 4096:
    raise ValueError('Invalid context reservation')
if directory.exists():
    raise FileExistsError(f'Refusing to overwrite evidence: {directory}')
repo = Path(__file__).resolve().parents[2]
root = Path(os.environ.get('CANTORAI_ROOT', repo.parent))
work = Path(os.environ.get('GARNET_BENCH_WORK_DIR', root / 'work'))
build = Path(os.environ.get('GARNET_BUILD_DIR', root / 'out/build/gpt-oss'))
runtime = Path(os.environ.get('XLANG3_RUNTIME_BIN', root / 'out/build/xlang3/bin/xlang3'))
weights = Path(os.environ.get('GARNET_GPT_OSS_WEIGHTS', root / 'models/gpt-oss-120b/original'))
model = Path(os.environ.get('GARNET_GPT_OSS_TOKENIZER', root / 'models/gpt-oss-120b-hf'))
cache = Path(os.environ.get('GARNET_GPT_OSS_CACHE', work / 'gpt-oss-tp-cache'))
env = os.environ.copy()
for key in list(env):
    if key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_')):
        del env[key]
for key, value in reference['optimization_environment'].items():
    if (not key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_')) or
            key.endswith(('_TOKEN', '_KEY', '_SECRET', '_PASSWORD')) or not isinstance(value, str)):
        raise ValueError(f'Invalid recorded optimization setting: {key}')
    env[key] = value
if args.prefill_chunk is not None:
    env['GARNET_BATCH_PREFILL_CHUNK'] = str(args.prefill_chunk)
for key in ('GARNET_BENCH_NSYS_OUTPUT', 'GARNET_BENCH_REFERENCE_JSON', 'GARNET_BATCH_PLAN_ONLY'):
    env.pop(key, None)
env.update(GARNET_GPT_OSS_PROFILE_DECODE_STEPS='0', GARNET_GPT_OSS_PROFILE_PREFILL='0',
           GARNET_BATCH_DECODE_TRIALS='3')
tensorrt = Path(env.get('GARNET_TENSORRT_ROOT', root / 'ThirdPartySDK/TensorRT'))
env.setdefault('XLANG3_PYTHON_LIB', str(root / 'ThirdPartySDK/Python-3.14.0/Lib'))
env['PYTHONPATH'] = str(build / 'bin') + os.pathsep + env.get('PYTHONPATH', '')
env['LD_LIBRARY_PATH'] = os.pathsep.join([str(build / 'bin'), str(tensorrt / 'lib'),
                                        '/usr/local/nvidia/lib64', env.get('LD_LIBRARY_PATH', '')])
version = importlib.metadata.version('vllm')
if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
    raise RuntimeError('Refusing suite: target GPUs are already busy')
directory.mkdir(parents=True)
manifest = {'garnet_profile': str(reference_path.resolve()), 'vllm_version': version,
            'batch': batch, 'output': output, 'cases': [], 'phase_order': ['admission', 'vllm', 'garnet'],
            'explicit_profile_overrides': {'batch': args.batch, 'output': args.output,
                'prefill_chunk': args.prefill_chunk, 'context': args.context},
            'measurement_limits': 'Garnet full_request_tok_s is execution only, excluding engine build/load and handoff. vLLM measures complete warmed requests. Garnet later decode trials reuse input KV. KV history is estimated, not live scheduler occupancy. See each raw result for timings and limits.'}
for name in names:
    saved = work / ('long-prompt-test' if name == 'long-context-retrieval' else f'prompt-benchmarks/{name}')
    request, expected = saved / 'request.json', saved / 'expected.json'
    ids = json.loads(request.read_text())['input_ids']
    if not expected.is_file():
        raise FileNotFoundError(expected)
    capacity = (max(reference['max_context_tokens_per_request'], ((len(ids) + output + 511) // 512) * 512)
                if args.context is None else args.context)
    if len(ids) + output > capacity:
        raise ValueError('Explicit context reservation is shorter than input plus output')
    if capacity > 4096:
        raise ValueError('Saved request exceeds supported context profile')
    case = {'name': name, 'request': str(request.resolve()), 'expected': str(expected.resolve()),
            'input': len(ids), 'context': capacity, 'batch': batch, 'output': output}
    manifest['cases'].append(case)
manifest_path = directory / 'manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2))

def command_logged(command, log, environment):
    with log.open('x') as stream:
        subprocess.run([str(arg) for arg in command], env=environment,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)

def validate(case, result, validation):
    subprocess.run([sys.executable, repo / 'tools/gpt_oss/validate_batch_results.py',
                    model, result, case['expected'], validation], check=True)
    measured = json.loads(result.read_text())
    request = json.loads(Path(case['request']).read_text())
    if (measured['input_token_ids'] != request['input_ids'] or measured['batch'] != batch or
            measured['output_tokens_per_request'] != output or
            measured['max_context_tokens_per_request'] != case['context']):
        raise AssertionError('Result shape differs from matched workload')
    return measured

for case in manifest['cases']:
    case_env = env.copy()
    case_env.update(GARNET_BATCH_CONTEXT_CAPACITY=str(case['context']), GARNET_BATCH_PLAN_ONLY='1')
    admission = directory / (case['name'] + '.admission.json')
    print('Checking admission', case['name'], 'batch', batch, 'input', case['input'],
          'output', output, 'context', case['context'], flush=True)
    command_logged([runtime, repo / 'tools/gpt_oss/run_tp_batch_throughput.py',
                    weights, cache, case['request'], admission, batch, output],
                   directory / (case['name'] + '.admission.log'), case_env)

for engine in ('vllm', 'garnet'):
    for case in manifest['cases']:
        prefix = directory / (case['name'] + '.' + engine)
        result = Path(str(prefix) + '.json')
        case_env = env.copy()
        case_env['GARNET_BATCH_CONTEXT_CAPACITY'] = str(case['context'])
        print('Starting', engine, case['name'], 'batch', batch, 'input', case['input'],
              'output', output, 'context', case['context'], flush=True)
        if engine == 'vllm':
            case_env.update(VLLM_BATCH_CONTEXT_CAPACITY=str(case['context']),
                            VLLM_BATCH_TRIALS='3', VLLM_BATCH_MAX_BATCHED_TOKENS='8192')
            command = [sys.executable, repo / 'tools/gpt_oss/run_vllm_batch_throughput.py',
                       model, case['request'], result, batch, output]
        else:
            command = ['bash', repo / 'tools/gpt_oss/benchmark_batch_tp2.sh',
                       case['request'], result, batch, output]
        command_logged(command, Path(str(prefix) + '.log'), case_env)
        measured = validate(case, result, Path(str(prefix) + '.validation.json'))
        case[engine] = {'result': str(result.resolve()),
            'median_decode_tok_s': statistics.median(
                trial['decode_aggregate_output_tokens_per_second'] for trial in measured['decode_trials']),
            'full_request_tok_s': measured['full_request_output_tokens_per_second'],
            'peak_gpu_mib': measured['sampled_peak_gpu_memory_mib']}
        if engine == 'garnet':
            if measured['kv_cache_dtype'] != 'bfloat16':
                raise AssertionError('Garnet KV dtype differs from the matched BF16 profile')
            case[engine].update(prefill_seconds=measured['prefill_seconds'],
                prefill_build_load_seconds=measured['prefill_build_seconds'],
                decode_build_load_seconds=measured['decode_build_seconds'],
                kv_allocated_bytes_per_gpu=measured['kv_cache_allocated_bytes_per_gpu'],
                estimated_full_kv_history_bytes_per_gpu=measured['logical_kv_bytes_per_gpu_at_completion'])
        else:
            if measured['vllm_version'] != version:
                raise AssertionError('vLLM version differs from installed-version evidence')
            profiles = measured['kv_profile_after_trials']
            if any(layer['dtype'] != 'torch.bfloat16' for rank in profiles for layer in rank['layers']):
                raise AssertionError('vLLM actual KV dtype differs from matched BF16')
            case[engine].update(
                kv_allocated_bytes_per_gpu=[rank['allocated_unique_backing_bytes'] for rank in profiles],
                estimated_full_kv_history_bytes_per_gpu=[rank['logical_full_history_bytes'] for rank in profiles],
                estimated_retained_kv_history_bytes_per_gpu=[rank['logical_retained_history_bytes'] for rank in profiles],
                median_full_request_tok_s=statistics.median(
                    trial['full_request_output_tokens_per_second'] for trial in measured['decode_trials']),
                ttft_min_median_max_seconds=[
                    [min(trial['request_first_token_seconds']), statistics.median(trial['request_first_token_seconds']),
                     max(trial['request_first_token_seconds'])] for trial in measured['decode_trials']])
        print(case[engine], flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
print('All saved cases passed matched batch validation. Performance target must be assessed separately.', flush=True)
