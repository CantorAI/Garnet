"""Load validated TP2 prepacked engines sequentially, without GPU inference.

Host Python: GARNET_REFERENCE NEW_DIRECTORY.
Requires the opt-in TensorRT11 memory diagnostic in the built Garnet library.
Reported weight/context bytes are engine statistics, not allocation peaks.
"""
import json
import os
from pathlib import Path
import argparse
import re
import subprocess
import sys

repo = Path(__file__).resolve().parents[2]
root = Path(os.environ.get('CANTORAI_ROOT', repo.parent))
from resident_budget import (native_identity, hardware_identity, kernel_environment,
                             checkpoint_identity, plan_identity, file_sha256)

if len(sys.argv) == 4 and sys.argv[1] == '--runtime':
    import garnet as G
    from pipeline import make_tensor_parallel_plan, build_tensor_parallel
    reference_path, output = map(Path, sys.argv[2:])
    reference = json.loads(reference_path.read_text())
    if output.exists():
        raise FileExistsError(output)
    weights = Path(os.environ.get('GARNET_GPT_OSS_WEIGHTS', root / 'models/gpt-oss-120b/original'))
    cache = Path(os.environ.get('GARNET_GPT_OSS_CACHE', root / 'work/gpt-oss-tp-cache'))
    devices = json.loads(G.cuda_devices_json())[:2]
    plan = make_tensor_parallel_plan(weights, devices, batch=reference['batch'],
        capacity=reference['max_context_tokens_per_request'],
        tokens=min(reference['prefill_chunk_tokens'] or reference['input_tokens_per_request'],
                   reference['input_tokens_per_request']))
    if not plan['marlin_prepacked']:
        raise ValueError('Resident budget probe requires validated prepacked constants')
    if plan['hardware'] != reference['hardware']:
        raise ValueError('Recorded and current hardware differ; regenerate the profile')
    model = None
    kv = None
    samples = []
    def memory():
        values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'], text=True)
        return [int(value.strip()) for value in values.splitlines()]
    try:
        for prefill in (True, False):
            before = memory()
            model = build_tensor_parallel(weights, cache, plan,
                plan['max_tokens'] if prefill else 1, prefill, kv,
                last_token_logits=prefill,
                padded_prefill=prefill and os.environ.get('GARNET_RESIDENT_PADDED_PREFILL') == '1')
            after = memory()
            samples.append(dict(phase='prefill' if prefill else 'decode',
                                memory_before_mib=before, memory_after_load_mib=after))
            kv = [(stage['keys'], stage['values']) for stage in model.stages]
            model.release()
            model = None
    finally:
        if model is not None:
            model.release()
    build = Path(os.environ.get('GARNET_BUILD_DIR', root / 'out/build/gpt-oss'))
    output.write_text(json.dumps(dict(plan=plan, phases=samples,
        resident_profile_schema=1, plan_identity=plan_identity(plan),
        native_binaries=native_identity(build), hardware_csv=hardware_identity(),
        kernel_environment=kernel_environment(), checkpoint=checkpoint_identity(weights),
        cache_root=str(cache.resolve()), padded_prefill=os.environ.get('GARNET_RESIDENT_PADDED_PREFILL') == '1',
        measurement='Sequential engine loads only; no inference or paired residency',
        limits='Sampled process memory and engine/context bounds do not include all execution peaks'), indent=2))
    sys.exit(0)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference', type=Path)
parser.add_argument('directory', type=Path)
parser.add_argument('--padded-prefill', action='store_true')
parser.add_argument('--prepacked-candidate', action='store_true',
    help='explicitly profile packed engines for a completed original-layout workload; no quality claim')
args = parser.parse_args()
reference_path, directory = args.reference, args.directory
reference = json.loads(reference_path.read_text())
if reference.get('profile_decode_steps') or reference.get('profile_prefill'):
    raise ValueError('Expected unprofiled completed reference')
if directory.exists():
    raise FileExistsError(directory)
env = os.environ.copy()
for key in list(env):
    if key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_')):
        del env[key]
for key, value in reference['optimization_environment'].items():
    if (not key.startswith(('GARNET_GPT_OSS_', 'GARNET_TP_', 'GARNET_BATCH_')) or
            key.endswith(('_TOKEN', '_KEY', '_SECRET', '_PASSWORD')) or not isinstance(value, str)):
        raise ValueError('Invalid recorded optimization environment')
    env[key] = value
if args.prepacked_candidate:
    env['GARNET_GPT_OSS_MARLIN_PREPACKED'] = '1'
if env.get('GARNET_GPT_OSS_MARLIN_PREPACKED') != '1':
    raise ValueError('Reference must be from a completed prepacked workload')
build = Path(env.get('GARNET_BUILD_DIR', root / 'out/build/gpt-oss'))
tensorrt = Path(env.get('GARNET_TENSORRT_ROOT', root / 'ThirdPartySDK/TensorRT'))
env['GARNET_TRT_LOG_ENGINE_MEMORY'] = '1'
env['GARNET_RESIDENT_PADDED_PREFILL'] = str(int(args.padded_prefill))
env['XLANG3_PYTHON_LIB'] = str(root / 'ThirdPartySDK/Python-3.14.0/Lib')
env['PYTHONPATH'] = str(build / 'bin') + os.pathsep + env.get('PYTHONPATH', '')
env['LD_LIBRARY_PATH'] = os.pathsep.join([str(build / 'bin'), str(tensorrt / 'lib'),
                                        '/usr/local/nvidia/lib64', env.get('LD_LIBRARY_PATH', '')])
directory.mkdir(parents=True)
report = directory / 'engine-loads.json'
log = directory / 'engine-loads.log'
lock = root / 'work/gpu-benchmark.lock'
# Keep this budget inspection exclusive despite not issuing model inference.
command = ['flock', '-n', str(lock), 'bash', '-c',
           '[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || exit 1; exec "$@"',
           'engine-profile', str(build / 'bin/xlang3'), str(Path(__file__).resolve()),
           '--runtime', str(reference_path.resolve()), str(report.resolve())]
if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
    raise RuntimeError('Another GPU process is active')
with log.open('x') as handle:
    subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
statistics = {}
pattern = re.compile(r'\[TRTEngineMemory\] device=(\d+) total_weights_bytes=(\d+) '
                     r'context_device_memory_upper_bound_bytes=(\d+) engine=(.+)')
for match in pattern.finditer(log.read_text()):
    device, weights, context, path = match.groups()
    phase = 'prefill' if '/prefill/' in path else 'decode' if '/decode/' in path else None
    if phase is None:
        raise ValueError('Unknown engine phase')
    record = dict(device=int(device), phase=phase, total_weights_bytes=int(weights),
        context_device_memory_upper_bound_bytes=int(context), engine_path=path,
        engine_sha256=file_sha256(path))
    key = (int(device), phase)
    if key in statistics and statistics[key] != record:
        raise ValueError('Inconsistent repeated engine statistics')
    statistics[key] = record
if len(statistics) != 4:
    raise ValueError('Expected exactly four engine statistics')
statistics = list(statistics.values())
result = json.loads(report.read_text())
result['engine_statistics'] = statistics
result['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
result['reference_result'] = str(reference_path.resolve())
result['explicit_prepacked_candidate_override'] = args.prepacked_candidate
report.write_text(json.dumps(result, indent=2))
print(json.dumps(statistics, indent=2))
print('Sequential engine statistics recorded; paired resident admission remains unproven')
