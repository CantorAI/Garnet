"""Run local/remote correctness checks using an already-built Garnet runtime."""
import argparse
import os
import subprocess
import sys
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--runtime-dir', required=True, type=Path)
parser.add_argument('--work-dir', required=True, type=Path)
parser.add_argument('--tensorrt-root', type=Path)
parser.add_argument('--multi-gpu', action='store_true', help='also test a real two-GPU pipeline')
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
runtime, work = args.runtime_dir.resolve(), args.work_dir.resolve()
work.mkdir(parents=True, exist_ok=True)
suffix = '.exe' if os.name == 'nt' else ''
executable = runtime / ('xlang3' + suffix)
environment = os.environ.copy()
paths = [str(runtime)]
if args.tensorrt_root:
    paths.append(str(args.tensorrt_root.resolve() / ('bin' if os.name == 'nt' else 'lib')))
environment['PATH'] = os.pathsep.join(paths + [environment.get('PATH', '')])
if os.name != 'nt':
    environment['LD_LIBRARY_PATH'] = os.pathsep.join(paths + [environment.get('LD_LIBRARY_PATH', '')])


def run(command, label, env=environment):
    with (work / (label + '.log')).open('w', encoding='utf-8') as log:
        completed = subprocess.run([str(x) for x in command], cwd=runtime, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
    print(label, 'passed' if completed.returncode == 0 else 'FAILED', flush=True)
    if completed.returncode:
        print((work / (label + '.log')).read_text(errors='replace')[-5000:])
        raise SystemExit(completed.returncode)


fixture = work / 'fixture'
run([sys.executable, repo / 'test2026/gpt_oss/make_fixture.py', fixture], 'fixture')
run([sys.executable, repo / 'test2026/gpt_oss/placement.py'], 'placement')
run([runtime / ('garnet_gpt_oss_kernel_parity' + suffix)], 'kernel-parity')
run([executable, repo / 'test2026/gpt_oss/compiled_parity.py', fixture, work / 'cache'], 'compiled-parity-cold')
run([executable, repo / 'test2026/gpt_oss/compiled_parity.py', fixture, work / 'cache'], 'compiled-parity-warm')
run([executable, repo / 'test2026/gpt_oss/plugin_requirements.py', fixture, work / 'cache'], 'plugin-requirements')
run([executable, repo / 'tools/gpt_oss/run_tokens.py', fixture, work / 'generation-cache',
     fixture / 'request.json', work / 'generation.json'], 'token-generation')
generated = json.loads((work / 'generation.json').read_text())
expected = json.loads((fixture / 'expected.json').read_text())
assert generated['token_ids'] == expected['generated'], (generated['token_ids'], expected['generated'])
if args.multi_gpu:
    for phase in ('cold', 'warm'):
        run([executable, repo / 'test2026/gpt_oss/multi_gpu.py', fixture, work / 'pipeline-cache'],
            'multi-gpu-' + phase)
    run([executable, repo / 'tools/gpt_oss/run_pipeline_tokens.py', fixture, work / 'pipeline-generation-cache',
         fixture / 'request.json', work / 'pipeline-generation.json'], 'pipeline-token-generation')
    generated = json.loads((work / 'pipeline-generation.json').read_text())
    assert generated['token_ids'] == expected['generated']
no_plugins = environment.copy()
no_plugins['GARNET_OPERATOR_PLUGIN_REGISTRY'] = str(work / 'missing-registry.json')
run([executable, repo / 'test2026/gpt_oss/plugin_requirements.py', fixture, work / 'cache'], 'requirements-without-registry', no_plugins)
run([executable, repo / 'test2026/gpt_oss/native_module.py'], 'native-module-bridge', no_plugins)
run([executable, repo / 'test2026/gpt_oss/qwen_compatibility.py', work / 'unused-qwen-cache'], 'qwen-compatibility', no_plugins)
run([executable, repo / 'test/xlang3/native_api.py'], 'native-api', no_plugins)
print('All GPT-OSS correctness checks passed. Full pretrained validation remains separate.')
