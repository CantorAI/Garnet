"""Run with XLang3 after compiled_parity.py: fixture-directory cache-directory."""
import sys
from pathlib import Path
import garnet as G

weights, cache = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
root = Path(__file__).resolve().parents[2] / 'xModel' / 'gpt_oss' / '120b'


def load(source, directory):
    return G.load_model(str(source), runtime_mode='compiled_xmodel', backend='tensorrt',
        precision='bf16', entry_function='GptOssPrefill', weights=str(weights),
        input_shapes=[[2, 3], [2, 3], [2, 2, 16, 2, 8], [2, 2, 16, 2, 8], [2, 1], [2], [2], [2]],
        input_dtypes=['int64', 'int64', 'bfloat16', 'bfloat16', 'int32', 'int32', 'int32', 'int32'],
        cache_dir=str(directory), compile={'builder_workspace_mb': 256, 'builder_optimization_level': 1})


model = load(root / 'prefill.py', cache / 'prefill')
assert model.runtime_status()['ready'], model.runtime_status()
plugins = model.runtime_status()['execution_plan']['operator_plugins']
assert len(plugins) == 1 and plugins[0]['id'] == 'gpt_oss'
assert plugins[0]['abi'] == 1 and plugins[0]['library_digest']
assert plugins[0]['module'] == 'garnet_gpt_oss'
for case in ['abi_mismatch', 'backend_mismatch', 'missing_plugin', 'missing_operator', 'invalid_list', 'undeclared']:
    failed = load(weights / 'cases' / case / 'prefill.py', cache / 'invalid' / case)
    status = failed.runtime_status()
    assert not status['ready'], case
    expected_code = 'undeclared_plugin_operator' if case == 'undeclared' else 'operator_plugin_requirement_failed'
    assert status['error_code'] == expected_code, (case, status['error_code'], status['error_message'])
    print(case, status['error_code'], status['error_message'], flush=True)
print('operator-plugin-requirements-passed', flush=True)
