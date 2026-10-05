"""Use the existing Qwen capture checks without its stale total-module count."""
import sys
import importlib
from pathlib import Path
import garnet as G

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / 'test' / 'xlang3' / 'models'))
import run_production_capture as existing

reference_roots = [item for item in existing.ROOTS if item[0].startswith('text_1_7b.') or item[0].startswith('vl_2b_instruct.')]
for layers in [1, 2]:
    for module_name, entry in reference_roots:
        module = importlib.import_module('xModel.qwen3.' + module_name)
        existing.capture(module, entry, layers)
print('existing Qwen text/VL capture checks passed:', len(reference_roots), 'entries at two layer counts', flush=True)

# A missing plugin registry must not affect models with no plugin requirement.
model = G.load_model(str(repo / 'xModel/qwen3/text_1_7b/prefill.py'),
    runtime_mode='compiled_xmodel', backend='tensorrt', entry_function='Qwen3Prefill',
    weights='', input_shapes=[[1, 1], [1, 1, 1], [1, 1], [1, 1, 16, 1, 8],
                             [1, 1, 16, 1, 8], [1], [1]],
    input_dtypes=['int64', 'int64', 'int64', 'bfloat16', 'bfloat16', 'int32', 'int32'],
    cache_dir=str(Path(sys.argv[1]).resolve()))
status = model.runtime_status()
assert status['error_code'] == 'model_spec_weights_unavailable', status['error_code']
print('Qwen without requires bypasses plugin registry', flush=True)
