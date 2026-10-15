"""Require verified installed prototype sources, then quiet native validation."""
from pathlib import Path
import hashlib,json,sys

root=Path(sys.argv[1]).resolve();manifest=json.loads((root/'mlp-prefill-native-staging/manifest.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
assert json.loads((root/'logs/stage-rounded-mlp-graph-20261015/exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_guarded_mlp_prefill_native_checks.py')
exec(compile(source,__file__,'exec'))
