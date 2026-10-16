"""Require verified installed prototype sources, then quiet native validation."""
from pathlib import Path
import hashlib,json,sys

root=Path(sys.argv[1]).resolve();manifest=json.loads((root/'mlp-prefill-native-staging/manifest.json').read_text())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
assert json.loads((root/'logs/stage-rounded-mlp-graph-20261015/exit.json').read_text())['exit_code']==0
assert json.loads((root/'logs/mlp-prefill-native-checks-20261015/exit.json').read_text())['exit_code']==1
assert hashlib.file_digest((root/'mlp-prefill-native-checks-20261015.tar.gz').open('rb'),'sha256').hexdigest()=='eada52786b8f2072c83c810d542823ff58ad9519f07c18b6fa526b5f989465fc'
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_guarded_mlp_prefill_native_checks.py')
exec(compile(source,__file__,'exec'))
