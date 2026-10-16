"""Require the exact archived guard-retry failure before resuming native checks."""
from pathlib import Path
import hashlib,json,sys

root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/mlp-prefill-native-checks-guardfix-20261015/exit.json').read_text())['exit_code']==1
with (root/'mlp-prefill-native-checks-guardfix-20261015.tar.gz').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='6d9fda15edcd9bf2e101918ed7615925b8f3d5f98cfc016ec34cd1da806b2147'
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','resume_mlp_prefill_native_checks.py')
exec(compile(source,__file__,'exec'))
