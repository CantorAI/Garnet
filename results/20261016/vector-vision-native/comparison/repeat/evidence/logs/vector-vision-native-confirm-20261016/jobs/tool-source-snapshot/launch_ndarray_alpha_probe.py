"""Only diagnose coefficient behavior after the failed API gate is frozen."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/ndarray-compat-20261015/exit.json').read_text())['exit_code']==1
with (root/'ndarray-compat-20261015.tar.gz').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='1274759a25ebfdb31746f2f9d8595b8a92348156d1c9259cce3e5e7f8c64116a'
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('compat',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_ndarray_alpha_probe.py'")
exec(compile(source,__file__,'exec'))
