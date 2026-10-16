"""Quiet API probe after both scheduling cohorts are terminal and archived."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for trial,sha in [('four-threadgroup-probe-20261015','2594e5f9bc1e9caace3176eb5bde5b296e9b9e34aba0ae5a653e96dc20102af5'),
                  ('four-threadgroup-probe-confirm-20261015','5a4a9a56e5bd455dff80908d32541c5d7928b38987216e5f101b87e87661cde7')]:
    assert json.loads((root/'logs'/trial/'exit.json').read_text())['exit_code']==0
    with (root/(trial+'.tar.gz')).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('compat',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_ndarray_compat.py'")
exec(compile(source,__file__,'exec'))
