"""Quiet per-matrix experiment only after both two-output cohorts are archived."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for name,sha in (
    ('two-body-20261015','b18dd030eccd5e0a6818611f7271daa4b95d050e488ee49facbe547bafe48004'),
    ('two-body-confirm-20261015','495b6a231a98df78f09a3607e89f1ec29733d579a54e28471f0974c692bbe60c')):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
    with (root/(name+'.tar.gz')).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('component',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_two_matrix_probe.py'")
exec(compile(source,__file__,'exec'))
