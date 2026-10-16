"""After both archived FMA trials, quiet sequential two-output GPU experiment."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for name,sha in (
    ('eight-fma-body-20261015','9efd6b5b1f16e983d1ec2283697f35810b45d3d816658b55c567c2c155daf56f'),
    ('eight-fma-body-confirm-20261015','f5a5cfe2440ef2adb2a4d2d4e7c8f5fab00f00cd66949ac069d2c7b668aaa8c9')):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
    with (root/(name+'.tar.gz')).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('component',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_two_body.py'")
exec(compile(source,__file__,'exec'))
