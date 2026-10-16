"""Exclusive component run after frozen full checkpoint and original-codec admission."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
archive=root/'ndarray-weight-admission-20261016.tar.gz'
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='66d403a090e6275e9a623459b33b9a9ffd30de19e9811cc3ce84a36da74f8e4e'
for trial in ('ndarray-weight-admission-20261016','ndarray-weight-admission-cpp-20261016'):
    assert json.loads((root/'logs'/trial/'exit.json').read_text())['exit_code']==0
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')", "mode in ('component',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'", "runner='run_ndarray_selective_prefill.py'")
exec(compile(source,__file__,'exec'))
