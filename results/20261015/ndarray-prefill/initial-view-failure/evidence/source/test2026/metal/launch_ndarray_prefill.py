"""Trained context experiment only after exact archived coefficient API gates."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
assert json.loads((root/'logs/ndarray-alpha-compat-20261015/exit.json').read_text())['exit_code']==0
with (root/'ndarray-alpha-compat-20261015.tar.gz').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='4bc6ecef917beb0c9c77b2c68c7722023c77bc4b1bea14168a166d594dd1f5b5'
audit=json.loads((root/'ndarray-alpha-compat-verification.json').read_text())
assert audit['all_compatible'] and audit['numerical_records']==64
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('component',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_ndarray_prefill.py'")
exec(compile(source,__file__,'exec'))
