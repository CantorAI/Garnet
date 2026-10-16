"""Quiet exclusive primitive, after both archived compact-eight trials."""
from pathlib import Path
import hashlib,json,sys

root=Path(sys.argv[1]).resolve()
for trial,sha in (
    ('coalesced-eight-body-20261015','d2be45e8cf1bf01199c476e7f97393c067e1c562f2d1bcce3d6926a260557346'),
    ('coalesced-eight-body-unrolled-fixed-20261015','bf10548d6e55ef01d8299e956fa1d9039519d98a66a026d07fe09bb8057c572f')):
    assert json.loads((root/'logs'/trial/'exit.json').read_text())['exit_code']==0
    with (root/(trial+'.tar.gz')).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('primitive',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_mlp_command_buffer_primitive.py'")
exec(compile(source,__file__,'exec'))
