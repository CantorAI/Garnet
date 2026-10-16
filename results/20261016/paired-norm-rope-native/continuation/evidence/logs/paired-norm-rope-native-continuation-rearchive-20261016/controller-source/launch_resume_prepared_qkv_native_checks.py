"""Exclusive completion after terminal archived bookkeeping failure; unchanged native candidate."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();parent=Path(json.loads((root/'preparation.json').read_text())['parent'])
for name,expected in [('prepared-qkv-native-checks-20261016.tar.gz','f1c7ba015ebaf86c6982bd0284c03d2f6116fb2d6e12fd05a3cdcc31d5c735eb'),('prepared-qkv-native-count-failure-controllers-20261016.tar.gz','156661c9779d94a5835f31c810f7e4e1cba05c8ae03adca936546865805f3bf0')]:
 with (parent/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
assert json.loads((root/'logs/prepared-qkv-native-checks-20261016/exit.json').read_text())['exit_code']==1
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text();source=source.replace("mode in ('regular','extra')","mode in ('resume',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='resume_prepared_qkv_native_checks.py'")
source=source.replace("str(root/'source/test2026/metal'/runner)","str(Path(__file__).with_name(runner))")
exec(compile(source,__file__,'exec'))
