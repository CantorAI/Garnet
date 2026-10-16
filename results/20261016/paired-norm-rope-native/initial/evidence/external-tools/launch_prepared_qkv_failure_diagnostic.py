"""Sequential exact-input control diagnostic only after candidate terminal archive."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();parent=Path(json.loads((root/'preparation.json').read_text())['parent'])
assert json.loads((root/'logs/prepared-qkv-native-checks-countfix-20261016/exit.json').read_text())['exit_code']==1
with (parent/'prepared-qkv-native-checks-countfix-20261016.tar.gz').open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()=='544e11d2c2f18582ceffb6a12404fb3113d4f03d5b382b55944076894e849c1a'
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text().replace("mode in ('regular','extra')","mode in ('diagnostic',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='diagnose_prepared_qkv_trained_failure.py'").replace("str(root/'source/test2026/metal'/runner)","str(Path(__file__).with_name(runner))")
exec(compile(source,__file__,'exec'))
