"""Require full33 terminal archive and complete raw re-audit, then exclusive native profiles."""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(sys.argv[1]).resolve();parent=Path(json.loads((root/'preparation.json').read_text())['parent']);trial='prepared-qkv-native-checks-countfix-20261016'
assert json.loads((root/'logs'/trial/'exit.json').read_text())['exit_code']==0
archive=parent/(trial+'.tar.gz');extracted=parent/'validation-audits'/trial
subprocess.run([sys.executable,str(Path(__file__).with_name('verify_prepared_qkv_native_checks_countfix.py')),str(extracted),str(root/'prepared-qkv-native-checks-archive-verification.json'),str(archive)],check=True)
audit=json.loads((root/'prepared-qkv-native-checks-archive-verification.json').read_text())
with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==audit['archive_sha256']
assert audit['validation_stages']==33 and audit['prepared_qkv_trained_steps']==24
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text();source=source.replace("mode in ('regular','extra')","mode in ('profiles',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_prepared_qkv_native_profiles.py'")
source=source.replace("str(root/'source/test2026/metal'/runner)","str(Path(__file__).with_name(runner))")
exec(compile(source,__file__,'exec'))
