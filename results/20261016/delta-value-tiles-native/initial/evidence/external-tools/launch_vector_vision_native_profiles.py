"""Admit full-model timings only after terminal/archive/raw audit of fresh30."""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(sys.argv[1]).resolve();parent=Path(json.loads((root/'preparation.json').read_text())['parent'])
trial='vector-vision-native-checks-20261016'
assert json.loads((root/'logs'/trial/'exit.json').read_text())['exit_code']==0
archive=parent/(trial+'.tar.gz')
with archive.open('rb') as stream:identity=hashlib.file_digest(stream,'sha256').hexdigest()
subprocess.run([sys.executable,str(Path(__file__).with_name('verify_vector_vision_native_checks.py')),
 str(root/'validation-audits'/trial),str(root/'vector-vision-native-checks-archive-verification.json'),str(archive)],check=True)
audit=json.loads((root/'vector-vision-native-checks-archive-verification.json').read_text())
assert audit['archive_sha256']==identity and audit['validation_stages']==30 and audit['vector_primitive_checks']==96
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('profiles',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_vector_vision_native_profiles.py'")
source=source.replace("str(root/'source/test2026/metal'/runner)","str(Path(__file__).with_name(runner))")
exec(compile(source,__file__,'exec'))
