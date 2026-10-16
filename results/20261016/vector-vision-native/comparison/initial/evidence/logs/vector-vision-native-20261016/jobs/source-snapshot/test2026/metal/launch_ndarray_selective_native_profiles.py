"""Dispatch original native profiles only after terminal, archived, full30 independent audit."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();parent=Path(json.loads((root/'preparation.json').read_text())['parent'])
assert root.name=='ndarray-selective-native-packed-investigation-20261016'
controller=parent/'logs/ndarray-selective-native-finalize-20261016'
assert json.loads((controller/'exit.json').read_text())['exit_code']==0
assert json.loads((controller/'result.json').read_text())['independent_audit_passed']
trial='ndarray-selective-native-packed-checks-20261016'
assert json.loads((root/'logs'/trial/'exit.json').read_text())['exit_code']==0
archive=parent/(trial+'.tar.gz')
with archive.open('rb') as stream:identity=hashlib.file_digest(stream,'sha256').hexdigest()
assert identity==json.loads((controller/'archive.json').read_text())['sha256']
# Re-audit exactly the frozen full30 dataset with explicit archive identity.
# Preserve the controller's original audit; no GPU stage is repeated.
saved=sys.argv[:]
try:
    sys.argv=[saved[0],str(root/'validation-audits'/trial),str(root/'ndarray-selective-native-checks-archive-verification.json'),str(archive)]
    helper=Path(__file__).with_name('verify_ndarray_selective_native_checks.py')
    exec(compile(helper.read_text(),str(helper),'exec'),{'__file__':str(helper)})
finally:sys.argv=saved
audit=json.loads((root/'ndarray-selective-native-checks-archive-verification.json').read_text())
prior=json.loads((root/'ndarray-selective-native-checks-verification.json').read_text())
assert {k:v for k,v in audit.items() if k!='archive_sha256'}==prior
assert audit['archive_sha256']==identity and audit['validation_stages']==30 and audit['ctests']==7
assert audit['coalesced_primitive_cases']==80 and audit['ndarray_cache_limit']==256
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('profiles',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_ndarray_selective_native_profiles.py'")
source=source.replace("str(root/'source/test2026/metal'/runner)","str(Path(__file__).with_name(runner))")
exec(compile(source,__file__,'exec'))
