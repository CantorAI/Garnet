"""Relaunch fresh build/gates after frozen self-guard failure; serving source stays frozen."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();parent=Path(json.loads((root/'preparation.json').read_text())['parent'])
assert json.loads((root/'logs/ndarray-native-checks-20261015/exit.json').read_text())['exit_code']==1
with (parent/'ndarray-native-checks-self-guard-20261015.tar.gz').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='2f827d302fa75759c9a85c0cafdf5ff1c65a21235b913be5016039113bb8ce1e'
prep=json.loads((root/'preparation.json').read_text());assert not (root/'build-ndarray-prefill-shape80-guardrepair').exists()
for name,sha in prep['source_sha256'].items():
    with (root/'source'/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha,name
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('checks',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_ndarray_native_checks_repaired.py'")
source=source.replace("str(root/'source/test2026/metal'/runner)","str(Path(__file__).with_name(runner))")
exec(compile(source,__file__,'exec'))
