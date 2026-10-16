"""Fresh isolated build and all validation stages; retained serving selection unchanged."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();prep=json.loads((root/'preparation.json').read_text())
assert str(root)==prep['target'] and root.name=='ndarray-selective-native-investigation-20261016'
assert not (root/'build-ndarray-selective-prefill').exists()
for name,sha in prep['source_sha256'].items():
    with (root/'source'/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha,name
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('checks',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_ndarray_selective_native_checks.py'")
source=source.replace("str(root/'source/test2026/metal'/runner)","str(Path(__file__).with_name(runner))")
exec(compile(source,__file__,'exec'))
