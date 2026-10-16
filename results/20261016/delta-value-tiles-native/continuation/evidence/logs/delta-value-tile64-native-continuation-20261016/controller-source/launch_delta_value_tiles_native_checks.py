"""Build and validate the exact private width64 source after repeated component selection."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();prep=json.loads((root/'preparation.json').read_text());assert str(root)==prep['target'] and root.name=='delta-value-tile64-native-investigation-20261016'
assert not (root/'build-delta-value-tile64-prefill').exists() and prep['selection']['confirmed_workloads']==[dict(width=64,kind='prefill')]
for name,sha in prep['source_sha256'].items():
 with (root/'source'/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha,name
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text().replace("mode in ('regular','extra')","mode in ('checks',)").replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_delta_value_tiles_native_checks.py'").replace("str(root/'source/test2026/metal'/runner)","str(Path(__file__).with_name(runner))")
exec(compile(source,__file__,'exec'))
