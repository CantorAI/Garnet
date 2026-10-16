"""Build/test frozen decode pairing after repeated component selection; no serving mutation."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();prep=json.loads((root/'preparation.json').read_text());assert str(root)==prep['target'] and root.name=='paired-norm-rope-native-investigation-20261016'
assert prep['selected_models'] and not (root/'build-paired-norm-rope-decode').exists()
assert all(v['kind']=='decode' for v in prep['selection']['confirmed_workloads'])
for name,sha in prep['source_sha256'].items():
 with (root/'source'/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==sha,name
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text().replace("mode in ('regular','extra')","mode in ('checks',)").replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_paired_norm_rope_native_checks.py'").replace("str(root/'source/test2026/metal'/runner)","str(Path(__file__).with_name(runner))")
exec(compile(source,__file__,'exec'))
