"""Exclusive compact-aware grouped experiment only after native vector chain archival."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();pipeline=root/'logs/vector-vision-native-continuation-20261016'
assert json.loads((pipeline/'exit.json').read_text())['exit_code']==0
completed=json.loads((pipeline/'result.json').read_text())['completed_cohorts'];assert completed in (['initial'],['initial','repeat'])
candidate=root/'vector-vision-native-investigation-20261016'
for label in completed:
    record=json.loads((pipeline/(label+'-archive.json')).read_text())
    trial='vector-vision-native-'+('confirm-' if label=='repeat' else '')+'20261016'
    with (root/(trial+'.tar.gz')).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256']
    audit=json.loads((candidate/'validation-audits/vector-vision-native-results'/label/'verification.json').read_text())
    assert audit['full_output_requests']==72 and audit['actual_vector_vision_routes'] and audit['archive_sha256']==record['sha256']
with (root/'build-coalesced-bf16-body/libgarnet.dylib').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('component',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_compact_fused_projection_groups.py'")
exec(compile(source,__file__,'exec'))
