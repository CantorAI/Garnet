"""Exclusive vector-vision probe; run only after prior native jobs are terminal and archived."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
pipeline=root/'logs/ndarray-selective-native-continue-20261016'
assert json.loads((pipeline/'exit.json').read_text())['exit_code']==0
completed=json.loads((pipeline/'result.json').read_text())['completed_cohorts'];assert completed in (['initial'],['initial','repeat'])
for label in completed:
    record=json.loads((pipeline/(label+'-archive.json')).read_text())
    trial='ndarray-selective-native-'+('confirm-' if label=='repeat' else '')+'20261016'
    with (root/(trial+'.tar.gz')).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256']
    candidate=root/'ndarray-selective-native-packed-investigation-20261016'
    audit=json.loads((candidate/'validation-audits/ndarray-selective-native-results'/label/'verification.json').read_text())
    assert audit['full_output_requests']==72 and audit['archive_sha256']==record['sha256']
with (root/'build-coalesced-bf16-body/libgarnet.dylib').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('component',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_vector_vision_context.py'")
exec(compile(source,__file__,'exec'))
