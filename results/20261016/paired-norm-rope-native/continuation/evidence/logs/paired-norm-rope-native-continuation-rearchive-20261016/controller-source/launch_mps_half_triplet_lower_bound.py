"""Exclusive synthetic prefill capability only after compact component chain terminal/archive."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();upstream=root/'logs/compact-fused-projection-groups-queue-20261016'
assert json.loads((upstream/'exit.json').read_text())['exit_code']==0
labels=json.loads((upstream/'result.json').read_text())['completed_cohorts'];assert labels in (['initial'],['initial','repeat'])
for label in labels:
    record=json.loads((upstream/(label+'-archive.json')).read_text());trial='compact-fused-projection-groups-'+('confirm-' if label=='repeat' else '')+'20261016'
    with (root/(trial+'.tar.gz')).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256']
    assert json.loads((root/'validation-audits/compact-fused-projection-groups'/label/'verification.json').read_text())['group_batch_checks']==16
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('capability',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_mps_half_triplet_lower_bound.py'")
exec(compile(source,__file__,'exec'))
