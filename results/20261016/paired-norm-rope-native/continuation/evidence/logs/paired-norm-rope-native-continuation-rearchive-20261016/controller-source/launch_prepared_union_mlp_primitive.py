"""Compile/test only after archived original-codec union admission proof and exclusive guard."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();audit=json.loads((root/'validation-audits/joined-union-half-admission-20261016/verification.json').read_text())
assert audit['pairs']==76 and audit['union_admitted_pairs']==75 and audit['rejected_weight_values']==1
with (root/'joined-union-half-admission-20261016.tar.gz').open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()=='f753aebd6121680da5ca162bb1caed5661f304068ba6f23608e6cc5c5755e466'
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('primitive',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_prepared_union_mlp_primitive.py'")
exec(compile(source,__file__,'exec'))
