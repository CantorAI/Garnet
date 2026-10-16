"""Launch only after both archived preparation prerequisites and a quiet exclusive guard."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for trial,expected in [('joined-union-half-admission-20261016','f753aebd6121680da5ca162bb1caed5661f304068ba6f23608e6cc5c5755e466'),('prepared-union-mlp-primitive-20261016','98db64683fec2df9d3f553ac9211f9031dae0a0365e136f8ddd85dc3324a85c6')]:
 with (root/(trial+'.tar.gz')).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
 audit=json.loads((root/'validation-audits'/trial/'verification.json').read_text());assert audit['full_goal_met'] is False
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text()
source=source.replace("mode in ('regular','extra')","mode in ('learned',)")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_prepared_union_mlp_learned.py'")
exec(compile(source,__file__,'exec'))
