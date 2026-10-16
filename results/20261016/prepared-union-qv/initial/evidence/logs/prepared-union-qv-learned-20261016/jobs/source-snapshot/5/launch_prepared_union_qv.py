"""Quiet exclusive QV with original K prototype; require captured full CPU admission before any GPU stage."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();mode=sys.argv[3];assert mode in ('primitive','learned')
with (root/'qkv-union-half-admission-20261016.tar.gz').open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()=='3142474829315dd54bbc8bb84ffa3b7d56ec0299c16c5e581d473afd1ca9c825'
cpu=json.loads((root/'validation-audits/qkv-union-half-admission-20261016/verification.json').read_text());assert (cpu['groups'],cpu['admitted_groups'],cpu['retained_fp32_key_matrices'])==(58,56,52)
if mode=='learned':
 audit=json.loads((root/'validation-audits/prepared-union-qv-primitive-20261016/verification.json').read_text());assert audit['dyadic_cases']==24 and audit['full_goal_met'] is False
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text();source=source.replace("mode in ('regular','extra')","mode in ('primitive','learned')")
source=source.replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_prepared_union_qv_primitive.py' if mode=='primitive' else 'run_prepared_union_qv_learned.py'")
exec(compile(source,__file__,'exec'))
