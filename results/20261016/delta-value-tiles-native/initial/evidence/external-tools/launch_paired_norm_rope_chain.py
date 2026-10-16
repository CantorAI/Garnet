"""Require archived single-pair result, then exclusive complete carried-chain calibration."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();prior=root/'validation-audits/paired-norm-rope-learned-20261016';audit=json.loads((prior/'verification.json').read_text());assert audit['complete_learned_profiles']==232 and audit['summary']['numeric']['original_output_bits_checked']==47321088
with (root/'paired-norm-rope-learned-20261016.tar.gz').open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()=='31d657a088e16ec3217c4db3ebb72f670b0e4e6fc301ff9fa3c8834d91a1acc2'
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text().replace("mode in ('regular','extra')","mode in ('chain',)").replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_paired_norm_rope_chain.py'")
exec(compile(source,__file__,'exec'))
