"""Require terminal archived primitive and unchanged paired context before learned execution."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();primitive=root/'validation-audits/paired-norm-rope-primitive-20261016';assert json.loads((primitive/'verification.json').read_text())['original_fp32_output_bits_checked']==3710720
with (root/'paired-norm-rope-primitive-20261016.tar.gz').open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()=='61cd813008cb2431819dc34fa2f719ac7acefce9ba2178ac90a1375b370fe4eb'
source=Path(__file__).with_name('launch_quiet_prefill_residual_norm_probe.py').read_text().replace("mode in ('regular','extra')","mode in ('learned',)").replace("runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'","runner='run_paired_norm_rope_learned.py'")
exec(compile(source,__file__,'exec'))
