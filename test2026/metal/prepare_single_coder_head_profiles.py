"""Prepare all-six native profiles against the frozen global-alpha bundle."""
import hashlib
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
base=root/'results/20261014/global-half-alpha/native/evidence/logs/global-half-alpha-native-20261014/jobs/Qwen3.5-0.executed.py'
raw=base.read_bytes();assert hashlib.sha256(raw).hexdigest()=='83069557526a369c05f83b68ea4f45dd0224a0f772be87ee1f4803f890ab6d74'
source=raw.decode().replace("('control','build-attention-prefix')","('control','build-global-half-alpha')").replace("('candidate','build-global-half-alpha')","('candidate','build-single-coder-head-repaired')")
source=source.replace('f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088','887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f')
source=source.replace('logs/global-half-alpha-checks-20261014','logs/single-coder-head-repair-checks-20261014')
source=source.replace("assert value['global_half_alpha_enabled']==(policy=='candidate')","assert value['global_half_alpha_enabled']\n                assert value['single_coder_head_enabled']==(policy=='candidate')")
needle="assert (value['prefill_profile'].get('global_half_alpha_dispatches',0)>0) if policy=='candidate' else (value['prefill_profile'].get('global_half_alpha_dispatches',0)==0)"
assert source.count(needle)==1
source=source.replace(needle,"assert value['prefill_profile'].get('global_half_alpha_dispatches',0)>0\n                counts=value['single_coder_head_counts'];assert counts\n                if not vl:assert len(counts)==value['decode_steps_after_first']+1\n                expected_count=1 if policy=='candidate' and (model,batch)==('Qwen2.5-Coder-0.5B-Instruct',1) else 0\n                assert all(c==expected_count for c in counts)\n                assert not value['joined_global_mlp_enabled'] and not value['packed_prefill_norm_enabled'] and not value['residual_norm_enabled'] and not value['constant_head_enabled'] and not value['direct_state_zero_enabled']")
(here/'compare_single_coder_head.py').write_bytes(source.encode())
runner=(here/'run_constant_head_profiles.py').read_text().replace('compare_constant_head.py','compare_single_coder_head.py').replace('for batch in (1,):','for batch in (1,2):').replace('assert len(profiles)==3','assert len(profiles)==6')
(here/'run_single_coder_head_profiles.py').write_bytes(runner.encode())
queue=(here/'queue_packed_prefill_norm_profiles.py').read_text().replace('packed-prefill-norm-checks-20261014','single-coder-head-repair-checks-20261014').replace('68972','80953').replace('launch_quiet_packed_prefill_norm_checks.py','launch_single_coder_head_repair_checks.py').replace('run_packed_prefill_norm_profiles.py','run_single_coder_head_profiles.py')
(here/'queue_single_coder_head_profiles.py').write_bytes(queue.encode())
print('Prepared six profiles with actual head-count checks on every dense prediction')
