"""Freeze all-six native comparison against the pinned global-alpha control."""
import hashlib
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
base=root/'results/20261014/global-half-alpha/native/evidence/logs/global-half-alpha-native-20261014/jobs/Qwen3.5-0.executed.py'
raw=base.read_bytes();assert hashlib.sha256(raw).hexdigest()=='83069557526a369c05f83b68ea4f45dd0224a0f772be87ee1f4803f890ab6d74'
source=raw.decode()
source=source.replace("('control','build-attention-prefix')","('control','build-global-half-alpha')").replace("('candidate','build-global-half-alpha')","('candidate','build-joined-global-mlp')")
source=source.replace('f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088','887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f')
source=source.replace('logs/global-half-alpha-checks-20261014','logs/joined-global-mlp-checks-20261014')
source=source.replace("assert value['global_half_alpha_enabled']==(policy=='candidate')","assert value['global_half_alpha_enabled']\n                assert value['joined_global_mlp_enabled']==(policy=='candidate')")
needle="assert (value['prefill_profile'].get('global_half_alpha_dispatches',0)>0) if policy=='candidate' else (value['prefill_profile'].get('global_half_alpha_dispatches',0)==0)"
assert source.count(needle)==1
source=source.replace(needle,"assert value['prefill_profile'].get('global_half_alpha_dispatches',0)>0\n                admitted=(model,batch) in [('Qwen3-0.6B',1),('Qwen3-0.6B',2),('Qwen2.5-Coder-0.5B-Instruct',1)]\n                optional_vl=(model,batch)==('Qwen3.5-0.8B',2)\n                blocks=value['prefill_profile'].get('joined_global_mlp_blocks',0)\n                assert (blocks>0) if policy=='candidate' and admitted else (blocks>=0 if policy=='candidate' and optional_vl else blocks==0)\n                assert not value['packed_prefill_norm_enabled'] and not value['residual_norm_enabled'] and not value['constant_head_enabled'] and not value['direct_state_zero_enabled']")
(here/'compare_joined_global_mlp.py').write_bytes(source.encode())
runner=(here/'run_constant_head_profiles.py').read_text().replace('compare_constant_head.py','compare_joined_global_mlp.py').replace('for batch in (1,):','for batch in (1,2):').replace('assert len(profiles)==3','assert len(profiles)==6')
(here/'run_joined_global_mlp_profiles.py').write_bytes(runner.encode())
queue=(here/'queue_packed_prefill_norm_profiles.py').read_text().replace('packed-prefill-norm-checks-20261014','joined-global-mlp-checks-20261014').replace('68972','53015').replace('launch_quiet_packed_prefill_norm_checks.py','launch_joined_global_mlp_checks.py').replace('run_packed_prefill_norm_profiles.py','run_joined_global_mlp_profiles.py')
(here/'queue_joined_global_mlp_profiles.py').write_bytes(queue.encode())
print('Prepared full six-profile native comparator and exact-PID dependency gate')
