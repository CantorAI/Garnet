"""All-six native vision-alpha comparison with exact inputs and actual use counts."""
import hashlib
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
base=root/'results/20261014/global-half-alpha/native/evidence/logs/global-half-alpha-native-20261014/jobs/Qwen3.5-0.executed.py'
raw=base.read_bytes();assert hashlib.sha256(raw).hexdigest()=='83069557526a369c05f83b68ea4f45dd0224a0f772be87ee1f4803f890ab6d74'
source=raw.decode().replace("('control','build-attention-prefix')","('control','build-global-half-alpha')").replace("('candidate','build-global-half-alpha')","('candidate','build-vision-global-alpha')")
source=source.replace('f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088','887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f')
source=source.replace('logs/global-half-alpha-checks-20261014','logs/vision-global-alpha-checks-20261014')
source=source.replace("assert value['global_half_alpha_enabled']==(policy=='candidate')","assert value['global_half_alpha_enabled']\n                assert value['vision_global_alpha_enabled']==(policy=='candidate')")
needle="assert (value['prefill_profile'].get('global_half_alpha_dispatches',0)>0) if policy=='candidate' else (value['prefill_profile'].get('global_half_alpha_dispatches',0)==0)"
assert source.count(needle)==1
source=source.replace(needle,"assert value['prefill_profile'].get('global_half_alpha_dispatches',0)>0\n                expected_vision=4*config['vision_config']['depth'] if vl and policy=='candidate' else 0\n                assert value['vision_global_alpha_dispatches']==expected_vision\n                assert not value['single_coder_head_enabled'] and not value['joined_global_mlp_enabled'] and not value['packed_prefill_norm_enabled'] and not value['residual_norm_enabled'] and not value['constant_head_enabled'] and not value['direct_state_zero_enabled']")
# The text-config alias above is useful for decoder layers; retain a separate
# complete config lookup for the vision depth used in the actual counter gate.
source=source.replace("4*config['vision_config']['depth']", "4*json.loads((folder/'config.json').read_text())['vision_config']['depth']")
(here/'compare_vision_global_alpha.py').write_bytes(source.encode())
runner=(here/'run_constant_head_profiles.py').read_text().replace('compare_constant_head.py','compare_vision_global_alpha.py').replace('for batch in (1,):','for batch in (1,2):').replace('assert len(profiles)==3','assert len(profiles)==6')
(here/'run_vision_global_alpha_profiles.py').write_bytes(runner.encode())
queue=(here/'queue_packed_prefill_norm_profiles.py').read_text().replace('packed-prefill-norm-checks-20261014','vision-global-alpha-checks-20261014').replace('68972','13040').replace('launch_quiet_packed_prefill_norm_checks.py','launch_vision_global_alpha_checks.py').replace('run_packed_prefill_norm_profiles.py','run_vision_global_alpha_profiles.py')
(here/'queue_vision_global_alpha_profiles.py').write_bytes(queue.encode())
print('Prepared six complete-output profiles with exact per-image projection counts')
