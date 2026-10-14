"""Derive a selected-control VL-B2 trial from the frozen executed comparator."""
import hashlib, json
from pathlib import Path

here = Path(__file__).resolve().parent
base = here.parents[1] / 'results/20261014/global-half-alpha/native/evidence/logs/global-half-alpha-native-20261014/jobs/Qwen3.5-0.executed.py'
data = base.read_bytes()
assert hashlib.sha256(data).hexdigest() == '83069557526a369c05f83b68ea4f45dd0224a0f772be87ee1f4803f890ab6d74'
source = data.decode()
assert source.count('build-attention-prefix') == 1
assert source.count('f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088') == 1
source = source.replace('build-attention-prefix', 'build-resident-kv-inplace')
source = source.replace('f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088', '73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2')
source = source.replace("assert value['attention_prefix_enabled']", "assert value['attention_prefix_enabled']==(policy=='candidate')")
source = source.replace("assert selected_model in (None,'Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') and selected_batch in (None,1,2)", "assert selected_model=='Qwen3.5-0.8B' and selected_batch==2")
source = source.replace('"""Native ABBA direct-zero candidate versus otherwise identical attention build."""', '"""Global-alpha and attention candidate versus selected resident-KV VL-B2 control."""')
target = here / 'compare_global_alpha_selected_vl2.py'
target.write_bytes(source.encode())
runner = (here / 'run_constant_head_profiles.py').read_text()
runner = runner.replace('compare_constant_head.py', target.name)
runner = runner.replace("for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):", "for model in ('Qwen3.5-0.8B',):")
runner = runner.replace('for batch in (1,):', 'for batch in (2,):').replace('assert len(profiles)==3', 'assert len(profiles)==1')
(here / 'run_global_alpha_selected_vl2.py').write_bytes(runner.encode())
launch = (here / 'launch_confirm_global_half_alpha_profiles.py').read_text().replace('run_global_half_alpha_profiles.py', 'run_global_alpha_selected_vl2.py')
(here / 'launch_global_alpha_selected_vl2.py').write_bytes(launch.encode())
print(json.dumps(dict(base_sha256=hashlib.sha256(data).hexdigest(), comparator_sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
