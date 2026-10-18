"""Exact model recovery and configuration admission after owned death proof."""
from pathlib import Path
import subprocess, sys, time, json, hashlib, datetime
opt=Path(__file__).resolve().parent;w=opt.parent/'worldsense-validated-text-worker'
r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor'
assert json.loads((w/'owner-death-result.json').read_text())['all_three_parent_death_cases_pass']
assert json.loads((w/'learned-independent-audit.json').read_text())['original_learned_native_state_token_vision_verifier_pass']
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False)
subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Normal service failed to unload')
try:
 with(w/'post-owner-death-recovery.log').open('w')as log:
  subprocess.run([str(n/'xlang3'),'-B',str(opt/'ws61_post_owner_recovery.py')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
 print('Post-owner-death exact model recovery PASS',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
subprocess.run([sys.executable,str(opt/'ws61_admission.py')],check=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
d=json.loads((w/'learned-input-identities-before.json').read_text())
for key,source in [('parent_source',r/'out/research/worldsense-validated-text-worker/source'),('legacy_source',r/'out/research/worldsense-prefix-cache/source')]:
 assert all(sha(source/k)==v for k,v in d[key].items())
assert all(sha(Path(k))==v for k,v in d['external'].items())
(w/'full-proof-complete.json').write_text(json.dumps(dict(
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 original_full_native_word_state_token_vision_prefix_capacity_ownership_proofs_pass=True,
 owned_parent_death_bootstrap_loaded_idle_and_active_native_model_pass=True,
 actual_active_native_stack_sample_before_kill_pass=True,
 post_owner_death_28layer_state_token_recovery_pass=True,
 thirteen_expected_config_rejections_pass=True,
 parent_legacy_model_runtime_assets_and_frozen_drivers_unchanged=True,
 no_performance_result=True),indent=2)+'\n')
print('WS61 full correctness/lifecycle/config admission PASS; no performance result',flush=True)
