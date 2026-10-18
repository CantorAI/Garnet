"""Only run after observed full-proof controller31658 terminal exit0."""
from pathlib import Path
import subprocess,sys,json,hashlib,time,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-thread-qos';opt=root/'work/worldsense-opt';r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';s=r/'out/research/worldsense-thread-qos/source';job='gui/501/com.cantorai.cantor';session=int(sys.argv[1]);assert session==31658
assert json.loads((w/'full-proof-complete.json').read_text())['all_original_native_model_vision_session_and_QoS_typed_scope_parent_death_guards_pass']
identity=json.loads((w/'full-proof-identities-before.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in identity['drivers'].items());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in identity['source'].items());assert all(hashlib.sha256((r/('out/build/garnet-metal-worldsense-thread-qos-'+role)/'libgarnet.dylib').read_bytes()).hexdigest()==v for role,v in identity['libraries'].items())
(w/'full-proof-observed-terminal.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),session_id=session,observed_terminal_exit=0,all_consumed_identities_unchanged=True),indent=2)+'\n')
subprocess.run([sys.executable,str(w/'verify_correctness_terminal.py'),str(session)],check=True)
subprocess.run([sys.executable,str(opt/'ws59_inherited_probe.py')],check=True)
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Cantor did not unload')
try:
 with(w/'post-owner-death-recovery.log').open('w')as log:subprocess.run([str(n/'xlang3'),'-B',str(w/'post_owner_recovery.py')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
 print('Native QoS post-active-ownerdeath exact28layer recovery PASS',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
subprocess.run([sys.executable,str(opt/'ws59_admission.py')],check=True)
(w/'postproof-complete.json').write_text(json.dumps(dict(full_numerical_lifecycle_inherited_mode_postfault_scope_source_config_admission_pass=True,no_performance_result=True),indent=2)+'\n');print('All WS59 full proof/postfault/inherited/config prerequisites PASS; benchmark refreeze pending',flush=True)
