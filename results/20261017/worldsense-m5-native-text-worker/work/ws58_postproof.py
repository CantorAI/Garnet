from pathlib import Path
import subprocess,sys,time,json,hashlib,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-native-text-worker';opt=root/'work/worldsense-opt';n=Path('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin');job='gui/501/com.cantorai.cantor'
assert json.loads((w/'reproof-complete.json').read_text())['full_native_text_VLM_corpus_vision_lifecycle_and_parent_telemetry_reproof_pass']
(w/'postproof-driver-identities.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__).resolve(),opt/'ws58_post_owner_recovery.py',opt/'ws58_admission.py',w/'verify_correctness_terminal.py']}),indent=2)+'\n')
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Cantor did not unload')
try:
 with(w/'post-owner-death-recovery.log').open('w')as log:subprocess.run([str(n/'xlang3'),'-B',str(opt/'ws58_post_owner_recovery.py')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
 print('PASS post-owner-death exact native state/token recovery',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
subprocess.run([sys.executable,str(w/'verify_correctness_terminal.py'),sys.argv[1]],check=True)
subprocess.run([sys.executable,str(opt/'ws58_admission.py')],check=True)
print('REVISED WS58 all numerical/lifecycle/source/config prerequisite checks PASS',flush=True)
