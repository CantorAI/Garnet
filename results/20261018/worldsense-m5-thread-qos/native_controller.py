"""Original four-role native correctness and exact WS54 shader/word confirmation."""
from pathlib import Path
import subprocess,json,hashlib,time,sys,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-thread-qos/source';n=r/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor';roles=['off','primary','on','fault']
manifest=json.loads((w/'gateway-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items());assert json.loads((w/'gateway-source-inverse-proof.json').read_text())['all_MSL_codecs_math_graph_session_xModel_original_tests_CUDA_unchanged']
(w/'native-driver-identities-before.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [Path(__file__).resolve(),w/'audit_source.py',w/'gateway-source-manifest.json',n/'xlang3',n/'libxlang3_runtime.dylib']}),indent=2)+'\n')
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Cantor service did not unload')
try:
 results={};words={};programs={}
 for mode in roles:
  b=r/('out/build/garnet-metal-worldsense-thread-qos-'+mode)
  with(w/('native-'+mode+'.log')).open('w')as log:p=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(b),'--output-on-failure'],stdout=log,stderr=subprocess.STDOUT,timeout=600)
  results[mode]=p.returncode;(w/'native-results.json').write_text(json.dumps(results,indent=2)+'\n');print('Original native tests '+mode+' exit '+str(p.returncode),flush=True);assert p.returncode==0
  words[mode]=(b/'primary_projection_words.bin').read_bytes();programs[mode]={}
  for kind in ['main','prefill']:
   data=(b/(kind+'_shader_program.metal')).read_bytes();prior=(w.parent/'worldsense-isolated-primary'/('actual-'+mode+'-'+kind+'-program.metal')).read_bytes();assert data==prior
   (w/('actual-'+mode+'-'+kind+'-program.metal')).write_bytes(data);programs[mode][kind]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),exact_WS54=True)
 ref=(r/'out/build/garnet-metal-worldsense-isolated-primary-off/primary_projection_words.bin').read_bytes();assert all(v==ref for v in words.values())
 (w/'native-word-comparison.json').write_text(json.dumps(dict(records=186,all_four_builds_FP32_words_identical=True,exact_WS54=True,bytes=len(ref),sha256=hashlib.sha256(ref).hexdigest(),scope='Original native projection output records, correctness only.'),indent=2)+'\n')
 (w/'actual-program-proof.json').write_text(json.dumps(dict(all_four_actual_main_and_prefill_programs_byte_exact_WS54=True,programs=programs,scope='Actual constructor-produced MSL source, not GPU machine-code identity or throughput.'),indent=2)+'\n')
 assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items());print('Four roles/native tests/186 exact FP32 projection records/actual shader sources PASS',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
