"""Native correctness only; run AFTER all four build sessions terminal success."""
from pathlib import Path
import subprocess,json,hashlib,time,sys,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-dense-prefetch/source';n=r/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor'
manifest=json.loads((w/'correctness-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
assert json.loads((w/'source-proof.json').read_text())['original_shader_literals_body_and_tests_unchanged']
for mode in ['off','primary','on','fault']:
 b=r/('out/build/garnet-metal-worldsense-dense-prefetch-'+mode);assert(b/'libgarnet.dylib').exists()
(w/'native-driver-identities-before-dispatch.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(w.rglob('*'))if p.is_file()and p.suffix in ['.py','.cpp','.json','.cmake']and p.name!='native-driver-identities-before-dispatch.json'}},indent=2)+'\n')
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False)
subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Cantor service did not unload')
try:
 results={}
 for mode in ['off','primary','on','fault']:
  b=r/('out/build/garnet-metal-worldsense-dense-prefetch-'+mode)
  with(w/('native-'+mode+'.log')).open('w')as f:p=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(b),'--output-on-failure'],stdout=f,stderr=subprocess.STDOUT)
  results[mode]=p.returncode;(w/'native-results.json').write_text(json.dumps(results,indent=2)+'\n')
  print('Native8tests',mode,'exit',p.returncode,flush=True)
  if p.returncode:raise RuntimeError('Native correctness failure '+mode)
 paths={mode:r/('out/build/garnet-metal-worldsense-dense-prefetch-'+mode)/'primary_projection_words.bin'for mode in ['off','primary','on','fault']}
 data=paths['off'].read_bytes();assert all(p.read_bytes()==data for p in paths.values())
 (w/'native-word-comparison.json').write_text(json.dumps({'records':186,'all_four_builds_FP32_words_identical':True,'all_FP32_words_identical':True,'bytes':{m:p.stat().st_size for m,p in paths.items()},'scope':'Original full native projection outputs, including prefill. Correctness only, no serving speed claim.'},indent=2)+'\n')
 assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
 programs={}
 for mode in ['off','primary','on','fault']:
  b=r/('out/build/garnet-metal-worldsense-dense-prefetch-'+mode)
  programs[mode]={'main':(b/'main_shader_program.metal').read_bytes(),'prefill':(b/'prefill_shader_program.metal').read_bytes()}
  for kind,value in programs[mode].items():(w/('actual-'+mode+'-'+kind+'-program.metal')).write_bytes(value)
 fragment=(s/'src/backends/metal/dense_prefetch_shader.h').read_text().split('R"DENSELOOK(',1)[1].split(')DENSELOOK"',1)[0].encode()
 for mode in ['off','primary','on','fault']:
  prior=w.parent/'worldsense-isolated-primary'/('actual-'+mode+'-main-program.metal')
  current=programs[mode]['main']
  if mode=='off':assert current==prior.read_bytes()and fragment not in current
  else:assert current.count(fragment)==1 and current.replace(fragment,b'',1)==prior.read_bytes()
  assert programs[mode]['prefill']==(w.parent/'worldsense-isolated-primary'/('actual-'+mode+'-prefill-program.metal')).read_bytes()
 rawpaths={mode:r/('out/build/garnet-metal-worldsense-dense-prefetch-'+mode)/'dense_prefetch_projection_words.bin'for mode in ['off','primary','on','fault']}
 baseline=rawpaths['off'].read_bytes();assert baseline and all(p.read_bytes()==baseline for p in rawpaths.values())
 (w/'dense-prefetch-word-comparison.json').write_text(json.dumps(dict(records=186,all_four_forced_raw_full_FP32_projection_files_identical=True,bytes=len(baseline),sha256=hashlib.sha256(baseline).hexdigest(),scope='Original93shape/twoinput numeric bounds, forcedraw actual new route and fallback/tail/poison/queuedsource/Abort recovery. Correctness-only, no performance evidence.'),indent=2)+'\n')
 (w/'actual-program-proof.json').write_text(json.dumps(dict(OFF_main_exact54=True,other_main_exact54_after_removing_one_exact_new_fragment=True,all_four_prefill_programs_exact54=True,programs={mode:{kind:dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest())for kind,value in parts.items()}for mode,parts in programs.items()},scope='Actual constructor-produced MSL source. Not GPU ISA/register allocation identity or throughput.'),indent=2)+'\n')
 print('ALL4BUILDS8NATIVE_TESTS/186ORIGINAL_AND186FORCEDRAW_PROJECTION_RECORDS_EXACT',flush=True)

finally:
 subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
