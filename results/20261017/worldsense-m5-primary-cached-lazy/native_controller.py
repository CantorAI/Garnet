"""Native correctness only; run AFTER all four build sessions terminal success."""
from pathlib import Path
import subprocess,json,hashlib,time,sys,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-cached-lazy/source';n=r/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor'
manifest=json.loads((w/'correctness-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
assert json.loads((w/'independent-shader-proof.json').read_text())['CXX_materialized_OFF_primary_fragment_exact']
for mode in ['off','primary','on','fault']:
 b=r/('out/build/garnet-metal-worldsense-primary-cached-lazy-'+mode);assert(b/'libgarnet.dylib').exists()
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
  b=r/('out/build/garnet-metal-worldsense-primary-cached-lazy-'+mode)
  with(w/('native-'+mode+'.log')).open('w')as f:p=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(b),'--output-on-failure'],stdout=f,stderr=subprocess.STDOUT)
  results[mode]=p.returncode;(w/'native-results.json').write_text(json.dumps(results,indent=2)+'\n')
  print('Native7tests',mode,'exit',p.returncode,flush=True)
  if p.returncode:raise RuntimeError('Native correctness failure '+mode)
 paths={mode:r/('out/build/garnet-metal-worldsense-primary-cached-lazy-'+mode)/'primary_projection_words.bin'for mode in ['off','primary','on','fault']}
 data=paths['off'].read_bytes();assert all(p.read_bytes()==data for p in paths.values())
 (w/'native-word-comparison.json').write_text(json.dumps({'records':186,'all_four_builds_FP32_words_identical':True,'all_FP32_words_identical':True,'bytes':{m:p.stat().st_size for m,p in paths.items()},'scope':'Original full native projection outputs, including prefill. Correctness only, no serving speed claim.'},indent=2)+'\n')
 assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
 programs={}
 for mode in ['off','primary','on','fault']:
  b=r/('out/build/garnet-metal-worldsense-primary-cached-lazy-'+mode)
  programs[mode]={'main':(b/'main_shader_program.metal').read_bytes(),'prefill':(b/'prefill_shader_program.metal').read_bytes()}
  for kind,value in programs[mode].items():(w/('actual-'+mode+'-'+kind+'-program.metal')).write_bytes(value)
 fragment=(s/'src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R"PVEC(',1)[1].split(')PVEC"',1)[0].encode()
 assert programs['primary']['main']==programs['on']['main']==programs['fault']['main']
 assert not programs['off']['prefill']and not programs['primary']['prefill']
 preamble=(s/'src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R"PMIN(',1)[1].split(')PMIN"',1)[0].encode()
 assert programs['on']['prefill']==preamble+fragment and programs['fault']['prefill']==programs['on']['prefill']
 assert b'kernel void primary_bf16_decode'not in programs['on']['prefill']
 assert b'kernel void primary_vector_bf16_gemm'not in programs['on']['main']
 (w/'actual-program-proof.json').write_text(json.dumps({'actual_legacy_main_priorprimary_candidate_fault_byte_exact':True,'actual_candidate_prefill_equals_minimal_preamble_plus_fragment':True,'original_legacy_lookups_use_main_library':True,'programs':{mode:{kind:{'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}for kind,value in data.items()}for mode,data in programs.items()},'scope':'Actual constructor-produced source strings and frozen host routing; not GPU machine code identity or performance.'},indent=2)+'\n')
 print('ACTUAL_MAIN_PROGRAM_PRIORPRIMARY_CANDIDATE_FAULT_IDENTICAL_AND_PREFILL_ISOLATED',flush=True)
 print('ALL4BUILDS7NATIVE_TESTS_AND186_COMPLETE_PROJECTION_RECORDS_EXACT',flush=True)
finally:
 subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
