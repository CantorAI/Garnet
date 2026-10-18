"""Native correctness only; run AFTER all four build sessions terminal success."""
from pathlib import Path
import subprocess,json,hashlib,time,sys,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-sole-interleaved/source';n=r/'out/build/macos-arm64-release/bin';job='gui/501/com.cantorai.cantor'
manifest=json.loads((w/'correctness-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
assert json.loads((w/'source-inverse-proof.json').read_text())['original_shader_literals_graph_session_xModel_image_CUDA_tests_unchanged']
for mode in ['off','primary','on','fault']:
 b=r/('out/build/garnet-metal-worldsense-sole-interleaved-'+mode);assert(b/'libgarnet.dylib').exists()
(w/'native-driver-identities-before-dispatch.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(w.rglob('*'))if p.is_file()and p.suffix in ['.py','.cpp','.json','.cmake']and p.name!='native-driver-identities-before-dispatch.json'}},indent=2)+'\n')
native_binaries=[n/'xlang3',n/'libxlang3_runtime.dylib']
for mode in ['off','primary','on','fault']:
 build=r/('out/build/garnet-metal-worldsense-sole-interleaved-'+mode)
 native_binaries += [p for p in build.iterdir()if p.is_file()and(p.name.startswith('garnet_')or p.name in ['libgarnet.dylib','libgarnet_metal.a','CMakeCache.txt'])]
frozen_binaries={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in native_binaries}
(w/'native-binaries-before.json').write_text(json.dumps(frozen_binaries,indent=2)+'\n')
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False)
subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(200):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Cantor service did not unload')
try:
 results={}
 for mode in ['off','primary','on','fault']:
  b=r/('out/build/garnet-metal-worldsense-sole-interleaved-'+mode)
  with(w/('native-'+mode+'.log')).open('w')as f:p=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(b),'--output-on-failure'],stdout=f,stderr=subprocess.STDOUT)
  results[mode]=p.returncode;(w/'native-results.json').write_text(json.dumps(results,indent=2)+'\n')
  print('Native8tests',mode,'exit',p.returncode,flush=True)
  if p.returncode:raise RuntimeError('Native correctness failure '+mode)
 paths={mode:r/('out/build/garnet-metal-worldsense-sole-interleaved-'+mode)/'primary_projection_words.bin'for mode in ['off','primary','on','fault']}
 data=paths['off'].read_bytes();assert all(p.read_bytes()==data for p in paths.values())
 (w/'native-word-comparison.json').write_text(json.dumps({'records':186,'all_four_builds_FP32_words_identical':True,'all_FP32_words_identical':True,'bytes':{m:p.stat().st_size for m,p in paths.items()},'scope':'Original full native projection outputs, including prefill. Correctness only, no serving speed claim.'},indent=2)+'\n')
 sole_paths={mode:r/('out/build/garnet-metal-worldsense-sole-interleaved-'+mode)/'sole_projection_words.bin'for mode in ['off','primary','on','fault']}
 sole_words=sole_paths['off'].read_bytes();assert all(p.read_bytes()==sole_words for p in sole_paths.values())
 (w/'sole-word-comparison.json').write_text(json.dumps(dict(records=28,all_four_builds_FP32_words_identical=True,bytes=len(sole_words),scope='Two exact MLP shapes, seven row counts, two changed inputs; full outputs and original FP64 bounds. Separate from model/speed proof.'),indent=2)+'\n')
 actual_sole={mode:(r/('out/build/garnet-metal-worldsense-sole-interleaved-'+mode)/'sole_projection_words.bin.metal').read_bytes()for mode in ['off','primary','on','fault']}
 assert not actual_sole['off']and not actual_sole['primary']and actual_sole['on']==actual_sole['fault']
 declared=(s/'src/backends/metal/sole_interleaved_mlp_shader.h').read_text().split('R"SOLEMLP(',1)[1].split(')SOLEMLP"',1)[0].encode();assert actual_sole['on']==declared
 for mode,value in actual_sole.items():(w/('actual-'+mode+'-sole-program.metal')).write_bytes(value)
 assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
 programs={}
 for mode in ['off','primary','on','fault']:
  b=r/('out/build/garnet-metal-worldsense-sole-interleaved-'+mode)
  programs[mode]={'main':(b/'main_shader_program.metal').read_bytes(),'prefill':(b/'prefill_shader_program.metal').read_bytes()}
  for kind,value in programs[mode].items():(w/('actual-'+mode+'-'+kind+'-program.metal')).write_bytes(value)
 fragment=(s/'src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R"PVEC(',1)[1].split(')PVEC"',1)[0].encode()
 assert programs['off']['main']==programs['on']['main']==programs['fault']['main']
 assert all(programs[mode][kind]==(w.parent/'worldsense-isolated-primary'/('actual-'+mode+'-'+kind+'-program.metal')).read_bytes()for mode in programs for kind in ['main','prefill'])
 prior=json.loads((w.parent/'worldsense-wide-cached-prefill/actual-program-proof.json').read_text());assert all(programs[mode]['main']==(w.parent/'worldsense-wide-cached-prefill'/('actual-'+mode+'-main-program.metal')).read_bytes()for mode in ['off','primary'])
 assert not programs['off']['prefill']and not programs['primary']['prefill']
 preamble=(s/'src/backends/metal/primary_isolation_shader.h').read_text().split('R"IPRIMARY(',1)[1].split(')IPRIMARY"',1)[0].encode()
 primary=(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0].encode()
 assert programs['on']['prefill']==preamble+primary+fragment and programs['fault']['prefill']==programs['on']['prefill']
 assert b'kernel void primary_bf16_decode'in programs['on']['prefill']
 assert b'kernel void primary_bf16_decode'not in programs['on']['main']
 assert b'kernel void primary_vector_bf16_gemm'not in programs['on']['main']
 (w/'actual-program-proof.json').write_text(json.dumps({'actual_candidate_main_equals_raw_control':True,'actual_candidate_prefill_equals_minimal_preamble_plus_full_primary_and_vector_fragment':True,'original_raw_lookups_use_main_library':True,'programs':{mode:{kind:{'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()}for kind,value in data.items()}for mode,data in programs.items()},'all_four_original_MAIN_and_prefill_exact_seed':True,'new_sole_program_exact_declared_separate_library':True,'scope':'Actual constructor-produced source strings and frozen host routing; not GPU machine code identity or performance.'},indent=2)+'\n')
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen_binaries.items())
 print('ACTUAL_MAIN_CANDIDATE_EQUALS_RAW_WITH_ALL_PRIMARY_KERNELS_IN_SECONDARY',flush=True)
 print('ALL4BUILDS8NATIVE_TESTS_186ORIGINAL_28SOLE_PROJECTION_RECORDS_EXACT',flush=True)
finally:
 subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
