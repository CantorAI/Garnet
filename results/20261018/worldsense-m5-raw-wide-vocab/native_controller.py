"""All original native tests and core program/routes; explicit INPUTS only, no speed."""
from pathlib import Path
import subprocess,sys,json,hashlib,time,datetime,struct
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-raw-wide-vocab/source';old=w.parent/'worldsense-request-weight-residency';n=r/'out/build/macos-arm64-release/bin';roles=['off','primary','on','fault','compat56','compat56-fault'];load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=load(w/'correctness-source-manifest.json')
assert len(manifest)==1723 and all(sha(s/k)==v for k,v in manifest.items())
assert(w/'rebuild-controller.log').read_text().splitlines()==['BUILD PASS '+v for v in roles]
assert not(w/'native-results.json').exists(),'No unchanged retry/overwritten native attempt'
assert sha(n/'libgarnet.dylib')==sha(w/'before-libgarnet.dylib')=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
files=[w/name for name in ['native_controller.py','restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib','protocol.json','correctness-source-manifest.json','source-inverse-proof.json','primary_core_isolation_test.cpp','native-design-before-implementation.txt','read-only-consumer-prior-audit.json','source-transformations.json','prospective-byte-budget.json','pre-native-fixture-repair.json','raw_wide_vocab_test.cpp','build.py','build-plan.json']]
files += [n/'xlang3',n/'libxlang3_runtime.dylib']+[w.parent/'worldsense-opt'/name for name in ['design_raw_wide_vocab_ws75.py','setup_raw_wide_vocab_ws75.py','repair_native_fixture_ws75.py','prepare_native_ws75.py','verify_native_ws75.py']]
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-raw-wide-vocab-'+role);files += [p for p in build.iterdir()if p.is_file()and(p.name.startswith('garnet_')or p.name in ['libgarnet.dylib','libgarnet_metal.a','CMakeCache.txt'])]
frozen={str(p):sha(p)for p in files};assert all(Path(k).name!='worldsense-http.py'for k in frozen)
(w/'native-input-identities-before-dispatch.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=frozen,scope='Explicit INPUTS and binaries only; generated outputs and private authentication helper excluded.'),indent=2)+'\n')
job='gui/501/com.cantorai.cantor'
try:
 subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
 for _ in range(200):
  if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
  time.sleep(.1)
 else:raise RuntimeError('Service unload did not complete')
 results={}
 for role in roles:
  build=r/('out/build/garnet-metal-worldsense-raw-wide-vocab-'+role)
  with(w/('native-'+role+'.log')).open('w')as log:p=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'],stdout=log,stderr=subprocess.STDOUT)
  results[role]=p.returncode;(w/'native-results.json').write_text(json.dumps(results,indent=2)+'\n');print('Native10tests',role,'exit',p.returncode,flush=True)
  if p.returncode:raise RuntimeError('Native correctness failure '+role)
 reference=(r/'out/build/garnet-metal-worldsense-raw-wide-vocab-off/primary_projection_words.bin').read_bytes()
 assert all((r/('out/build/garnet-metal-worldsense-raw-wide-vocab-'+role)/'primary_projection_words.bin').read_bytes()==reference for role in roles)
 assert reference==(r/'out/build/garnet-metal-worldsense-request-weight-residency-off/primary_projection_words.bin').read_bytes()
 (w/'native-word-comparison.json').write_text(json.dumps(dict(records=186,all_six_complete_original_FP32_streams_identical=True,original56_complete_stream_exact=True,bytes=len(reference),sha256=hashlib.sha256(reference).hexdigest()),indent=2)+'\n')
 reference=None;summaries={}
 cases=[(rows,inner,columns,0)for inner in [128,256]for columns in [260,1024]for rows in [1,2,16,32]]+[(rows,128,100352,0)for rows in [1,2]]+[(rows,128,260,1)for rows in [1,2]]
 for role in roles:
  build=r/('out/build/garnet-metal-worldsense-raw-wide-vocab-'+role);data=(build/'primary_core_projection_words.bin').read_bytes()
  if reference is None:reference=data
  assert reference==data;position=0;count_words=0
  for rows,inner,columns,raw in cases:
   count=rows*columns;assert struct.unpack('<6Q',data[position:position+48])==(rows,inner,columns,raw,int(rows<=2 and columns<100000),count);position+=48
   expected=b''.join(struct.pack('<f',inner*(row+1)/16)*columns for row in range(rows));assert data[position:position+count*4]==expected;position+=count*4;count_words+=count
  assert position==len(data)and count_words==432804
  summary=load(build/'primary_core_projection_words.bin.summary.json');summaries[role]=summary;assert summary['complete_records']==20 and summary['complete_FP32_words']==432804 and summary['specialized_body_cases']==10 and summary['core_enabled'] is False
  assert [summary[k]for k in ['core_decode_delta','core_prefill_delta','core_head_delta']]==[0,0,0]
 (w/'core-native-word-comparison.json').write_text(json.dumps(dict(records=20,words_per_build=432804,all_six_complete_streams_equal=True,all_original_binary_fraction_references_exact=True,bytes=len(reference),sha256=hashlib.sha256(reference).hexdigest(),summaries=summaries),indent=2)+'\n')
 programs={}
 for role in roles:
  build=r/('out/build/garnet-metal-worldsense-raw-wide-vocab-'+role);programs[role]={}
  for kind,name in [('main','main_shader_program.metal'),('prefill','prefill_shader_program.metal'),('core','primary_core_shader_program.metal')]:
   data=(build/name).read_bytes();(w/('actual-'+role+'-'+kind+'-program.metal')).write_bytes(data);programs[role][kind]=data
 for role,previous in [('off','off'),('primary','primary'),('compat56','on'),('compat56-fault','fault')]:
  for kind in ['main','prefill']:assert programs[role][kind]==(old/('actual-'+previous+'-'+kind+'-program.metal')).read_bytes()
 full=(s/'src/backends/metal/primary_isolation_shader.h').read_text().split('R"IPRIMARY(',1)[1].split(')IPRIMARY"',1)[0]+(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]+(s/'src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R"PVEC(',1)[1].split(')PVEC"',1)[0]
 for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==full.encode()and programs[role]['prefill']==programs['compat56']['prefill']
 assert all(not programs[role]['core']for role in roles)
 (w/'actual-program-proof.json').write_text(json.dumps(dict(original_OFF_PRIMARY_COMPAT56_constructor_programs_exact=True,ON_FAULT_MAIN_exact_RAW=True,ON_FAULT_prefill_exact_unchanged_preamble_primary_vector_fragments=True,CORE_getter_and_actual_route_counters_zero_ALL_ROLES=True,programs={role:{kind:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())for kind,data in group.items()}for role,group in programs.items()},scope='Actual constructor source strings and native route counters, not GPU ISA/speed proof.'),indent=2)+'\n')

 reference=None;raw_summaries={};raw_cases=[(0,65536),(1,17),(127,129),(65536-33,33),(64,128),(0,4)]
 diagnostic=(s/'src/backends/metal/metal_context.mm').read_text().split('R"RAWGPU(',1)[1].split(')RAWGPU"',1)[0].encode()
 for role in roles:
  build=r/('out/build/garnet-metal-worldsense-raw-wide-vocab-'+role);data=(build/'raw_wide_vocab_words.bin').read_bytes()
  if reference is None:reference=data
  assert data==reference;position=0;total=0
  for start,count in raw_cases:
   assert struct.unpack('<2Q',data[position:position+16])==(start,count);position+=16
   bits=[(i%65280 if i%65280<32640 else i%65280+128)<<16 for i in range(start,start+count)];expected=struct.pack('<'+'I'*count,*bits)
   assert data[position:position+count*4]==expected;position+=count*4;total+=count
  assert position==len(data)and total==65847
  summary=load(build/'raw_wide_vocab_words.bin.summary.json');raw_summaries[role]=summary
  assert summary['pure_policy_boundary_cases']==192 and summary['complete_records']==6 and summary['complete_FP32_words']==65847 and summary['original_HOST_words']==65536 and summary['all_finite_BF16_patterns']==65280
  assert summary['new_policy_enabled']==(role in ['on','fault'])and summary['invalid_cases_rejected']==(6 if role=='off'else 7)and summary['same_device_other_context_pass']and summary['owner_accounting_restored']
  program=(build/'raw_bf16_gpu_program.metal').read_bytes();assert program==diagnostic;(w/('actual-'+role+'-rawGPU-program.metal')).write_bytes(program)
 (w/'rawGPU-native-word-comparison.json').write_text(json.dumps(dict(records=6,words_per_build=65847,all_six_complete_streams_equal=True,all_original_BF16_bit_references_exact=True,all65280finite_patterns_covered=True,bytes=len(reference),sha256=hashlib.sha256(reference).hexdigest(),summaries=raw_summaries,diagnostic_source_bytes=len(diagnostic),diagnostic_source_sha256=hashlib.sha256(diagnostic).hexdigest(),scope='Pure192semantic cases plus bounded original raw GPU word proof; not an actual learned vocabulary/serving/speed result.'),indent=2)+'\n')
 assert all(sha(s/k)==v for k,v in manifest.items())and all(sha(Path(k))==v for k,v in frozen.items())
 print('ALL6ROLES10TESTS_ORIGINAL186_NEW20RECORDS432804WORDS_PROGRAM_PARTITION_AND_ACTUAL_ROUTE_COUNTERS_PASS',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
