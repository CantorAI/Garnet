"""All original native tests and core program/routes; explicit INPUTS only, no speed."""
from pathlib import Path
import subprocess,sys,json,hashlib,time,datetime,struct
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-decode-groups8/source';old=w.parent/'worldsense-request-weight-residency';n=r/'out/build/macos-arm64-release/bin';roles=['off','primary','on','fault','compat56','compat56-fault'];load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=load(w/'correctness-source-manifest.json')
assert len(manifest)==1723 and all(sha(s/k)==v for k,v in manifest.items())
assert(w/'build-controller.log').read_text().splitlines()==(w/'rebuild-controller.log').read_text().splitlines()==['BUILD PASS '+v for v in roles]
assert not(w/'native-results.json').exists(),'No unchanged retry/overwritten native attempt'
assert sha(n/'libgarnet.dylib')==sha(w/'before-libgarnet.dylib')=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
files=[w/name for name in ['native_controller.py','restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib','protocol.json','correctness-source-manifest.json','source-inverse-proof.json','source-transformations.json','primary_decode_groups8_shader.h','primary_decode_groups8_test.cpp','pre-native-fixture-amendment.txt','primary_core_isolation_test.cpp','native-design-before-implementation.txt','read-only-consumer-prior-audit.json']]
files += [n/'xlang3',n/'libxlang3_runtime.dylib']+[w.parent/'worldsense-opt'/name for name in ['design_groups8_ws72.py','setup_groups8_ws72.py','repair_fixture_groups8_ws72.py','prepare_native_ws72.py','verify_native_ws72.py']]
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role);files += [p for p in build.iterdir()if p.is_file()and(p.name.startswith('garnet_')or p.name in ['libgarnet.dylib','libgarnet_metal.a','CMakeCache.txt'])]
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
  build=r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role)
  with(w/('native-'+role+'.log')).open('w')as log:p=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'],stdout=log,stderr=subprocess.STDOUT)
  results[role]=p.returncode;(w/'native-results.json').write_text(json.dumps(results,indent=2)+'\n');print('Native10tests',role,'exit',p.returncode,flush=True)
  if p.returncode:raise RuntimeError('Native correctness failure '+role)
 reference=(r/'out/build/garnet-metal-worldsense-primary-decode-groups8-off/primary_projection_words.bin').read_bytes()
 assert all((r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role)/'primary_projection_words.bin').read_bytes()==reference for role in roles)
 assert reference==(r/'out/build/garnet-metal-worldsense-request-weight-residency-off/primary_projection_words.bin').read_bytes()
 (w/'native-word-comparison.json').write_text(json.dumps(dict(records=186,all_six_complete_original_FP32_streams_identical=True,original56_complete_stream_exact=True,bytes=len(reference),sha256=hashlib.sha256(reference).hexdigest()),indent=2)+'\n')
 reference=None;summaries={}
 cases=[(rows,inner,columns,0)for inner in [128,256]for columns in [260,1024]for rows in [1,2,16,32]]+[(rows,128,100352,0)for rows in [1,2]]+[(rows,128,260,1)for rows in [1,2]]
 for role in roles:
  build=r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role);data=(build/'primary_core_projection_words.bin').read_bytes()
  if reference is None:reference=data
  assert reference==data;position=0;count_words=0
  for rows,inner,columns,raw in cases:
   count=rows*columns;assert struct.unpack('<6Q',data[position:position+48])==(rows,inner,columns,raw,int(rows<=2 and columns<100000),count);position+=48
   expected=b''.join(struct.pack('<f',inner*(row+1)/16)*columns for row in range(rows));assert data[position:position+count*4]==expected;position+=count*4;count_words+=count
  assert position==len(data)and count_words==432804
  summary=load(build/'primary_core_projection_words.bin.summary.json');summaries[role]=summary;assert summary['complete_records']==20 and summary['complete_FP32_words']==432804 and summary['specialized_body_cases']==10 and summary['core_enabled'] is False
  assert [summary[k]for k in ['core_decode_delta','core_prefill_delta','core_head_delta']]==[0,0,0]
 (w/'core-native-word-comparison.json').write_text(json.dumps(dict(records=20,words_per_build=432804,all_six_complete_streams_equal=True,all_original_binary_fraction_references_exact=True,bytes=len(reference),sha256=hashlib.sha256(reference).hexdigest(),summaries=summaries),indent=2)+'\n')
 cases=[]
 for inner in [128,256]:
  for columns in [260,272,288,1024]:
   for rows in [1,2,16,32]:
    cases.append((rows,inner,columns,0,0))
    if rows<=2:cases.append((rows,inner,columns,0,1))
 cases += [(rows,128,columns,0,0)for columns in [100352,100368]for rows in [1,2]]
 cases += [(rows,512,columns,1,prepare)for columns in [260,272]for rows in [1,2]for prepare in [0,1]]
 assert len(cases)==60 and sum(row*col for row,inner,col,raw,prepare in cases)==804504
 group_reference=None;group_summaries={}
 for role in roles:
  build=r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role);data=(build/'primary_decode_groups8_words.bin').read_bytes()
  if group_reference is None:group_reference=data
  assert data==group_reference;position=0
  for rows,inner,columns,raw,prepare in cases:
   count=rows*columns;assert struct.unpack('<7Q',data[position:position+56])==(rows,inner,columns,raw,prepare,prepare,count);position+=56
   reference=b''.join(struct.pack('<f',inner*(row+1)/16)*columns for row in range(rows));assert data[position:position+count*4]==reference;position+=count*4
  assert position==len(data)
  summary=load(build/'primary_decode_groups8_words.bin.summary.json');group_summaries[role]=summary
  assert summary['complete_records']==60 and summary['complete_FP32_words']==804504 and summary['generic_body_cases']==summary['specialized_body_cases']==20 and summary['prefill_cases']==16 and summary['head_cases']==4 and summary['groups8_enabled']==(role in ['on','fault'])
  assert [summary[k]for k in ['groups8_decode_delta','groups8_head_delta']]==([32,4]if role in ['on','fault']else[0,0])
 (w/'groups8-native-word-comparison.json').write_text(json.dumps(dict(records=60,words_per_build=804504,all_six_complete_streams_equal=True,all_original_binary_fraction_references_exact=True,bytes=len(group_reference),sha256=hashlib.sha256(group_reference).hexdigest(),summaries=group_summaries),indent=2)+'\n')
 programs={}
 for role in roles:
  build=r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role);programs[role]={}
  for kind,name in [('main','main_shader_program.metal'),('prefill','prefill_shader_program.metal'),('core','primary_core_shader_program.metal')]:
   data=(build/name).read_bytes();(w/('actual-'+role+'-'+kind+'-program.metal')).write_bytes(data);programs[role][kind]=data
 for role,previous in [('off','off'),('primary','primary'),('compat56','on'),('compat56-fault','fault')]:
  for kind in ['main','prefill']:assert programs[role][kind]==(old/('actual-'+previous+'-'+kind+'-program.metal')).read_bytes()
 full=(s/'src/backends/metal/primary_isolation_shader.h').read_text().split('R"IPRIMARY(',1)[1].split(')IPRIMARY"',1)[0]+(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]+(s/'src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R"PVEC(',1)[1].split(')PVEC"',1)[0]
 original_primary=(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]
 selected_primary=original_primary.replace('ulong column=ulong(group)*16+simd*4;','ulong column=ulong(group)*32+simd*4;');assert original_primary.count('ulong column=ulong(group)*16+simd*4;')==2
 selected_full=full.replace(original_primary,selected_primary);assert selected_full!=full
 for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==selected_full.encode()
 for role in roles:
  selected=(r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role)/'groups8_selected_primary_program.metal').read_bytes();assert selected==(selected_primary if role in ['on','fault']else original_primary).encode();(w/('actual-'+role+'-groups8-selected-primary.metal')).write_bytes(selected)
 assert all(not programs[role]['core']for role in roles)
 (w/'actual-program-proof.json').write_text(json.dumps(dict(original_OFF_PRIMARY_COMPAT56_constructor_programs_exact=True,ON_FAULT_MAIN_exact_RAW=True,ON_FAULT_prefill_exact_two_origin_selected_primary_plus_original_preamble_vector=True,CORE_getter_and_actual_route_counters_zero_ALL_ROLES=True,programs={role:{kind:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())for kind,data in group.items()}for role,group in programs.items()},scope='Actual constructor source strings and native route counters, not GPU ISA/speed proof.'),indent=2)+'\n')
 assert all(sha(s/k)==v for k,v in manifest.items())and all(sha(Path(k))==v for k,v in frozen.items())
 print('ALL6ROLES10TESTS_ORIGINAL186_NEW20RECORDS432804WORDS_PROGRAM_PARTITION_AND_ACTUAL_ROUTE_COUNTERS_PASS',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
