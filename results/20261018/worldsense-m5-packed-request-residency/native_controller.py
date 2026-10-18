"""All original native checks plus packed lifecycle; correctness only, explicit inputs."""
from pathlib import Path
import subprocess,sys,json,hashlib,datetime,time,struct
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-packed-request-residency/source';old=w.parent/'worldsense-request-weight-residency';n=r/'out/build/macos-arm64-release/bin';roles=['off','primary','on','fault','compat56','compat56-fault'];load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=load(w/'correctness-source-manifest.json');assert len(manifest)==1721 and all(sha(s/k)==v for k,v in manifest.items())
assert(w/'build-controller.log').read_text().splitlines()==['BUILD PASS '+role for role in roles]
assert not(w/'native-results.json').exists(),'No unchanged retry/overwriting a native attempt'
assert sha(n/'libgarnet.dylib')==sha(w/'before-libgarnet.dylib')=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
files=[w/name for name in ['native_controller.py','restore_normal.py','wait_cantor.py','cases.json','before-libgarnet.dylib','protocol.json','correctness-source-manifest.json','source-transformations.json','packed-context-adaptation-proof.json','packed-session-adaptation-proof.json','packed_request_residency_test.cpp','native-design-before-implementation.txt']]
files += [n/'xlang3',n/'libxlang3_runtime.dylib']
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-packed-request-residency-'+role);files += [p for p in build.iterdir()if p.is_file()and(p.name.startswith('garnet_')or p.name in ['libgarnet.dylib','libgarnet_metal.a','CMakeCache.txt'])]
frozen={str(p):sha(p)for p in files};(w/'native-input-identities-before-dispatch.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=frozen,scope='Explicit INPUTS and binaries only; generated results excluded, no private auth helper.'),indent=2)+'\n')
job='gui/501/com.cantorai.cantor'
try:
 subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
 for _ in range(200):
  if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
  time.sleep(.1)
 else:raise RuntimeError('Cantor did not unload')
 results={}
 for role in roles:
  build=r/('out/build/garnet-metal-worldsense-packed-request-residency-'+role)
  with(w/('native-'+role+'.log')).open('w')as log:p=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'],stdout=log,stderr=subprocess.STDOUT)
  results[role]=p.returncode;(w/'native-results.json').write_text(json.dumps(results,indent=2)+'\n');print('Native9tests',role,'exit',p.returncode,flush=True)
  if p.returncode:raise RuntimeError('Native correctness failure '+role)
 reference=(r/'out/build/garnet-metal-worldsense-packed-request-residency-off/primary_projection_words.bin').read_bytes();assert all((r/('out/build/garnet-metal-worldsense-packed-request-residency-'+role)/'primary_projection_words.bin').read_bytes()==reference for role in roles)
 assert reference==(r/'out/build/garnet-metal-worldsense-request-weight-residency-off/primary_projection_words.bin').read_bytes()
 (w/'native-word-comparison.json').write_text(json.dumps(dict(records=186,all_six_complete_original_FP32_streams_identical=True,original56_complete_stream_exact=True,bytes=len(reference),sha256=hashlib.sha256(reference).hexdigest()),indent=2)+'\n')
 reference=None;leased_reference=None
 for role in roles:
  build=r/('out/build/garnet-metal-worldsense-packed-request-residency-'+role);p=build/'packed_request_projection_words.bin';data=p.read_bytes()
  if reference is None:reference=data
  assert data==reference;position=0
  for pattern,value in [(0,128.),(1,-1.25)]:
   assert struct.unpack('4Q',data[position:position+32])==(260,128,pattern,260);position+=32;raw=data[position:position+1040];position+=1040;assert raw==struct.pack('<f',value)*260
  assert position==len(data)
  extra=Path(str(p)+'.leased.bin').read_bytes()
  if role in ['on','fault']:
   assert extra==data
   if leased_reference is None:leased_reference=extra
   assert extra==leased_reference
  else:assert not extra
 (w/'packed-native-word-comparison.json').write_text(json.dumps(dict(records=2,words_per_reference=520,all_six_complete_reference_streams_identical=True,both_ON_FAULT_complete_leased_streams_match=True,exact_binary_fraction_reference_values=[128.,-1.25],bytes=len(reference),sha256=hashlib.sha256(reference).hexdigest()),indent=2)+'\n')
 programs={}
 for role in roles:
  build=r/('out/build/garnet-metal-worldsense-packed-request-residency-'+role);programs[role]={}
  for kind,filename in [('main','main_shader_program.metal'),('prefill','prefill_shader_program.metal')]:
   raw=(build/filename).read_bytes();(w/('actual-'+role+'-'+kind+'-program.metal')).write_bytes(raw);programs[role][kind]=raw
 for role in ['primary','on','fault']:assert programs[role]==programs['primary']
 for role,prior in [('off','off'),('primary','primary'),('compat56','on'),('compat56-fault','fault')]:
  for kind in ['main','prefill']:assert programs[role][kind]==(old/('actual-'+prior+'-'+kind+'-program.metal')).read_bytes()
 assert all(not programs[role]['prefill']for role in ['off','primary','on','fault'])
 (w/'actual-program-proof.json').write_text(json.dumps(dict(original_OFF_PRIMARY_and_COMPAT56_constructor_programs_exact=True,ON_FAULT_programs_exact_PRIMARY=True,programs={role:{kind:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())for kind,raw in rows.items()}for role,rows in programs.items()},scope='Actual constructor-produced Metal source strings, not GPU ISA/residency/performance proof.'),indent=2)+'\n')
 assert all(sha(s/k)==v for k,v in manifest.items())and all(sha(Path(k))==v for k,v in frozen.items())
 print('ALL6ROLES9TESTS_186ORIGINAL_COMPLETE_RECORDS_PACKED_OWNERS_BUDGETS_FAILURES_AND_SHADER_PROGRAMS_PASS',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
