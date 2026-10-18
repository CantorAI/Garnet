"""Native correctness only, after all five build roles terminal. No speed selection."""
from pathlib import Path
import subprocess,sys,json,hashlib,datetime,time,struct
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-simd-metadata/source';old=w.parent/'worldsense-cooperative-prefill';roles=['off','primary','on','fault','compat67'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text());source=load(w/'correctness-source-manifest.json');assert all(sha(s/k)==v for k,v in source.items())
assert (w/'build-controller.log').read_text().splitlines()==['BUILD PASS '+role for role in roles]
n=r/'out/build/macos-arm64-release/bin';assert sha(n/'libgarnet.dylib')=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
binaries=[n/'xlang3',n/'libxlang3_runtime.dylib']
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-primary-simd-metadata-'+role);binaries += [p for p in build.iterdir()if p.is_file()and(p.name.startswith('garnet_')or p.name in ['libgarnet.dylib','libgarnet_metal.a','CMakeCache.txt'])]
frozen={str(p):sha(p)for p in binaries};(w/'native-binaries-before.json').write_text(json.dumps(frozen,indent=2)+'\n');drivers={str(p):sha(p)for p in sorted(w.rglob('*'))if p.is_file()and p.suffix in ['.py','.json','.cmake']and p.name!='native-driver-identities-before-dispatch.json'};(w/'native-driver-identities-before-dispatch.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=drivers),indent=2)+'\n')
job='gui/501/com.cantorai.cantor'
try:
 subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
 for _ in range(200):
  if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
  time.sleep(.1)
 else:raise RuntimeError('Cantor did not unload')
 results={}
 for role in roles:
  build=r/('out/build/garnet-metal-worldsense-primary-simd-metadata-'+role)
  with(w/('native-'+role+'.log')).open('w')as log:p=subprocess.run(['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'],stdout=log,stderr=subprocess.STDOUT)
  results[role]=p.returncode;(w/'native-results.json').write_text(json.dumps(results,indent=2)+'\n');print('Native10tests',role,'exit',p.returncode,flush=True)
  if p.returncode:raise RuntimeError('Native correctness failed '+role)
 for filename,records in [('primary_projection_words.bin',186),('sole_projection_words.bin',28),('sole_dual_projection_words.bin',36)]:
  data=(r/('out/build/garnet-metal-worldsense-primary-simd-metadata-off')/filename).read_bytes();assert all((r/('out/build/garnet-metal-worldsense-primary-simd-metadata-'+role)/filename).read_bytes()==data for role in roles)
  assert data==(r/'out/build/garnet-metal-worldsense-cooperative-prefill-off'/filename).read_bytes()
  (w/(filename+'.comparison.json')).write_text(json.dumps(dict(records=records,all_five_complete_FP32_streams_identical=True,original67_complete_bytes_identical=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),indent=2)+'\n')
 paths={role:r/('out/build/garnet-metal-worldsense-primary-simd-metadata-'+role)/'primary_simd_metadata_words.bin'for role in roles};data=paths['on'].read_bytes();assert data==paths['fault'].read_bytes()and all(not paths[role].read_bytes()for role in ['off','primary','compat67']);pos=0;recordlist=[]
 for inner in [128,256,512]:
  for pattern in range(3):
   n=260*inner
   for prefill in [0,1]:
    for start,count in [(0,n),(1,1),(127,3),(inner-1,7),(3*inner+125,517),(n-1,1)]:
     shape=(260,inner,pattern,prefill,start,count);assert struct.unpack('6Q',data[pos:pos+48])==shape;pos+=48;raw=data[pos:pos+count*4];pos+=count*4;values=struct.unpack(str(count)+'I',raw)
     for j,value in enumerate(values):
      i=start+j;row,k=divmod(i,inner);block=(row//4)*(inner//128)+k//128;bits=(i&65535)if pattern==0 else 0x3f00+(i*13)%128
      if pattern==2 and block%3==1:bits=i*19&65535
      assert value==bits<<16
     recordlist.append(dict(shape=shape,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 assert pos==len(data)and len(recordlist)==108
 (w/'metadata-word-comparison.json').write_text(json.dumps(dict(records=recordlist,record_count=108,on_fault_complete_words_identical=True,all_original_expected_BF16_pattern_bits_independently_recomputed=True,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),indent=2)+'\n')
 programs={}
 for role in roles:
  build=r/('out/build/garnet-metal-worldsense-primary-simd-metadata-'+role);programs[role]={}
  for kind,filename in [('main','main_shader_program.metal'),('prefill','prefill_shader_program.metal')]:
   value=(build/filename).read_bytes();(w/('actual-'+role+'-'+kind+'-program.metal')).write_bytes(value);programs[role][kind]=value
 original=(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0].encode();new=(s/'src/backends/metal/primary_simd_metadata_shader.h').read_text().split('R"PSIMD(',1)[1].split(')PSIMD"',1)[0].encode()
 assert programs['primary']['main'].count(original)==1 and programs['on']['main']==programs['fault']['main']==programs['primary']['main'].replace(original,new)
 for role in ['off','primary']:
  for kind in ['main','prefill']:assert programs[role][kind]==(old/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()
 for kind in ['main','prefill']:assert programs['compat67'][kind]==(old/('actual-on-'+kind+'-program.metal')).read_bytes()
 assert all(not programs[role]['prefill']for role in ['off','primary','on','fault'])
 (w/'actual-program-proof.json').write_text(json.dumps(dict(programs={role:{kind:dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest())for kind,value in rows.items()}for role,rows in programs.items()},candidate_exact_primary_program_with_declared_shader_substitution=True,off_primary_compat67_actual_programs_exact_seed=True,no_GPU_ISA_or_performance_claim=True),indent=2)+'\n')
 assert all(sha(s/k)==v for k,v in source.items())and all(sha(Path(k))==v for k,v in frozen.items())
 assert all(sha(Path(k))==v for k,v in drivers.items())
 print('ALL5BUILDS10TESTS_186ORIGINAL_28MLP_36DUAL_108METADATA_RECORDS_EXACT',flush=True)
finally:subprocess.run([sys.executable,str(w/'restore_normal.py')],check=True)
