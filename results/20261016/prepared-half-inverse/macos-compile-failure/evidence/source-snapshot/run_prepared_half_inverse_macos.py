"""Exhaustive actual Apple-clang integer inverse prerequisite; CPU-only, no speed comparison."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();code=1;statuses=[]
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
try:
 assert sys.platform=='darwin'
 paths=[Path(__file__),Path(__file__).with_name('prepared_half_inverse_cpu.cpp'),Path(__file__).with_name('prepared_half_inverse_bits.h'),root/'source/src/backends/metal/exact_half_codec.h'];assert digest(paths[-1])=='e8800f4983bb2786a7e376ca7784a7817176a5e6dd2d46c7e353116f99b2a4c9'
 snap=out/'source-snapshot';snap.mkdir()
 for p in paths:(snap/p.name).write_bytes(p.read_bytes())
 compiler=Path(subprocess.check_output(['/usr/bin/xcrun','-f','clang++'],text=True).strip())
 meta=dict(command=sys.argv,pid=os.getpid(),source_sha256={p.as_posix():digest(p) for p in paths},compiler=str(compiler),compiler_sha256=digest(compiler),compiler_version=subprocess.check_output([str(compiler),'--version'],text=True),python=sys.version,hardware={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')},scope=__doc__,full_goal_met=False)
 (out/'metadata.json').write_bytes((json.dumps(meta,indent=2)+'\n').encode());binary=out/'prepared_half_inverse_cpu'
 for name,argv in [('macros',[str(compiler),'-dM','-E','-x','c++','/dev/null']),('compile',[str(compiler),'-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-I'+str(snap),str(snap/'prepared_half_inverse_cpu.cpp'),'-o',str(binary)]),('exhaustive',[str(binary)])]:
  begin=time.monotonic()
  with (out/(name+'.log')).open('w') as stream:p=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT)
  statuses.append(dict(name=name,command=argv,exit_code=p.returncode,wall_seconds=time.monotonic()-begin));(out/'statuses.json').write_bytes((json.dumps(statuses,indent=2)+'\n').encode());assert p.returncode==0,name
 records=[json.loads(line) for line in (out/'exhaustive.log').read_text().splitlines()];assert len(records)==1 and records[0]['bf16_scale_cases']==records[0]['half_scale_cases']==8454144 and records[0]['invalid_scale_cases']==393216
 meta['binary_sha256']=digest(binary);(out/'metadata.json').write_bytes((json.dumps(meta,indent=2)+'\n').encode());(out/'report.json').write_bytes((json.dumps(dict(records=records,scope='Actual Apple-clang CLZ branch mathematical CPU verification; no learned weight packing, model setup or GPU benchmark.',full_goal_met=False),indent=2)+'\n').encode())
 for name,sha in meta['source_sha256'].items():assert digest(Path(name))==sha
 print(json.dumps(records[0]),flush=True);code=0
finally:(out/'exit.json').write_bytes((json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start))+'\n').encode())
