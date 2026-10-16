"""Compile and exhaustively verify an unused CPU bit inverse locally; no Mac workload interference."""
from pathlib import Path
import hashlib,json,os,platform,subprocess,sys,time
root=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();code=1;statuses=[]
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
try:
 assert sys.platform=='win32'
 vswhere=Path(r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe')
 install=Path(subprocess.check_output([str(vswhere),'-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True).strip());vcvars=install/'VC/Auxiliary/Build/vcvars64.bat';assert vcvars.is_file()
 versions=list((install/'VC/Tools/MSVC').iterdir());version=max(versions,key=lambda p:tuple(int(v) for v in p.name.split('.')));compiler=version/'bin/Hostx64/x64/cl.exe';assert compiler.is_file()
 sources=[Path(__file__),root/'test2026/metal/prepared_half_inverse_cpu.cpp',root/'test2026/metal/prepared_half_inverse_bits.h',root/'src/backends/metal/exact_half_codec.h'];assert digest(sources[-1])=='e8800f4983bb2786a7e376ca7784a7817176a5e6dd2d46c7e353116f99b2a4c9'
 snap=out/'source-snapshot';snap.mkdir()
 for p in sources:(snap/p.name).write_bytes(p.read_bytes())
 binary=out/'prepared_half_inverse_cpu.exe';obj=out/'prepared_half_inverse_cpu.obj';script=out/'compile.cmd'
 command=' '.join([f'"{compiler}"','/nologo','/Bv','/std:c++17','/O2','/EHsc','/fp:precise',f'/I"{snap}"',f'"{snap/sources[1].name}"',f'/Fe"{binary}"',f'/Fo"{obj}"'])
 script.write_bytes((f'@echo off\r\ncall "{vcvars}" > nul\r\nif errorlevel 1 exit /b 1\r\n{command}\r\nexit /b %errorlevel%\r\n').encode())
 metadata=dict(command=sys.argv,pid=os.getpid(),python=sys.version,platform=platform.platform(),architecture=platform.machine(),source_sha256={p.as_posix():digest(p) for p in sources},compiler=str(compiler),compiler_sha256=digest(compiler),compile_script_sha256=digest(script),inverse_branch='Portable highest-bit loop: MSVC; Apple clang builtin branch not executed here',scope=__doc__,full_goal_met=False)
 (out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
 for name,argv in [('compile',['cmd.exe','/d','/c',str(script)]),('exhaustive',[str(binary)])]:
  begin=time.monotonic()
  with (out/(name+'.log')).open('w') as stream:result=subprocess.run(argv,cwd=out,stdout=stream,stderr=subprocess.STDOUT)
  statuses.append(dict(name=name,command=argv,exit_code=result.returncode,wall_seconds=time.monotonic()-begin));(out/'statuses.json').write_bytes((json.dumps(statuses,indent=2)+'\n').encode());assert result.returncode==0,name
 records=[json.loads(line) for line in (out/'exhaustive.log').read_text().splitlines()];assert len(records)==1 and records[0]['kind']=='complete'
 assert records[0]['bf16_scale_cases']==records[0]['half_scale_cases']==8454144 and records[0]['invalid_scale_cases']==393216
 metadata['binary_sha256']=digest(binary);metadata['object_sha256']=digest(obj);(out/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
 report=dict(records=records,scope='Exhaustive Windows CPU algebraic prerequisite only. No learned packing, Mac compiler branch, M4 setup speed, GPU/native correctness or model speed claim.',full_goal_met=False)
 (out/'report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
 for name,sha in metadata['source_sha256'].items():assert digest(Path(name))==sha
 print(json.dumps(records[0]),flush=True);code=0
finally:(out/'exit.json').write_bytes((json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start))+'\n').encode())
