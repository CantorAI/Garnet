"""Compile original native codec and crosscheck every paired learned weight on CPU."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();code=1
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
try:
    prior=root/'logs/joined-union-half-admission-20261016/report.json';assert json.loads((prior.parent/'exit.json').read_text())['exit_code']==0
    paths=[Path(__file__),Path(__file__).with_name('joined_union_half_admission_cpu.mm'),root/'source/src/backends/metal/exact_half_codec.h']
    metadata=dict(source_sha256={p.as_posix():digest(p) for p in paths},prior_report_sha256=digest(prior),scope=__doc__,full_goal_met=False)
    (out/'pid.json').write_text(json.dumps(dict(pid=os.getpid())));(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    for p in paths:
        q=out/'source-snapshot'/p.name;q.parent.mkdir(exist_ok=True);q.write_bytes(p.read_bytes())
    binary=out/'joined_union_half_admission_cpu';statuses=[]
    for name,command in [('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-I'+str(root/'source/src/backends/metal'),str(paths[1]),'-framework','Foundation','-o',str(binary)]),('audit',[str(binary),str(prior)])]:
        with (out/(name+'.log')).open('w') as stream:p=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        statuses.append(dict(name=name,command=command,exit_code=p.returncode));(out/'statuses.json').write_text(json.dumps(statuses,indent=2));assert p.returncode==0,name
    metadata['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    records=[json.loads(line) for line in (out/'audit.log').read_text().splitlines() if line.startswith('{')];assert len(records)==77 and records[-1]['pairs']==76
    (out/'report.json').write_text(json.dumps(dict(records=records,full_goal_met=False),indent=2)+'\n');print(json.dumps(records[-1]),flush=True);code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start)))
