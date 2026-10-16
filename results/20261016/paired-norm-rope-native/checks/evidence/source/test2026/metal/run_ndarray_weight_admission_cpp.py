"""Original compiled codec versus CPU diagnostic; preserves silent compiler logs and exit statuses."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path(sys.argv[1]).resolve();prior=root/'logs/ndarray-weight-admission-20261016';out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
assert json.loads((prior/'exit.json').read_text())['exit_code']==0
start=time.monotonic();code=1;stages=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
    paths=[Path(__file__),Path(__file__).with_name('ndarray_weight_admission_cpu.mm'),prior/'source-snapshot/exact_half_codec.h']
    snap=out/'source-snapshot';snap.mkdir()
    for p in paths:(snap/p.name).write_bytes(p.read_bytes())
    metadata=dict(source_sha256={p.name:digest(p) for p in paths},prior_report_sha256=digest(prior/'report.json'),
        command=sys.argv,compiler=subprocess.check_output(['xcrun','clang++','--version'],text=True),
        scope='CPU-only original exact-half codec; every exported non-common matrix, no GPU/inference/speed claim.',full_goal_met=False)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));(out/'pid.json').write_text(json.dumps(dict(pid=os.getpid())))
    binary=out/'weight_admission_cpu'
    for name,command in [('compile',['xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-I'+str(snap),str(snap/'ndarray_weight_admission_cpu.mm'),'-framework','Foundation','-o',str(binary)]),('audit',[str(binary),str(prior)])]:
        with (out/(name+'.log')).open('w') as stream:p=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        stages.append(dict(name=name,command=command,exit_code=p.returncode));(out/'statuses.json').write_text(json.dumps(stages,indent=2));assert p.returncode==0,name
    metadata['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    values=[json.loads(v) for v in (out/'audit.log').read_text().splitlines() if v.startswith('{')]
    assert values[-1]['kind']=='complete' and values[-1]['matrices']==5
    assert len(values)==6 and sum(not r['exact_half_ready'] for r in values[:-1])==1
    (out/'report.json').write_text(json.dumps(dict(records=values,full_goal_met=False),indent=2));print(json.dumps(values[-1]));code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start)))
