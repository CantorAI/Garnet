"""Run each dtype compatibility check in a separate exclusive child process."""
import ast,hashlib,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
r=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';output=out;statuses=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
def check(owned=None):
    if shutil.disk_usage(r).free<2*2**30:raise OSError('Disk free space below 2 GiB')
    strict_guard()
    for line in snapshot().splitlines():
        f=line.split()
        if f and f[-1].lower().endswith(('_gpu','_probe')) and int(f[0])!=owned:raise RuntimeError('Other GPU probe PID '+f[0])
def run(name,command):
    check();samples=[];start=time.monotonic()
    with (out/(name+'.log')).open('w') as stream:
        p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                check(p.pid);samples.append(dict(elapsed_seconds=time.monotonic()-start,guard_passed=True))
                if time.monotonic()-start>120:raise TimeoutError('Owned compatibility child exceeded deadline')
                time.sleep(.5)
        except BaseException:
            if p.poll() is None:
                assert os.getpgid(p.pid)==p.pid;os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
            raise
    (out/(name+'-monitor.json')).write_text(json.dumps(samples,indent=2)+'\n')
    value=dict(name=name,command=command,exit_code=p.returncode);statuses.append(value)
    (out/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n');check();return p.returncode
paths=[Path(__file__),s/'indirect_command_probe.mm',s/'compare_vllm_metal.py'];metadata=dict(scope='Recorded indirect GPU commands: component timing and exact dependency checks; no model performance claim',
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')})
copy=out/'source-snapshot';copy.mkdir()
for p in paths:(copy/p.name).write_bytes(p.read_bytes())
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');check()
binary=out/'indirect_command_gpu'
assert run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(s/'indirect_command_probe.mm'),'-o',str(binary)])==0
assert run('probe',[str(binary)])==0
(out/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],statuses=statuses),indent=2)+'\n')
print(json.dumps(statuses),flush=True)
