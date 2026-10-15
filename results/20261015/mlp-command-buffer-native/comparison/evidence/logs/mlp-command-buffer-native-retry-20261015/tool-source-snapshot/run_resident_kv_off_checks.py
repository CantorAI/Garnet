"""Build/check the optional private KV updater without rebuilding the control."""
import ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';build=r/'build-resident-kv-off';statuses=[];background_checks=[];resume=False
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
def guard(label):
    started=time.monotonic();waits=[]
    while True:
        try:
            strict_guard()
            for line in snapshot().splitlines():
                f=line.split()
                if f and f[-1].lower().endswith(('/split_half_prefill_probe','/lossless_block_gpu_probe','/lossless_gpu_probe','/prefill_tile_gpu','/parallel_delta_gpu')):
                    raise RuntimeError('Other GPU probe PID '+f[0])
            break
        except RuntimeError as e:
            waits.append(dict(elapsed_seconds=time.monotonic()-started,reason=str(e)))
            (output/(label+'-wait.json')).write_text(json.dumps(waits,indent=2))
            if time.monotonic()-started>900:raise
            time.sleep(5)
    (output/(label+'-processes.txt')).write_text(snapshot())

def run(name,command,gpu=False):
    if gpu:guard(name+'-before')
    print('Starting',name,flush=True);begin=time.monotonic()
    with (output/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
    statuses.append(dict(name=name,command=command,exit_code=result.returncode,wall_seconds=time.monotonic()-begin))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2));assert result.returncode==0,(name,result.returncode)
    if gpu:guard(name+'-after')
    print('Completed',name,flush=True)
if not resume:assert not build.exists(),'Use a new build directory; retain control/candidate binaries'
options=['METAL','METAL_XMODEL','METAL_EXACT_HALF_PREFILL','METAL_MPS_VISION_SCORES','METAL_PARALLEL_DELTA_SPLIT','METAL_FLOAT_DENSE_KEY_PREFILL','METAL_LINEAR_CHUNK_DIRECT_WRITE','METAL_LINEAR_CHUNK_CONV','METAL_LINEAR_CHUNK_DELTA','METAL_SMALL_OUTPUT_DECODE','METAL_QWEN35_EXTENDED_PREFILL']
command=['/opt/homebrew/bin/cmake','-S',str(r/'source'),'-B',str(build)]+['-DGARNET_'+v+'=ON' for v in options]+['-DXLANG3_SDK_ROOT=/Users/zsdx/CantorAI/xlang3/sdk','-DGARNET_XLANG3_RUNTIME_LIBRARY=/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/libxlang3_runtime.dylib','-DCMAKE_BUILD_TYPE=Release']
if not resume:
    run('configure',command);run('build',['/opt/homebrew/bin/cmake','--build',str(build),'--parallel','4'])
metadata=dict(candidate_sha256=hashlib.file_digest((build/'libgarnet.dylib').open('rb'),'sha256').hexdigest(),control_sha256=hashlib.file_digest((r/'build-qwen35-extended-prefill/libgarnet.dylib').open('rb'),'sha256').hexdigest(),options=options)
assert metadata['control_sha256']=='cc5539990dcb4effbece7c611787a656df56b51271f1f40e1c3e8b5dbf90c8f8'
(output/'metadata.json').write_text(json.dumps(metadata,indent=2))
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3';py=sys.executable;models=r/'models'
if resume:
    prior=r/'logs/resident-kv-candidate-checks-20261013/jobs'
    previous=json.loads((prior/'statuses.json').read_text());assert len(previous)==4 and all(v['exit_code']==0 for v in previous)
    assert metadata['candidate_sha256']==json.loads((prior/'metadata.json').read_text())['candidate_sha256']
    assert (prior/'Qwen3-0.6B-private-kv.log').read_text().count('private_kv_reference PASS')==8
    installed=json.loads((r/'resident-kv-manifest.json').read_text())
    for name,sha in installed['files'].items():
        if name.startswith(('src/','cmake/')):assert hashlib.file_digest((r/'source'/name).open('rb'),'sha256').hexdigest()==sha
    (output/'reused-checks.json').write_text(json.dumps(dict(prior=str(prior),successful_stages=previous,guard_interruption_retained=True),indent=2))
else:run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'],True)
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    if resume and model=='Qwen3-0.6B':continue
    run(model+'-private-kv',[py,str(s/'test_resident_kv_reference.py'),runtime,str(build),str(r/'source'),str(models/model)],True)
(output/'CTest-LastTest.log').write_bytes((build/'Testing/Temporary/LastTest.log').read_bytes())
assert len(statuses)==6
print('Default-off kernel and private-session compatibility checks passed',flush=True)
