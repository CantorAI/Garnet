"""Build and validate a bounded exact-half joined prefill; preserve control."""
import ast,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';build=r/'build-ndarray-selective-prefill-packedguard';statuses=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
raw_snapshot=snapshot;owned_group=None
owned_ndarray_binary=Path(sys.argv[2]).resolve()/'ndarray_selective_prefill_cache_gpu'
def snapshot():
    retained=[];owned=0
    for line in raw_snapshot().splitlines():
        fields=line.split()
        if owned_group is not None and fields and (fields[-1].lower().endswith('/xlang3') or fields[-1]==str(owned_ndarray_binary)):
            try:group=os.getpgid(int(fields[0]))
            except ProcessLookupError:continue
            if group==owned_group:
                if fields[-1].lower().endswith('/xlang3'):owned+=1
                continue
        retained.append(line)
    if owned>1:raise RuntimeError('Multiple xlang3 workers in owned validation group')
    return '\n'.join(retained)
def guard(label):
    import shutil
    if shutil.disk_usage(r).free<2*2**30:raise OSError('Disk free space below 2 GiB')
    strict_guard()
    for line in snapshot().splitlines():
        fields=line.split()
        if fields and fields[-1].lower().endswith(('_gpu','_probe')):
            raise RuntimeError('Other GPU probe PID '+fields[0])
    (output/(label+'-processes.txt')).write_text(snapshot())
def run_once(name,command):
    global owned_group
    guard(name+'-before');print('Starting',name,flush=True);begin=time.monotonic();samples=[];interruption=None
    with (output/(name+'.log')).open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);owned_group=process.pid
        try:
            while process.poll() is None:
                guard(name+'-monitor-current');samples.append(dict(elapsed_seconds=time.monotonic()-begin,guard_passed=True))
                if time.monotonic()-begin>1800:raise TimeoutError('Owned build/check exceeded deadline')
                time.sleep(.5)
            code=process.returncode
        except Exception as error:
            interruption=str(error);code=1
            if process.poll() is None:
                assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        finally:owned_group=None
    (output/(name+'-monitor.json')).write_text(json.dumps(samples,indent=2))
    statuses.append(dict(name=name,command=command,exit_code=code,interruption=interruption,wall_seconds=time.monotonic()-begin))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2));assert code==0,(name,interruption)
    guard(name+'-after');print('Completed',name,flush=True)

def run(name,command):
    global output,statuses
    parent=output;completed=statuses
    commands=[command]
    if name.endswith('-exact-tails'):
        assert command[1].endswith('test_exact_dense_tail_reference.py')
        commands=[[command[0],str(s/'test_exact_dense_tail_case_reference.py'),*command[2:],str(i)] for i in range(8)]
    accepted=[];start=time.monotonic()
    for index,call in enumerate(commands):
        for attempt in range(20):
            quiet=None;wait_start=time.monotonic();waits=[]
            while True:
                now=time.monotonic()
                try:guard(name+'-quiet')
                except RuntimeError as error:quiet=None;waits.append(dict(elapsed_seconds=now-wait_start,reason=str(error)))
                else:
                    if quiet is None:quiet=now
                    if now-quiet>=5:break
                (parent/(name+'-quiet-waits.json')).write_text(json.dumps(waits,indent=2))
                if now-wait_start>900:raise TimeoutError('No five-second quiet validation interval')
                time.sleep(.5)
            folder=parent/'attempts'/name/('case'+str(index)+'-attempt'+str(attempt));folder.mkdir(parents=True,exist_ok=False)
            output=folder;statuses=[];failure=None
            try:run_once(name,call)
            except (AssertionError,RuntimeError) as error:failure=str(error)
            finally:
                observed=list(statuses);output=parent;statuses=completed
            if failure is not None:
                if 'Another inference runtime is active' not in failure and 'Other GPU probe PID' not in failure:raise RuntimeError('Validation failed for non-contention reason: '+failure)
                print('Discarded interrupted validation case',name,index,attempt,flush=True);continue
            assert len(observed)==1 and observed[0]['exit_code']==0
            accepted.append(dict(case_index=index,command=call,evidence=str(folder.relative_to(parent)),status=observed[0]));break
        else:raise RuntimeError('Repeated contention exhausted validation case')
    status=dict(name=name,command=command,exit_code=0,interruption=None,wall_seconds=time.monotonic()-start,accepted_cases=accepted)
    completed.append(status);(parent/'statuses.json').write_text(json.dumps(completed,indent=2))
    print('Completed all clean cases for',name,flush=True)

external=[Path(__file__),Path(__file__).with_name('run_ndarray_native_checks.py'),Path(__file__).with_name('run_ndarray_native_checks_repaired.py'),Path(__file__).with_name('launch_ndarray_selective_native_checks.py')]
external_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in external}
for p in external:
    dest=output/'external-driver-snapshot'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
(output/'external-driver-identities.json').write_text(json.dumps(external_hashes,indent=2)+'\n')
assert not build.exists(),'Keep existing build artifacts; use a fresh candidate directory'
options=['METAL','METAL_XMODEL','METAL_EXACT_HALF_PREFILL','METAL_MPS_VISION_SCORES','METAL_PARALLEL_DELTA_SPLIT','METAL_FLOAT_DENSE_KEY_PREFILL','METAL_LINEAR_CHUNK_DIRECT_WRITE','METAL_LINEAR_CHUNK_CONV','METAL_LINEAR_CHUNK_DELTA','METAL_SMALL_OUTPUT_DECODE','METAL_QWEN35_EXTENDED_PREFILL','METAL_RESIDENT_KV_INPLACE','METAL_ATTENTION_PREFIX','METAL_GLOBAL_HALF_ALPHA','METAL_SCREENED_HEAD','METAL_PREFILL_STREAM','METAL_DIRTY_PREFIX_RESET','METAL_SERIAL_VISION_PREFETCH','METAL_SMALL_SCREEN_SEED','METAL_INTERLEAVED_SCREEN_INDEX','METAL_QWEN35_WIDE_BATCH_PREFILL','METAL_INTERLEAVED_BF16_BODY','METAL_COALESCED_BF16_BODY']
command=['/opt/homebrew/bin/cmake','-S',str(r/'source'),'-B',str(build)]+['-DGARNET_'+v+'=ON' for v in options]+['-DXLANG3_SDK_ROOT=/Users/zsdx/CantorAI/xlang3/sdk','-DGARNET_XLANG3_RUNTIME_LIBRARY=/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/libxlang3_runtime.dylib','-DCMAKE_BUILD_TYPE=Release']
run('configure',command);run('build',['/opt/homebrew/bin/cmake','--build',str(build),'--parallel','4'])
control=r/'build-resident-kv-inplace/libgarnet.dylib'
metadata=dict(candidate_sha256=hashlib.file_digest((build/'libgarnet.dylib').open('rb'),'sha256').hexdigest(),control_sha256=hashlib.file_digest(control.open('rb'),'sha256').hexdigest(),options=options,source_sha256={name:hashlib.file_digest((r/'source'/name).open('rb'),'sha256').hexdigest() for name in ('src/backends/metal/metal_context.mm','cmake/GarnetMetal.cmake','src/entry/metal_main.cpp','src/backends/metal/compiled_graph.cpp','src/backends/metal/metal_context.h','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','xModel/qwen/metal_dense/decoder.py','xModel/qwen35/vl_0_8b/metal_decoder.py','test2026/metal/run_dense_reference.py','test2026/metal/run_joined_prefill_checks.py')})
assert metadata['control_sha256']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'
(output/'metadata.json').write_text(json.dumps(metadata,indent=2))
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3';py=sys.executable;models=r/'models'
run('ndarray-cache-compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-I'+str(r/'source/src/backends/metal'),str(s/'ndarray_selective_prefill_cache_gpu.mm'),str(build/'libgarnet_metal.a'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(output/'ndarray_selective_prefill_cache_gpu')])
run('ndarray-cache',[str(output/'ndarray_selective_prefill_cache_gpu')])
run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])
run('vision-full-grid',[py,str(s/'test_serial_vision_grid.py'),str(r),str(output/'full-grid'),str(build)])
run('screened-real-heads',[py,str(s/'test_screened_head_real.py'),str(r),str(output/'trained-heads'),str(build)])
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
    run(model+'-exact-tails',[py,str(s/'test_exact_dense_tail_reference.py'),runtime,str(build),str(r/'source'),str(models/model),'bf16'])
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    run(model+'-private-kv',[py,str(s/'test_resident_kv_reference.py'),runtime,str(build),str(r/'source'),str(models/model)])
tasks=[('serving','test_decoder_model.py',[]),('mixed-image-pipeline','test_image_pipeline.py',[]),('qwen35-layers','test_chunk_graph.py',[str(models/'Qwen3.5-0.8B'),'bf16'])]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):tasks.append((model+'-layers','test_dense_reference.py',[str(models/model),'bf16']))
for model in ('Qwen3.5-0.8B','Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):tasks.append((model+'-head','test_head_bf16.py',[str(models/model)]))
tasks += [('vision-small','test_vision_graph.py',[]),('vision-pretrained','test_pretrained_vision_graph.py',[str(models/'Qwen3.5-0.8B')])]
for name,script,args in tasks:run(name,[py,str(s/script),runtime,str(build),str(r/'source')]+args)
run('qwen35-extended-prefill',[py,str(s/'test_qwen35_extended_prefill_reference.py'),runtime,str(build),str(r/'source'),str(models/'Qwen3.5-0.8B')])
(output/'CTest-LastTest.log').write_bytes((build/'Testing/Temporary/LastTest.log').read_bytes())
assert len(statuses)==23
print('All global-half-alpha checks, exact tails and private KV suites passed',flush=True)

for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for label,helper in [('prefix','test_prefix_reset_session_reference.py'),('screened','test_interleaved_head_session_reference.py')]:
        run(model+'-'+label+'-recovery',[sys.executable,str(s/helper),runtime,str(build),str(r/'source'),str(r/'models'/model),str(output/(label+'-recovery-'+model))])
assert len(statuses)==29 and all(v['exit_code']==0 for v in statuses)
print('Original19, exact trained vision, trained heads and both recovery suites PASS',flush=True)


run('wide-batch-trained-reference',[sys.executable,str(s/'test_wide_batch_prefill_reference.py'),runtime,str(build),str(r/'source'),str(r/'models/Qwen3.5-0.8B'),str(output/'wide-reference')])
assert len(statuses)==30

for name,sha in external_hashes.items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==sha
