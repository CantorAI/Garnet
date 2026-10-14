"""Exclusive CPU/GPU reset component timings, with actual model state sizes."""
import ast,hashlib,json,os,signal,statistics,subprocess,sys,time,shutil
from pathlib import Path
r=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
scripts=r/'source/test2026/metal';background_checks=[];statuses=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((scripts/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
def guard(owned=None):
    if shutil.disk_usage(r).free<2*2**30:raise OSError('Disk free space below 2 GiB')
    strict_guard()
    for line in snapshot().splitlines():
        f=line.split()
        if f and f[-1].lower().endswith(('_gpu','_probe')) and int(f[0])!=owned:raise RuntimeError('Other GPU probe PID '+f[0])
def run(name,command):
    guard();samples=[];start=time.monotonic()
    with (output/(name+'.log')).open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                guard(process.pid);samples.append(dict(elapsed_seconds=time.monotonic()-start,guard_passed=True))
                if time.monotonic()-start>600:raise TimeoutError('Owned reset component exceeded deadline')
                time.sleep(.5)
        except BaseException:
            if process.poll() is None:
                assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=15)
            raise
    (output/(name+'-monitor.json')).write_text(json.dumps(samples,indent=2)+'\n')
    statuses.append(dict(name=name,command=command,exit_code=process.returncode));(output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    assert process.returncode==0,name
profiles=[];configs={}
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    path=r/'models'/model/'config.json';raw=path.read_bytes();c=json.loads(raw);c=c.get('text_config',c)
    configs[model]=dict(sha256=hashlib.sha256(raw).hexdigest(),config=c)
    for batch in (1,2):
        columns=[]
        for kind in c.get('layer_types',['full_attention']*c['num_hidden_layers']):
            if kind=='linear_attention':
                channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim']
                columns.extend([channels*c['linear_conv_kernel_dim'],c['linear_num_value_heads']*c['linear_key_head_dim']*c['linear_value_head_dim']])
            else:
                dim=c.get('head_dim',c['hidden_size']//c['num_attention_heads']);width=c['num_key_value_heads']*256*dim;columns.extend([width,width])
        profiles.append(dict(name=model+'-b'+str(batch),batch=batch,columns=columns,logical_bytes=batch*sum(columns)*4))
profiles.append(dict(name='padded',batch=3,columns=[1,3,17,65],logical_bytes=3*(1+3+17+65)*4))
profile_path=output/'profiles.tsv';profile_path.write_text('\n'.join(p['name']+' '+str(p['batch'])+' '+str(len(p['columns']))+' '+' '.join(map(str,p['columns'])) for p in profiles)+'\n')
metadata=dict(scope='Synthetic reset component; no full-model performance claim; GPU submission and wait included in wall timing',
    policies=['temporary_zero_vector_and_copy','direct_cpu_memset','single_command_gpu_blit_fill'],profiles=profiles,configs=configs,
    hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),scripts/'state_zero_gpu.mm',scripts/'compare_vllm_metal.py')})
source_snapshot=output/'source-snapshot';source_snapshot.mkdir()
for name in metadata['source_sha256']:(source_snapshot/name).write_bytes((scripts/name).read_bytes())
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');guard();(output/'before-processes.txt').write_text(snapshot())
binary=output/'state_zero_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(scripts/'state_zero_gpu.mm'),'-o',str(binary)])
guard();time.sleep(5);guard();run('gpu',[str(binary),str(profile_path)]);guard()
rows=[json.loads(line) for line in (output/'gpu.log').read_text().splitlines() if line.startswith('{')]
checks=[v for v in rows if v['kind']=='check'];assert len(checks)==63 and all(v['cpu_gpu_positive_zero'] and v['padding_preserved'] for v in checks)
result=[]
for p in profiles:
    timings=[v for v in rows if v['kind']=='timing' and v['profile']==p['name']];assert len(timings)==12
    medians={str(policy):statistics.median(v['wall_ms_median'] for v in timings if v['policy']==policy) for policy in range(3)}
    result.append(dict(profile=p['name'],logical_bytes=p['logical_bytes'],wall_ms_median=medians,gpu_over_direct_cpu=medians['1']/medians['2']))
(output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],checks=len(checks),profiles=result),indent=2)+'\n')
(output/'after-processes.txt').write_text(snapshot());print(json.dumps(result),flush=True)
