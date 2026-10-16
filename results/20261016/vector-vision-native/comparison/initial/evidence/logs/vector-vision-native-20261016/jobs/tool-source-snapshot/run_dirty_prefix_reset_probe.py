"""Guarded bounded-write reset component: exact zero state, no serving claim."""
import ast,hashlib,json,os,signal,statistics,subprocess,sys,time,shutil
from pathlib import Path
root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
scripts=root/'source/test2026/metal';background_checks=[];statuses=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((scripts/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
def check(owned=None):
 if shutil.disk_usage(root).free<2*2**30:raise OSError('Disk free space below 2 GiB')
 strict_guard()
 for line in snapshot().splitlines():
  fields=line.split()
  if fields and fields[-1].lower().endswith(('_gpu','_probe')) and int(fields[0])!=owned:raise RuntimeError('Other GPU probe PID '+fields[0])
def run(name,command):
 check();samples=[];begin=time.monotonic()
 with (output/(name+'.log')).open('w') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while p.poll() is None:
    check(p.pid);samples.append(dict(elapsed_seconds=time.monotonic()-begin,guard_passed=True))
    if time.monotonic()-begin>600:raise TimeoutError('Owned reset probe exceeded deadline')
    time.sleep(.5)
  except BaseException:
   if p.poll() is None:
    assert os.getpgid(p.pid)==p.pid;os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
   raise
 (output/(name+'-monitor.json')).write_bytes((json.dumps(samples,indent=2)+'\n').encode())
 statuses.append(dict(name=name,command=command,exit_code=p.returncode));(output/'statuses.json').write_bytes((json.dumps(statuses,indent=2)+'\n').encode());assert p.returncode==0,name
 check()
profiles=[];configs={};capacity=256
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 raw=(root/'models'/model/'config.json').read_bytes();config=json.loads(raw);c=config.get('text_config',config)
 configs[model]=dict(sha256=hashlib.sha256(raw).hexdigest(),config=config)
 for batch in (1,2):
  for dirty in (0,1,64,256):
   states=[]
   for kind in c.get('layer_types',['full_attention']*c['num_hidden_layers']):
    if kind=='linear_attention':
     channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim']
     for col in (channels*c['linear_conv_kernel_dim'],c['linear_num_value_heads']*c['linear_key_head_dim']*c['linear_value_head_dim']):states.append(dict(columns=col,groups=1,group_columns=col,dirty_columns=col))
    else:
     dim=c.get('head_dim',c['hidden_size']//c['num_attention_heads']);groups=c['num_key_value_heads']
     for _ in range(2):states.append(dict(columns=groups*capacity*dim,groups=groups,group_columns=capacity*dim,dirty_columns=dirty*dim))
   profiles.append(dict(name=model+'-b'+str(batch)+'-dirty'+str(dirty),model=model,batch=batch,capacity=capacity,dirty_tokens=dirty,states=states))
profiles.append(dict(name='padded',model='synthetic',batch=3,capacity=17,dirty_tokens=7,states=[dict(columns=51,groups=3,group_columns=17,dirty_columns=7),dict(columns=65,groups=1,group_columns=65,dirty_columns=65)]))
manifest=output/'profiles.json';manifest.write_bytes((json.dumps(profiles,indent=2)+'\n').encode())
paths=[Path(__file__),scripts/'dirty_prefix_reset_gpu.mm',scripts/'compare_vllm_metal.py',scripts/'launch_dirty_prefix_reset_probe.py']
sources=output/'source-snapshot';sources.mkdir()
for p in paths:(sources/p.name).write_bytes(p.read_bytes())
metadata=dict(scope=__doc__,configs=configs,profiles=profiles,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 policies=['legacy_temporary_vector_copy','direct_full_cpu_zero','direct_dirty_prefix_cpu_zero'],
 conditioning='Full zero initialization; fresh bounded GPU writes before each reset, excluded from reset clock; all linear state is fully cleared',
 hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')},
 versions=dict(python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)))
(output/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode());check()
binary=output/'dirty_prefix_reset_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(scripts/'dirty_prefix_reset_gpu.mm'),'-o',str(binary)])
metadata['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest();(output/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode())
run('gpu',[str(binary),str(manifest)])
rows=[json.loads(line) for line in (output/'gpu.log').read_text().splitlines() if line.startswith('{')]
checks=[r for r in rows if r['kind']=='check'];assert len(checks)==len(profiles)*12 and all(r['all_cpu_gpu_positive_zero'] and r['padding_preserved'] for r in checks)
results=[]
for profile in profiles:
 name=profile['name'];warm=next(r for r in rows if r['kind']=='warmup' and r['profile']==name);assert warm['elapsed_ms']>=250
 timings=[r for r in rows if r['kind']=='timing' and r['profile']==name];assert len(timings)==12
 for round in (0,1):
  windows=sorted((r for r in timings if r['round']==round),key=lambda r:r['window']);assert [r['policy'] for r in windows]==[0,1,2,2,1,0]
  assert all(len(r['wall_ms'])==12 and r['all_cpu_gpu_positive_zero'] and r['padding_preserved'] for r in windows)
 means={str(policy):statistics.mean(v for r in timings if r['policy']==policy for v in r['wall_ms']) for policy in (0,1,2)}
 results.append(dict(profile=name,model=profile['model'],batch=profile['batch'],dirty_tokens=profile['dirty_tokens'],means_ms=means,prefix_over_legacy=means['2']/means['0'],prefix_over_full_direct=means['2']/means['1']))
result=dict(scope=__doc__,checks=len(checks),timed_resets=len(profiles)*144,profiles=results,full_goal_met=False)
(output/'summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result),flush=True)
