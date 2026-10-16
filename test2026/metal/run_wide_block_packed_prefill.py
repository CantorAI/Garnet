"""Exclusive wide-output K32/N64 BF16 prefill tiles; complete all-model original bounds before clocks."""
from pathlib import Path
import ast,hashlib,json,math,mmap,os,re,shlex,subprocess,sys,time
from make_wide_block_packed_prefill import context,header,shader,harness
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();tools=Path(__file__).parent;shared=root/'source/test2026/metal';base=root/'vector-vision-native-investigation-20261016/source/src/backends/metal';build=root/'vector-vision-native-investigation-20261016/build-vector-vision-width4'

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

prior=root/'logs/block-packed-prefill-20261016';pid=json.loads((prior/'pid.json').read_text())['pid'];state=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat='],capture_output=True,text=True);assert state.returncode in (0,1) and (not state.stdout.strip() or state.stdout.strip().startswith('Z'))
assert json.loads((prior/'exit.json').read_text())['exit_code']==0 and digest(root/'block-packed-prefill-20261016.tar.gz')=='f1eaf86ceb751984013df8b723b2ff04f45d158cb8e04d04d3347baee78f0579'
assert digest(build/'libgarnet.dylib')=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
assert not out.exists();out.mkdir(parents=True);output=out;start=time.monotonic();code=1;statuses=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((shared/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'exclusive_guard','exec'),globals());strict_guard=guard

def check(owned=None):
 strict_guard()
 for line in snapshot().splitlines():
  fields=line.split()
  if fields and fields[-1].lower().endswith(('_gpu','_probe')) and int(fields[0])!=owned:raise RuntimeError('Other GPU probe PID '+fields[0])

def unchanged():
 for name,sha in identities.items():assert digest(Path(name))==sha,name

def run(name,command):
 unchanged();check();samples=[]
 with (out/(name+'.log')).open('w') as stream:
  process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while process.poll() is None:check(process.pid);samples.append(dict(elapsed_seconds=time.monotonic()-start,guard_passed=True));time.sleep(.5)
  except BaseException:
   if process.poll() is None:
    assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,15);process.wait()
   raise
 (out/(name+'-monitor.json')).write_text(json.dumps(samples,indent=2)+'\n');statuses.append(dict(name=name,command=command,exit_code=process.returncode));(out/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n');assert process.returncode==0,name
 unchanged();check()

flags=build/'CMakeFiles/garnet_metal.dir/flags.make';match=re.search(r'^OBJCXX_DEFINES = (.*)$',flags.read_text(),re.M) or re.search(r'^CXX_DEFINES = (.*)$',flags.read_text(),re.M);assert match
defines=[v for v in shlex.split(match.group(1)) if v.startswith('-DGARNET_METAL_')];assert '-DGARNET_METAL_COALESCED_BF16_BODY=1' in defines and '-DGARNET_METAL_MLP_PREFILL_GRAPH=1' not in defines
paths=[Path(__file__),tools/'make_block_packed_prefill.py',tools/'make_wide_block_packed_prefill.py',shared/'packed_body_prefill_gpu.mm',shared/'compare_vllm_metal.py',flags,*[p for p in base.iterdir() if p.suffix in ('.h','.mm')]]
identities={str(p):digest(p) for p in paths}
for i,p in enumerate(paths):
 target=out/'source-snapshot'/str(i)/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
generated=out/'generated';generated.mkdir();(generated/'metal_context.h').write_bytes(header((base/'metal_context.h').read_text()).encode());(generated/'metal_context.mm').write_bytes(context((base/'metal_context.mm').read_text()).encode());(generated/'block_packed_prefill_shader.h').write_bytes(shader().encode());(generated/'block_packed_prefill_gpu.mm').write_bytes(harness((shared/'packed_body_prefill_gpu.mm').read_text()).encode())
meta=dict(source_sha256=identities,generated_sha256={p.name:digest(p) for p in generated.iterdir()},defines=defines,reference_library_sha256=digest(build/'libgarnet.dylib'),scope=__doc__,assets={},numeric_design=dict(rows=[[24,44],[26,52],[80,160]],patterns=4,policies=[1,2],tile_geometry=[[32,64,32],[64,64,32]],threads=256,fp32_activation_and_accumulation=True,all_models_before_any_timing=True,original_fp64_and_comparison_bounds=True),timing_design=dict(warmup_ms_per_profile=2000,null_before_comparison=True,rounds=10,order='ABBA',warm_calls_per_window=3,samples_per_window=5,all_samples_retained=True),admission=dict(each_policy_all_six_numeric_profiles_required=True,targeted_profiles=[['Qwen3-0.6B',2],['Qwen2.5-Coder-0.5B-Instruct',2],['Qwen3.5-0.8B',1],['Qwen3.5-0.8B',2]],targeted_gpu_and_wall_fraction_max=.95,all_profiles_gpu_and_wall_fraction_max=1.02,all_null_gpu_and_wall_fraction=[.95,1.05],complete_unchanged_confirmation_required=True,no_favorable_third_repeat=True),limitations='Follow-up after rejected all-eligible tiles: fixed N>1024 wide-output selection for gate/up and wide query/linear matrices, all other projections retain original control. Wide output provides more output tiles per row block; speed and original numerical gates remain unproven. Same six first-layer profiles and structured synthetic activations; no vision tower, sequential learned network, full-model or vLLM result. Added full BF16 weight copy and CPU roundtrip verification remain setup/memory costs. Original special key projections excluded by original interleaved eligibility. No ownership/abort/tail GPU suite yet.',full_goal_met=False)
models=('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B');assets={};admitted={1:True,2:True}
try:
 check();quiet=time.monotonic()
 while time.monotonic()-quiet<5:time.sleep(.5);check()
 for model in models:
  folder=root/'logs/decoder-codec-gpu-20261013/jobs'/(model+'-assets');info=json.loads((folder/'manifest.json').read_text());checkpoint=root/'models'/model/'model.safetensors';assert digest(checkpoint)==info['checkpoint_sha256'];assets[model]=folder
  import struct
  with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
   length=struct.unpack_from('<Q',mapping)[0];data_start=8+length;tensors=json.loads(mapping[8:data_start])
   for case in info['cases']:
    p=folder/case['name']/'raw.bin';item=tensors[case['tensor']];assert digest(p)==case['sha256']['raw.bin'] and item['dtype']=='BF16' and item['shape']==[case['rows'],case['inner']]
    lo,hi=item['data_offsets'];assert hashlib.sha256(mapping[data_start+lo:data_start+hi]).hexdigest()==case['sha256']['raw.bin']
  meta['assets'][model]=dict(directory=str(folder),manifest=info);target=out/'assets'/model;target.mkdir(parents=True);(target/'manifest.json').write_bytes((folder/'manifest.json').read_bytes())
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');binary=out/'block_packed_prefill_gpu'
 run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3',*defines,'-I'+str(generated),'-I'+str(base),str(generated/'block_packed_prefill_gpu.mm'),str(generated/'metal_context.mm'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(binary)])
 meta['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 for model in models:
  name=model+'-numeric';run(name,[str(binary),str(assets[model]),model,'numeric','0']);records=[json.loads(line) for line in (out/(name+'.log')).read_text().splitlines() if line.startswith('{')];assert records[-1]==dict(kind='complete',mode='numeric',numeric_checks=16,timing_samples=0,full_goal_met=False)
  checks=[r for r in records if r['kind']=='check'];assert len(checks)==16
  for policy in (1,2):
   subset=[r for r in checks if r['policy']==policy];assert {(r['batch'],r['pattern']) for r in subset}=={(b,p) for b in (1,2) for p in range(4)}
   for r in subset:
    assert r['passed']==(r['finite'] and r['fp64_worst']<=1 and r['comparison_worst']<=1) and math.isfinite(r['control_fp64_worst']) and r['control_fp64_worst']<=1
    admitted[policy]=admitted[policy] and r['passed']
 (out/'numeric-admission.json').write_text(json.dumps(dict(admitted=admitted,scope='Each policy must pass all three models, batches and patterns; no per-endpoint favorable timing',full_goal_met=False),indent=2)+'\n')
 for policy in (1,2):
  if not admitted[policy]:continue
  for model in models:
   name=model+'-policy'+str(policy)+'-timing';run(name,[str(binary),str(assets[model]),model,'timing',str(policy)]);records=[json.loads(line) for line in (out/(name+'.log')).read_text().splitlines() if line.startswith('{')];assert records[-1]==dict(kind='complete',mode='timing',numeric_checks=0,timing_samples=800,full_goal_met=False)
 unchanged();code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start,full_goal_met=False))+'\n')
