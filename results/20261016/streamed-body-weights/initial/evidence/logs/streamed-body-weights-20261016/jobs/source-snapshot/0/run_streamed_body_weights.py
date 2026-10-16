"""Exclusive original-bit body-weight lifetime experiment, complete numeric gate before timing."""
from pathlib import Path
import ast,hashlib,json,mmap,os,re,shlex,shutil,struct,subprocess,sys,time
from make_streamed_body_weights import context,header,shader
from make_streamed_body_weight_probe import harness
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();tools=Path(__file__).parent;shared=root/'source/test2026/metal';base=root/'vector-vision-native-investigation-20261016/source/src/backends/metal';build=root/'vector-vision-native-investigation-20261016/build-vector-vision-width4'
prior=root/'logs/delta-value-tile64-native-continuation-20261016';pid=json.loads((prior/'pid.json').read_text())['pid'];state=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat='],capture_output=True,text=True);assert state.returncode in (0,1) and (not state.stdout.strip() or state.stdout.strip().startswith('Z'))
assert json.loads((prior/'exit.json').read_text())['exit_code']==0

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

assert digest(root/'delta-value-tile64-native-continuation-20261016.tar.gz')=='b22c3531007f4d234e276820c22418bce688558993937006fc8de1e2af8a2054'
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
paths=[Path(__file__),tools/'make_streamed_body_weights.py',tools/'make_streamed_body_weight_probe.py',shared/'coalesced_body_block_gpu.mm',shared/'compare_vllm_metal.py',flags,*[p for p in base.iterdir() if p.suffix in ('.h','.mm')]]
identities={str(p):digest(p) for p in paths}
for i,p in enumerate(paths):
 target=out/'source-snapshot'/str(i)/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
generated=out/'generated';generated.mkdir();(generated/'metal_context.h').write_bytes(header((base/'metal_context.h').read_text()).encode());(generated/'metal_context.mm').write_bytes(context((base/'metal_context.mm').read_text()).encode());(generated/'streamed_body_weight_shader.h').write_bytes(shader((base/'coalesced_body_block_shader.h').read_text()).encode());(generated/'streamed_body_weights_gpu.mm').write_bytes(harness((shared/'coalesced_body_block_gpu.mm').read_text()).encode())
meta=dict(source_sha256=identities,generated_sha256={p.name:digest(p) for p in generated.iterdir()},defines=defines,reference_library_sha256=digest(build/'libgarnet.dylib'),scope=__doc__,assets={},numeric_design=dict(fixed_threads=64,decode_rows=[1,2],patterns=4,models_before_any_timing=True,bit_exact_original_control=True,original_fp64_and_comparison_bounds=True),timing_design=dict(warmup_ms_per_profile=2000,null_before_comparison=True,rounds=10,order='ABBA',warm_calls_per_window=3,samples_per_window=5,all_samples_retained=True,expected_samples=2400),admission=dict(unmet_b1_models=['Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'],each_unmet_gpu_and_wall_fraction_max=.95,all_profiles_gpu_and_wall_fraction_max=1.02,all_null_gpu_and_wall_fraction=[.95,1.05],complete_unchanged_confirmation_required=True,no_favorable_third_repeat=True),limitations='Independent first-layer projection sets and changed structured inputs; component only. Original payload/metadata/layout/FP32 sum order retained; compiler register scheduling and full-model benefit unproven. No model input/prefix/image reuse, inference throughput or TTFT claim.',full_goal_met=False)
models=('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B');assets={}
try:
 check();quiet=time.monotonic()
 while time.monotonic()-quiet<5:time.sleep(.5);check()
 for model in models:
  folder=root/'logs/decoder-codec-gpu-20261013/jobs'/(model+'-assets');info=json.loads((folder/'manifest.json').read_text());checkpoint=root/'models'/model/'model.safetensors';assert digest(checkpoint)==info['checkpoint_sha256'];assets[model]=folder
  with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
   length=struct.unpack_from('<Q',mapping)[0];data_start=8+length;tensors=json.loads(mapping[8:data_start])
   for case in info['cases']:
    p=folder/case['name']/'raw.bin';item=tensors[case['tensor']];assert digest(p)==case['sha256']['raw.bin'] and item['dtype']=='BF16' and item['shape']==[case['rows'],case['inner']]
    lo,hi=item['data_offsets'];assert hashlib.sha256(mapping[data_start+lo:data_start+hi]).hexdigest()==case['sha256']['raw.bin']
  meta['assets'][model]=dict(directory=str(folder),manifest=info);target=out/'assets'/model;target.mkdir(parents=True);(target/'manifest.json').write_bytes((folder/'manifest.json').read_bytes())
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');binary=out/'streamed_body_weights_gpu'
 run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3',*defines,'-I'+str(generated),'-I'+str(base),str(generated/'streamed_body_weights_gpu.mm'),str(generated/'metal_context.mm'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(binary)])
 meta['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 for mode in ('numeric','timing'):
  for model in models:
   name=model+'-'+mode;run(name,[str(binary),str(assets[model]),model,mode]);records=[json.loads(line) for line in (out/(name+'.log')).read_text().splitlines() if line.startswith('{')];assert records[-1]['kind']=='complete' and records[-1]['mode']==mode
   if mode=='numeric':
    controls=[r for r in records if r['kind']=='controls'];checks=[r for r in records if r['kind']=='check'];assert len(controls)==1 and controls[0]['profiles']==21 and controls[0]['passed'] and len(checks)==8 and all(r['passed'] and r['finite'] and r['bitwise_equal'] and r['control_fp64_worst']<=1 and r['fp64_worst']<=1 and r['comparison_worst']<=1 for r in checks)
   else:assert records[-1]['timing_samples']==800 and len([r for r in records if r['kind']=='timing'])==800
 unchanged();code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start,full_goal_met=False))+'\n')
