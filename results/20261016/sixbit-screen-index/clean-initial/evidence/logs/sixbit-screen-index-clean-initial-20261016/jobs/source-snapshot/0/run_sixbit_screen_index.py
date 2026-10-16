"""Exclusive learned-query six-bit rejection index versus retained interleaved INT8 screening."""
from pathlib import Path
import ast,hashlib,json,mmap,os,re,shlex,struct,subprocess,sys,time
from make_sixbit_screen_index import context,header,index,native_impl,shader
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();tools=Path(__file__).parent;shared=root/'source/test2026/metal';base=root/'vector-vision-native-investigation-20261016/source/src/backends/metal';build=root/'vector-vision-native-investigation-20261016/build-vector-vision-width4'

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

prior=root/'logs/nibble-screen-index-20261016';pid=json.loads((prior/'pid.json').read_text())['pid'];state=subprocess.run(['/bin/ps','-p',str(pid),'-o','stat='],capture_output=True,text=True);assert state.returncode in (0,1) and (not state.stdout.strip() or state.stdout.strip().startswith('Z'))
assert json.loads((prior/'exit.json').read_text())['exit_code']==0 and digest(root/'nibble-screen-index-20261016.tar.gz')=='de0ccb3e0085a73793d839b0e8d403ab175ea538ee13dd1e4b64a9fc249e8a85'
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
   while process.poll() is None:
    check(process.pid);rss=subprocess.run(['/bin/ps','-p',str(process.pid),'-o','rss='],capture_output=True,text=True).stdout.strip();samples.append(dict(elapsed_seconds=time.monotonic()-start,guard_passed=True,rss_bytes=int(rss)*1024 if rss else None));time.sleep(.5)
  except BaseException:
   if process.poll() is None:
    assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,15);process.wait()
   raise
 (out/(name+'-monitor.json')).write_text(json.dumps(samples,indent=2)+'\n');statuses.append(dict(name=name,command=command,exit_code=process.returncode));(out/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n');assert process.returncode==0,name
 unchanged();check()

flags=build/'CMakeFiles/garnet_metal.dir/flags.make';match=re.search(r'^OBJCXX_DEFINES = (.*)$',flags.read_text(),re.M) or re.search(r'^CXX_DEFINES = (.*)$',flags.read_text(),re.M);assert match
defines=[v for v in shlex.split(match.group(1)) if v.startswith('-DGARNET_METAL_')];assert all(v in defines for v in ('-DGARNET_METAL_SCREENED_HEAD=1','-DGARNET_METAL_INTERLEAVED_SCREEN_INDEX=1','-DGARNET_METAL_SMALL_SCREEN_SEED=1'))
paths=[Path(__file__),tools/'make_nibble_screen_index.py',tools/'make_sixbit_screen_index.py',tools/'sixbit_screen_index_gpu.mm',shared/'compare_vllm_metal.py',flags,*[p for p in base.iterdir() if p.suffix in ('.h','.mm')]]
identities={str(p):digest(p) for p in paths}
for i,p in enumerate(paths):
 target=out/'source-snapshot'/str(i)/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
generated=out/'generated';generated.mkdir();texts={'metal_context.h':header((base/'metal_context.h').read_text()),'metal_context.mm':context((base/'metal_context.mm').read_text()),'screened_head_index.h':index((base/'screened_head_index.h').read_text()),'screened_head_native_impl.h':native_impl((base/'screened_head_native_impl.h').read_text()),'nibble_screened_head_shader.h':shader((base/'screened_head_shader.h').read_text()),'nibble_screen_index_gpu.mm':(tools/'sixbit_screen_index_gpu.mm').read_text()}
for name,text in texts.items():(generated/name).write_bytes(text.encode())
meta=dict(source_sha256=identities,generated_sha256={p.name:digest(p) for p in generated.iterdir()},defines=defines,reference_library_sha256=digest(build/'libgarnet.dylib'),scope=__doc__,assets={},numeric_design=dict(models_before_any_timing=True,captured_queries_per_model=17,policies=[0,1],seed_rows=1024,control_index_bits=8,candidate_index_bits=6,all_retained_original_bits=True,all_rejected_original_scores_below_seed=True,original_query_guards_and_error_allowances=True,unsafe_patterns=['zero','tiny','large','NaN','Inf','mixed_safe_unsafe'],pending_foreign_abort_recovery_uninitialized=True),timing_design=dict(conditioning_ms_per_model_batch=2000,ordinary_steps_per_batch=8,null_before_comparison_per_query=True,rounds=10,order='ABBA',warm_calls_per_window=3,samples_per_window=5,all_samples_retained=True,expected_samples=19200),admission=dict(unmet_b1_models=['Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'],each_unmet_gpu_and_wall_fraction_max=.95,all_profiles_gpu_and_wall_fraction_max=1.02,all_null_gpu_and_wall_fraction=[.95,1.05],complete_unchanged_confirmation_required=True,no_favorable_third_repeat=True),limitations='Six-bit follow-up after rejected four-bit survivor inflation; Captured trained queries only, not current full inference or certification of the original screening bounds for arbitrary future queries. Nibble index filters only; original BF16/FP32 exact seed/refinement and complete full-logit route retained. All captured full logits reexecuted and checked before timing. No published model throughput/TTFT/vLLM gain.',full_goal_met=False)
models=('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B');assets={}
try:
 check();quiet=time.monotonic()
 while time.monotonic()-quiet<5:time.sleep(.5);check()
 for model in models:
  folder=root/'logs/screened-head-after-vision-20261015/jobs'/(model+'-assets');info=json.loads((folder/'manifest.json').read_text());checkpoint=root/'models'/model/'model.safetensors';assert digest(checkpoint)==info['checkpoint_sha256'] and digest(folder/'raw.bin')==info['raw_sha256'] and len(info['queries'])==17;assets[model]=folder
  with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
   length=struct.unpack_from('<Q',mapping)[0];data_start=8+length;tensors=json.loads(mapping[8:data_start]);item=tensors[info['head_tensor']];assert item['dtype']=='BF16' and item['shape']==[info['rows'],info['inner']];lo,hi=item['data_offsets'];assert hashlib.sha256(mapping[data_start+lo:data_start+hi]).hexdigest()==info['raw_sha256']
  for query in info['queries']:
   for suffix,key in (('.input.bin','input_sha256'),('.reference.bin','reference_sha256')):assert digest(folder/(query['name']+suffix))==query[key]
  meta['assets'][model]=dict(directory=str(folder),manifest=info);target=out/'assets'/model;target.mkdir(parents=True);(target/'manifest.json').write_bytes((folder/'manifest.json').read_bytes())
 meta['environment']=dict(hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize','hw.model')},python=sys.version,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True))
 (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');binary=out/'nibble_screen_index_gpu'
 run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3',*defines,'-I'+str(generated),'-I'+str(base),str(generated/'nibble_screen_index_gpu.mm'),str(generated/'metal_context.mm'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(binary)])
 meta['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 for mode in ('numeric','timing'):
  for model in models:
   name=model+'-'+mode;run(name,[str(binary),str(assets[model]),model,mode]);records=[json.loads(line) for line in (out/(name+'.log')).read_text().splitlines() if line.startswith('{')];expected=dict(kind='complete',mode=mode,checks=34 if mode=='numeric' else 0,fallback_checks=22 if mode=='numeric' else 0,ownership_checks=34 if mode=='numeric' else 0,timing_samples=6400 if mode=='timing' else 0,full_goal_met=False);assert records[-1]==expected
 unchanged();code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start,full_goal_met=False))+'\n')
