"""Exclusive wider-value-tile GPU prerequisite and calibrated component screen; no serving promotion."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
from make_delta_value_tiles import shader

root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();tools=Path(__file__).parent
assert not out.exists();out.mkdir(parents=True);start=time.monotonic();code=1;statuses=[]

def digest(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def run(name,command):
 unchanged()
 with (out/(name+'.log')).open('w') as stream:p=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 statuses.append(dict(name=name,command=command,exit_code=p.returncode));(out/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n');assert p.returncode==0,name
 unchanged()

base=root/'vector-vision-native-investigation-20261016';context=base/'source/src/backends/metal/metal_context.mm';shared=root/'source/test2026/metal'
paths=[Path(__file__),tools/'make_delta_value_tiles.py',tools/'delta_value_tiles_gpu.mm',shared/'prepare_parallel_delta_probe.py',shared/'probe_real_chunk_delta_formulation.py',shared/'probe_chunk_delta_formulation.py',shared.parent/'pi5b/reference_decoder.py',shared/'compare_vllm_metal.py',context]
paths += [shared.parent/'pi5b'/name for name in ('reference_linear.py','reference_attention.py')]
paths.append(tools/'launch_delta_value_tiles_gpu.py')
flags=base/'build-vector-vision-width4/CMakeFiles/garnet_metal.dir/flags.make';paths.append(flags)
assert '-DGARNET_METAL_NATIVE_BFLOAT_BODY' not in flags.read_text()
identities={str(p):digest(p) for p in paths}
for i,p in enumerate(paths):
 target=out/'source-snapshot'/str(i)/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())

def unchanged():
 for name,sha in identities.items():assert digest(Path(name))==sha,name

background_checks=[];worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((shared/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'exclusive_guard','exec'),globals())
generated=out/'delta-value-tiles.metal';generated.write_bytes(shader(context.read_text()).encode())
prepare=(shared/'prepare_parallel_delta_probe.py').read_text()
for old,new in [('for chunk in (16,32,64,80):','for chunk in (1,2,16,32,64,80):'),('((1,80),(2,32))','((1,1),(1,80),(2,1),(2,80))')]:
 assert prepare.count(old)==1;prepare=prepare.replace(old,new)
prepare="import sys\nsys.path.insert(0,"+repr(str(shared))+')\n'+prepare
prepared_driver=out/'prepare.executed.py';prepared_driver.write_bytes(prepare.encode())
checkpoint=root/'models/Qwen3.5-0.8B/model.safetensors'
meta=dict(source_sha256=identities,generated_shader_sha256=digest(generated),generated_prepare_sha256=digest(prepared_driver),checkpoint_sha256=digest(checkpoint),reference_library_sha256=digest(base/'build-vector-vision-width4/libgarnet.dylib'),scope=__doc__,numeric_design=dict(profiles=208,carried_steps=3,widths=[16,32,64],padding_floats=[0,4],changed_queries_each_step=True,original_bits=True,read_only_inputs_and_padding_guards=True),timing_design=dict(profiles=8,selected_layers=[0,22],batches=[1,2],chunks=[1,80],widths=[32,64],dispatches_per_command=16,common_warmup_ms=2000,null_before_comparison=True,rounds=10,order='ABBA',warm_calls_per_window=3,samples_per_window=5,all_samples_retained=True),admission=dict(each_workload_all_four_endpoints=True,minimum_gpu_and_wall_gain=.05,own_null_gpu_and_wall_fraction=[.95,1.05],complete_unchanged_confirmation_required=True,no_favorable_third_repeat=True),limitations='Component repeated input/command calibration only; model reuse, complete inference performance, native ownership/route tests and fresh vLLM comparisons are separate. Wider registers may be slower.',full_goal_met=False)
assert meta['reference_library_sha256']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
 guard();quiet=time.monotonic()
 while time.monotonic()-quiet<5:time.sleep(.5);guard()
 run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(tools/'delta_value_tiles_gpu.mm'),'-o',str(out/'delta_value_tiles_gpu')])
 meta['binary_sha256']=digest(out/'delta_value_tiles_gpu');(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 run('prepare',[sys.executable,str(prepared_driver),str(checkpoint.parent),str(out/'assets')])
 cases=json.loads((out/'assets/manifest.json').read_text())['cases'];assert len(cases)==104 and sum(c['name'].startswith('real-layer') for c in cases)==8
 for case in cases:
  for key in ('prepared','state','active'):assert digest(out/'assets'/case[key])==case[key+'_sha256']
 for mode in ('numeric','timing'):
  guard();(out/(mode+'-before-processes.txt')).write_text(snapshot());run(mode,[str(out/'delta_value_tiles_gpu'),str(generated),str(out/'assets'),mode]);guard();(out/(mode+'-after-processes.txt')).write_text(snapshot())
  records=[json.loads(line) for line in (out/(mode+'.log')).read_text().splitlines()];summary=records[-1];assert summary['kind']=='complete' and summary['mode']==mode
  if mode=='numeric':assert summary['profiles']==208 and summary['checks']==1872 and all(v['original_bits_equal'] for v in records if v['kind']=='check')
  else:assert summary['profiles']==8 and summary['timing_samples']==6400
  (out/(mode+'-summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
 unchanged();code=0
finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start,full_goal_met=False))+'\n')
