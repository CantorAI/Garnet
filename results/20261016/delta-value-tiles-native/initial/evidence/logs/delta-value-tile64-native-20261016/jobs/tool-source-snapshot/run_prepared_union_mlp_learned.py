"""Actual all-layer learned numeric gate before predetermined endpoint component timings."""
from pathlib import Path
import re,shlex
from make_prepared_union_mlp import context,header
p=Path(__file__).with_name('run_constant_bf16_probe.py');source=p.read_text();ns={'__file__':__file__}
exec(compile(source[:source.index('native=root/')],__file__,'exec'),ns);globals().update(ns)
selected=root/'vector-vision-native-investigation-20261016';base=selected/'source/src/backends/metal';build=selected/'build-vector-vision-width4'
assert digest(build/'libgarnet.dylib')=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
primitive=root/'validation-audits/prepared-union-mlp-primitive-20261016/evidence/logs/prepared-union-mlp-primitive-20261016/jobs'
prior=json.loads((primitive/'metadata.json').read_text());factory=[sha for name,sha in prior['source_sha256'].items() if name.endswith('/make_prepared_union_mlp.py')];assert factory==[digest(s/'make_prepared_union_mlp.py')]
cpu=root/'validation-audits/joined-union-half-admission-20261016/evidence/logs/joined-union-half-admission-20261016/report.json'
report=json.loads(cpu.read_text());cpu_copy=out/'cpu-report.json';cpu_copy.write_bytes(cpu.read_bytes())
generated=out/'generated';generated.mkdir();(generated/'metal_context.h').write_bytes(header((base/'metal_context.h').read_text()).encode());(generated/'metal_context.mm').write_bytes(context((base/'metal_context.mm').read_text()).encode())
flags_path=build/'CMakeFiles/garnet_metal.dir/flags.make';flags=flags_path.read_text();match=re.search(r'^OBJCXX_DEFINES = (.*)$',flags,re.M) or re.search(r'^CXX_DEFINES = (.*)$',flags,re.M);assert match
defines=[v for v in shlex.split(match.group(1)) if v.startswith('-DGARNET_METAL_')];assert defines==prior['defines']
paths=[Path(__file__),s/'make_prepared_union_mlp.py',s/'prepared_union_mlp_learned.mm',s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py',s/'launch_prepared_union_mlp_learned.py',s/'launch_quiet_prefill_residual_norm_probe.py',flags_path,*[p for p in base.iterdir() if p.suffix in ('.h','.mm')]]
metadata=dict(source_sha256={p.as_posix():digest(p) for p in paths},generated_sha256={p.name:digest(p) for p in generated.iterdir()},defines=defines,reference_library_sha256=digest(build/'libgarnet.dylib'),cpu_report_sha256=digest(cpu),
 timing_design=dict(common_warmup_ms=2000,rounds=10,window_order='ABBA',samples_per_window=5,warm_calls_per_window=3,null_before_comparison=True,all_samples_retained=True,timed_layers='First and last of each model, both serving batches; complete numeric gate first'),
 scope=__doc__,limitations='Structured synthetic inputs, not captured model activations; extra exact union weight buffers; no graph/model speed admission.',assets={},full_goal_met=False)
for i,p in enumerate(paths):
 target=out/'source-snapshot'/str(i)/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
for model in report['models']:
 checkpoint=Path(model['checkpoint']);check();actual=digest(checkpoint);assert actual==model['checkpoint_sha256'];metadata['assets'][model['model']]=dict(checkpoint_sha256=actual,bytes=checkpoint.stat().st_size,layers=model['layers'])
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');binary=out/'prepared_union_mlp_learned_probe'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3',*defines,'-I'+str(generated),'-I'+str(base),str(s/'prepared_union_mlp_learned.mm'),str(generated/'metal_context.mm'),'-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders','-o',str(binary)])
metadata['binary_sha256']=digest(binary);(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
run('numeric',[str(binary),str(cpu_copy),'numeric'])
records=[json.loads(line) for line in (out/'numeric.log').read_text().splitlines()];assert records[-1]['kind']=='complete' and records[-1]['profiles']==152 and records[-1]['fallback_cases']==8
assert len([v for v in records if v['kind']=='numeric' and v['prepared']])==150
run('timing',[str(binary),str(cpu_copy),'timing'])
for name,sha in metadata['source_sha256'].items():assert digest(Path(name))==sha
print('All-layer numeric and endpoint timing execution complete; component/model admission requires independent audit and analysis.',flush=True)
