"""Regenerate all component sources and audit every original numeric, route/safety and clock record."""
from pathlib import Path
import hashlib,json,math,re,runpy,shlex,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text());assert len(proof['jobs'])==1;trial,status=next(iter(proof['jobs'].items()));assert status['exit_code']==0 and not proof['full_goal_met'];folder=root/'logs'/trial;jobs=folder/'jobs'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256'],name
assert json.loads((jobs/'exit.json').read_text())['exit_code']==0;meta=json.loads((jobs/'metadata.json').read_text());sources={}
for absolute,sha in meta['source_sha256'].items():
 name=Path(absolute).name;found=[p for p in (jobs/'source-snapshot').rglob(name) if p.is_file() and digest(p)==sha];assert len(found)==1 and name not in sources,name;sources[name]=found[0]
sys.path.insert(0,str(sources['make_streamed_body_weights.py'].parent));factory=runpy.run_path(str(sources['make_streamed_body_weights.py']));probe=runpy.run_path(str(sources['make_streamed_body_weight_probe.py']))
expected={'metal_context.mm':factory['context'](sources['metal_context.mm'].read_text()),'metal_context.h':factory['header'](sources['metal_context.h'].read_text()),'streamed_body_weight_shader.h':factory['shader'](sources['coalesced_body_block_shader.h'].read_text()),'streamed_body_weights_gpu.mm':probe['harness'](sources['coalesced_body_block_gpu.mm'].read_text())};assert set(expected)==set(meta['generated_sha256'])
for name,text in expected.items():assert (jobs/'generated'/name).read_bytes()==text.encode() and digest(jobs/'generated'/name)==meta['generated_sha256'][name]
flags=sources['flags.make'].read_text();match=re.search(r'^OBJCXX_DEFINES = (.*)$',flags,re.M) or re.search(r'^CXX_DEFINES = (.*)$',flags,re.M);assert match
assert [v for v in shlex.split(match.group(1)) if v.startswith('-DGARNET_METAL_')]==meta['defines'] and '-DGARNET_METAL_COALESCED_BF16_BODY=1' in meta['defines'] and '-DGARNET_METAL_MLP_PREFILL_GRAPH=1' not in meta['defines']
assert meta['reference_library_sha256']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
assert meta['numeric_design']==dict(fixed_threads=64,decode_rows=[1,2],patterns=4,models_before_any_timing=True,bit_exact_original_control=True,original_fp64_and_comparison_bounds=True)
assert meta['timing_design']==dict(warmup_ms_per_profile=2000,null_before_comparison=True,rounds=10,order='ABBA',warm_calls_per_window=3,samples_per_window=5,all_samples_retained=True,expected_samples=2400)
assert meta['admission']==dict(unmet_b1_models=['Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'],each_unmet_gpu_and_wall_fraction_max=.95,all_profiles_gpu_and_wall_fraction_max=1.02,all_null_gpu_and_wall_fraction=[.95,1.05],complete_unchanged_confirmation_required=True,no_favorable_third_repeat=True)
models=['Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'];stages=json.loads((jobs/'statuses.json').read_text());assert [s['name'] for s in stages]==['compile']+[m+'-numeric' for m in models]+[m+'-timing' for m in models] and all(s['exit_code']==0 for s in stages)
binary='logs/'+trial+'/jobs/streamed_body_weights_gpu';assert proof['omitted'][binary]['sha256']==meta['binary_sha256'];compile_command=stages[0]['command'];assert compile_command[0:2]==['/usr/bin/xcrun','clang++'] and compile_command[compile_command.index('-o')+1]==stages[1]['command'][0]
assert digest(folder/'launcher-source.py')==json.loads((folder/'command.json').read_text())['launcher_sha256']
checks_total=0;safety=0;words=0;matrices=0;timed=[];selected_counts={'Qwen3-0.6B':6,'Qwen2.5-Coder-0.5B-Instruct':3,'Qwen3.5-0.8B':11}
for model in models:
 asset=meta['assets'][model];info=asset['manifest'];assert info==json.loads((jobs/'assets'/model/'manifest.json').read_text());cases=info['cases'];matrices+=len(cases)
 assert proof['omitted']['models/'+model+'/model.safetensors']['sha256']==info['checkpoint_sha256']
 for case in cases:
  name='logs/decoder-codec-gpu-20261013/jobs/'+model+'-assets/'+case['name']+'/raw.bin';assert proof['omitted'][name]['sha256']==case['sha256']['raw.bin'] and proof['omitted'][name]['bytes']==case['rows']*case['inner']*2
 for mode in ('numeric','timing'):
  stage=next(s for s in stages if s['name']==model+'-'+mode);assert stage['command'][1:]==[asset['directory'],model,mode]
  records=[json.loads(line) for line in (jobs/(model+'-'+mode+'.log')).read_text().splitlines() if line.startswith('{')];packing=[v for v in records if v['kind']=='packing'];assert len(packing)==len(cases) and sum(v['selected'] for v in packing)==selected_counts[model]
  assert {v['matrix'] for v in packing}=={c['name'] for c in cases} and all(v['compact_blocks']<=v['blocks'] and (v['compact_blocks']>0)==v['selected'] for v in packing)
  complete=records[-1];assert complete==dict(kind='complete',mode=mode,numeric_checks=8 if mode=='numeric' else 0,timing_samples=800 if mode=='timing' else 0,full_goal_met=False)
  if mode=='numeric':
   controls=[v for v in records if v['kind']=='controls'];assert controls==[dict(kind='controls',profiles=21,passed=True)];safety+=21
   checks=[v for v in records if v['kind']=='check'];assert len(checks)==8 and {(v['batch'],v['pattern']) for v in checks}=={(b,p) for b in (1,2) for p in range(4)}
   for v in checks:
    assert v['rows']==v['batch'] and v['policy']==1 and v['projections']==len(cases) and v['selected_projections']==selected_counts[model] and v['values']==v['batch']*sum(c['rows'] for c in cases)
    assert v['passed'] and v['finite'] and v['bitwise_equal'] and all(math.isfinite(v[k]) and 0<=v[k]<=1 for k in ('control_fp64_worst','fp64_worst','comparison_worst'));words+=v['values']
   checks_total+=len(checks)
  else:
   samples=[v for v in records if v['kind']=='timing'];assert len(samples)==800;seen=set()
   for v in samples:
    assert v['batch'] in (1,2) and v['rows']==v['batch'] and isinstance(v['null'],bool) and v['round'] in range(10) and v['position'] in range(4) and v['sample'] in range(5)
    assert v['candidate']==(v['position'] in (1,2)) and v['actual_policy']==int(v['candidate'] and not v['null']) and all(math.isfinite(v[k]) and v[k]>0 for k in ('gpu_ms','wall_ms'));key=(v['batch'],v['null'],v['round'],v['position'],v['sample']);assert key not in seen;seen.add(key);timed.append(dict(model=model,**v))
assert checks_total==24 and safety==63 and matrices==29 and len(timed)==2400
result=dict(verified_files=len(proof['files']),trial=trial,regenerated_sources=4,source_files=len(sources),first_layer_matrices=matrices,selected_matrices=sum(selected_counts.values()),numeric_profiles=24,original_output_values_checked=words,safety_profiles=safety,timing_profiles=6,timing_samples=len(timed),all_models_numeric_before_timing=True,original_bits_and_bounds_pass=True,scope='Original-bit/FP64/safety and raw clock/source audit for isolated first-layer projections only; no complete inference speed or new vLLM ratio.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
