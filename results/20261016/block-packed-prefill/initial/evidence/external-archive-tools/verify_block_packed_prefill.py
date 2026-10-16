"""Audit immutable prefill source regeneration, all numeric failures and admitted clocks."""
from pathlib import Path
import hashlib,json,math,re,runpy,shlex,statistics,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text());assert len(proof['jobs'])==1;trial,status=next(iter(proof['jobs'].items()));assert status['exit_code']==0 and not proof['full_goal_met'];folder=root/'logs'/trial;jobs=folder/'jobs'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256'],name
assert json.loads((jobs/'exit.json').read_text())['exit_code']==0;meta=json.loads((jobs/'metadata.json').read_text());sources={}
for absolute,sha in meta['source_sha256'].items():
 name=Path(absolute).name;found=[p for p in (jobs/'source-snapshot').rglob(name) if p.is_file() and digest(p)==sha];assert len(found)==1 and name not in sources,name;sources[name]=found[0]
factory=runpy.run_path(str(sources['make_block_packed_prefill.py']))
expected={'metal_context.mm':factory['context'](sources['metal_context.mm'].read_text()),'metal_context.h':factory['header'](sources['metal_context.h'].read_text()),'block_packed_prefill_shader.h':factory['shader'](),'block_packed_prefill_gpu.mm':factory['harness'](sources['packed_body_prefill_gpu.mm'].read_text())};assert set(expected)==set(meta['generated_sha256'])
for name,text in expected.items():assert (jobs/'generated'/name).read_bytes()==text.encode() and digest(jobs/'generated'/name)==meta['generated_sha256'][name]
flags=sources['flags.make'].read_text();match=re.search(r'^OBJCXX_DEFINES = (.*)$',flags,re.M) or re.search(r'^CXX_DEFINES = (.*)$',flags,re.M);assert match
assert [v for v in shlex.split(match.group(1)) if v.startswith('-DGARNET_METAL_')]==meta['defines'] and '-DGARNET_METAL_COALESCED_BF16_BODY=1' in meta['defines'] and '-DGARNET_METAL_MLP_PREFILL_GRAPH=1' not in meta['defines']
assert meta['reference_library_sha256']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
assert meta['numeric_design']==dict(rows=[[24,44],[26,52],[80,160]],patterns=4,policies=[1,2],tile_geometry=[[32,64,32],[64,64,32]],threads=256,fp32_activation_and_accumulation=True,all_models_before_any_timing=True,original_fp64_and_comparison_bounds=True)
assert meta['admission']==dict(each_policy_all_six_numeric_profiles_required=True,all_profiles_gpu_and_wall_fraction_max=.95,all_null_gpu_and_wall_fraction=[.95,1.05],complete_unchanged_confirmation_required=True,no_favorable_third_repeat=True)
assert meta['timing_design']==dict(warmup_ms_per_profile=2000,null_before_comparison=True,rounds=10,order='ABBA',warm_calls_per_window=3,samples_per_window=5,all_samples_retained=True)
models=['Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'];stages=json.loads((jobs/'statuses.json').read_text());assert all(s['exit_code']==0 for s in stages)
assert digest(folder/'launcher-source.py')==json.loads((folder/'command.json').read_text())['launcher_sha256']
assert proof['omitted']['logs/'+trial+'/jobs/block_packed_prefill_gpu']['sha256']==meta['binary_sha256']
assert stages[0]['name']=='compile' and stages[0]['command'][0:2]==['/usr/bin/xcrun','clang++']
checks=[];packing=[];admitted={1:True,2:True};samples=[];profiles=[]
for model in models:
 asset=meta['assets'][model];info=asset['manifest'];assert info==json.loads((jobs/'assets'/model/'manifest.json').read_text());cases=info['cases']
 assert proof['omitted']['models/'+model+'/model.safetensors']['sha256']==info['checkpoint_sha256']
 for case in cases:
  name='logs/decoder-codec-gpu-20261013/jobs/'+model+'-assets/'+case['name']+'/raw.bin';assert proof['omitted'][name]['sha256']==case['sha256']['raw.bin'] and proof['omitted'][name]['bytes']==case['rows']*case['inner']*2
 stage=next(s for s in stages if s['name']==model+'-numeric');assert stage['command'][1:]==[asset['directory'],model,'numeric','0']
 records=[json.loads(line) for line in (jobs/(model+'-numeric.log')).read_text().splitlines() if line.startswith('{')];assert records[-1]==dict(kind='complete',mode='numeric',numeric_checks=16,timing_samples=0,full_goal_met=False)
 packs=[v for v in records if v['kind']=='packing'];assert len(packs)==len(cases) and {v['matrix'] for v in packs}=={c['name'] for c in cases}
 for pack in packs:
  c=next(c for c in cases if c['name']==pack['matrix']);assert pack['additional_bytes']==(((c['rows']+63)//64)*((c['inner']+31)//32)*4096 if pack['selected'] else 0) and math.isfinite(pack['upload_ms']) and pack['upload_ms']>0
  packing.append(dict(model=model,**pack))
 current=[v for v in records if v['kind']=='check'];assert len(current)==16 and {(v['policy'],v['batch'],v['pattern']) for v in current}=={(p,b,t) for p in (1,2) for b in (1,2) for t in range(4)}
 for v in current:
  rows=(80*v['batch'] if model=='Qwen3.5-0.8B' else (24 if v['batch']==1 else 44) if model=='Qwen3-0.6B' else 26*v['batch']);assert v['rows']==rows and v['projections']==len(cases) and v['selected_projections']==sum(p['selected'] for p in packs) and v['values']==rows*sum(c['rows'] for c in cases)
  assert math.isfinite(v['control_fp64_worst']) and 0<=v['control_fp64_worst']<=1 and all(math.isfinite(v[k]) and v[k]>=0 for k in ('fp64_worst','comparison_worst'))
  assert v['passed']==(v['finite'] and v['fp64_worst']<=1 and v['comparison_worst']<=1);admitted[v['policy']]=admitted[v['policy']] and v['passed'];checks.append(dict(model=model,**v))
assert json.loads((jobs/'numeric-admission.json').read_text())['admitted']=={str(k):v for k,v in admitted.items()}
assert [s['name'] for s in stages]==['compile']+[m+'-numeric' for m in models]+[m+'-policy'+str(p)+'-timing' for p in (1,2) if admitted[p] for m in models]
for policy in (1,2):
 if not admitted[policy]:continue
 for model in models:
  name=model+'-policy'+str(policy)+'-timing';stage=next(s for s in stages if s['name']==name);assert stage['command'][1:]==[meta['assets'][model]['directory'],model,'timing',str(policy)]
  records=[json.loads(line) for line in (jobs/(name+'.log')).read_text().splitlines() if line.startswith('{')];assert records[-1]==dict(kind='complete',mode='timing',numeric_checks=0,timing_samples=800,full_goal_met=False)
  clock=[v for v in records if v['kind']=='timing'];assert len(clock)==800;seen=set()
  for v in clock:
   assert v['batch'] in (1,2) and v['policy']==policy and isinstance(v['null'],bool) and v['round'] in range(10) and v['position'] in range(4) and v['sample'] in range(5) and v['candidate']==(v['position'] in (1,2)) and v['actual_policy']==(policy if v['candidate'] and not v['null'] else 0) and all(math.isfinite(v[k]) and v[k]>0 for k in ('gpu_ms','wall_ms'))
   key=(v['batch'],v['null'],v['round'],v['position'],v['sample']);assert key not in seen;seen.add(key);samples.append(dict(model=model,**v))
  for batch in (1,2):
   fractions={}
   for null in (True,False):
    subset=[v for v in clock if v['batch']==batch and v['null']==null];assert len(subset)==200
    fractions['null' if null else 'comparison']={k:statistics.mean(v[k] for v in subset if v['candidate'])/statistics.mean(v[k] for v in subset if not v['candidate']) for k in ('gpu_ms','wall_ms')}
   profiles.append(dict(model=model,batch=batch,policy=policy,fractions=fractions,passed=all(.95<=v<=1.05 for v in fractions['null'].values()) and all(v<=.95 for v in fractions['comparison'].values())))
for s in stages:
 monitor=json.loads((jobs/(s['name']+'-monitor.json')).read_text());assert all(v['guard_passed'] for v in monitor)
failed=[v for v in checks if not v['passed']];policy_decisions={str(p):dict(numeric_admitted=admitted[p],unchanged_confirmation_admitted=admitted[p] and all(v['passed'] for v in profiles if v['policy']==p),native_integration_admitted=False) for p in (1,2)}
result=dict(verified_files=len(proof['files']),trial=trial,regenerated_sources=4,source_files=len(sources),matrices=len(packing),selected_matrices=sum(v['selected'] for v in packing),numeric_profiles=len(checks),numeric_failures=len(failed),fp32_values_compared=sum(v['values'] for v in checks),worst_control_fp64_ratio=max(v['control_fp64_worst'] for v in checks),worst_candidate_fp64_ratio=max(v['fp64_worst'] for v in checks),worst_comparison_ratio=max(v['comparison_worst'] for v in checks),additional_weight_bytes=sum(v['additional_bytes'] for v in packing),weight_upload_ms=sum(v['upload_ms'] for v in packing),policies=policy_decisions,checks=checks,packing=packing,timing_profiles=profiles,timing_samples=len(samples),statistic='All 100 raw samples/policy/clock/null-or-comparison side; arithmetic means, no filtering or null normalization',scope='Private first-layer decoder component only. No vision-tower/whole-model gain or fresh vLLM comparison; original envelope unchanged. Safety/tail graph suites remain required for any native integration.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k not in ('checks','packing','timing_profiles')}))
