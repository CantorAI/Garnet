"""Audit exact sources/arrays, original head checks, unsafe guards and complete null/clock grid."""
from pathlib import Path
import hashlib,json,math,re,runpy,shlex,statistics,struct,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text());assert len(proof['jobs'])==1;trial,status=next(iter(proof['jobs'].items()));assert status['exit_code']==0 and not proof['full_goal_met'];folder=root/'logs'/trial;jobs=folder/'jobs'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256'],name
assert json.loads((jobs/'exit.json').read_text())['exit_code']==0;meta=json.loads((jobs/'metadata.json').read_text());sources={}
for absolute,sha in meta['source_sha256'].items():
 name=Path(absolute).name;found=[p for p in (jobs/'source-snapshot').rglob(name) if p.is_file() and digest(p)==sha];assert len(found)==1 and name not in sources,name;sources[name]=found[0]
sys.path.insert(0,str(sources['make_nibble_screen_index.py'].parent));factory=runpy.run_path(str(sources['make_sixbit_screen_index.py']))
expected={'metal_context.mm':factory['context'](sources['metal_context.mm'].read_text()),'metal_context.h':factory['header'](sources['metal_context.h'].read_text()),'screened_head_index.h':factory['index'](sources['screened_head_index.h'].read_text()),'screened_head_native_impl.h':factory['native_impl'](sources['screened_head_native_impl.h'].read_text()),'nibble_screened_head_shader.h':factory['shader'](sources['screened_head_shader.h'].read_text()),'nibble_screen_index_gpu.mm':sources['sixbit_screen_index_gpu.mm'].read_text()};assert set(expected)==set(meta['generated_sha256'])
for name,text in expected.items():assert (jobs/'generated'/name).read_bytes()==text.encode() and digest(jobs/'generated'/name)==meta['generated_sha256'][name]
flags=sources['flags.make'].read_text();match=re.search(r'^OBJCXX_DEFINES = (.*)$',flags,re.M) or re.search(r'^CXX_DEFINES = (.*)$',flags,re.M);assert match
assert [v for v in shlex.split(match.group(1)) if v.startswith('-DGARNET_METAL_')]==meta['defines'] and all(v in meta['defines'] for v in ('-DGARNET_METAL_SCREENED_HEAD=1','-DGARNET_METAL_INTERLEAVED_SCREEN_INDEX=1','-DGARNET_METAL_SMALL_SCREEN_SEED=1'))
assert meta['reference_library_sha256']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
assert meta['numeric_design']==dict(models_before_any_timing=True,captured_queries_per_model=17,policies=[0,1],seed_rows=1024,control_index_bits=8,candidate_index_bits=6,all_retained_original_bits=True,all_rejected_original_scores_below_seed=True,original_query_guards_and_error_allowances=True,unsafe_patterns=['zero','tiny','large','NaN','Inf','mixed_safe_unsafe'],pending_foreign_abort_recovery_uninitialized=True)
assert meta['timing_design']==dict(conditioning_ms_per_model_batch=2000,ordinary_steps_per_batch=8,null_before_comparison_per_query=True,rounds=10,order='ABBA',warm_calls_per_window=3,samples_per_window=5,all_samples_retained=True,expected_samples=19200)
assert meta['admission']==dict(unmet_b1_models=['Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'],each_unmet_gpu_and_wall_fraction_max=.95,all_profiles_gpu_and_wall_fraction_max=1.02,all_null_gpu_and_wall_fraction=[.95,1.05],complete_unchanged_confirmation_required=True,no_favorable_third_repeat=True)
models=['Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'];stages=json.loads((jobs/'statuses.json').read_text());assert [s['name'] for s in stages]==['compile']+[m+'-numeric' for m in models]+[m+'-timing' for m in models] and all(s['exit_code']==0 for s in stages)
assert digest(folder/'launcher-source.py')==json.loads((folder/'command.json').read_text())['launcher_sha256'] and proof['omitted']['logs/'+trial+'/jobs/nibble_screen_index_gpu']['sha256']==meta['binary_sha256']
assert stages[0]['command'][0:2]==['/usr/bin/xcrun','clang++']
raw_checks=checks=fallbacks=ownership=0;score_values=0;timing_count=0;profiles=[];packing=[];pruning=[];peak_samples={}
for model in models:
 asset=meta['assets'][model];info=asset['manifest'];assert info==json.loads((jobs/'assets'/model/'manifest.json').read_text());n,k=info['rows'],info['inner'];queries=info['queries'];assert len(queries)==17
 assert proof['omitted']['models/'+model+'/model.safetensors']['sha256']==info['checkpoint_sha256'] and proof['omitted'][(asset['directory'].split('/garnet-metal-investigation/')[1]+'/raw.bin')]['sha256']==info['raw_sha256']
 for query in queries:
  for suffix,key,count in (('.input.bin','input_sha256',query['batch']*k),('.reference.bin','reference_sha256',query['batch']*n)):
   p=root/'captured-arrays'/model/(query['name']+suffix);assert digest(p)==query[key] and p.stat().st_size==count*4
 for mode in ('numeric','timing'):
  stage=next(s for s in stages if s['name']==model+'-'+mode);assert stage['command'][1:]==[asset['directory'],model,mode]
  records=[json.loads(line) for line in (jobs/(model+'-'+mode+'.log')).read_text().splitlines() if line.startswith('{')];assert records[-1]==dict(kind='complete',mode=mode,checks=34 if mode=='numeric' else 0,fallback_checks=22 if mode=='numeric' else 0,ownership_checks=34 if mode=='numeric' else 0,timing_samples=6400 if mode=='timing' else 0,full_goal_met=False)
  packs=[v for v in records if v['kind']=='packing'];assert len(packs)==2 and {v['policy'] for v in packs}=={0,1}
  for v in packs:assert v['rows']==n and v['inner']==k and v['index_bytes']==((n+3)//4*4)*k*(3 if v['policy'] else 4)//4 and math.isfinite(v['upload_and_index_ms']) and v['upload_and_index_ms']>0
  if mode=='numeric':packing.extend(dict(model=model,**v) for v in packs)
  monitors=json.loads((jobs/(model+'-'+mode+'-monitor.json')).read_text());assert all(v['guard_passed'] for v in monitors);rss=[v['rss_bytes'] for v in monitors if v['rss_bytes'] is not None];peak_samples[model+'-'+mode]=max(rss) if rss else None
  if mode=='numeric':
   original=[v for v in records if v['kind']=='raw_check'];assert len(original)==17 and {v['query'] for v in original}=={q['name'] for q in queries}
   for v in original:assert v['bits_exact'] and v['values']==v['batch']*n
   raw_checks+=17;current=[v for v in records if v['kind']=='check'];unsafe=[v for v in records if v['kind']=='fallback_check'];assert len(current)==34 and len(unsafe)==22
   assert {(v['query'],v['policy']) for v in current}=={(q['name'],p) for q in queries for p in (0,1)}
   assert {(v['batch'],v['pattern'],v['policy']) for v in unsafe}=={(b,t,p) for b,patterns in ((1,5),(2,6)) for t in range(patterns) for p in (0,1)}
   for v in current+unsafe:
    assert v['kept']>=1024*v['batch'] and v['kept']+v['rejected']==n*v['batch'] and all(v[x] for x in ('all_original_scores_checked','retained_bits_exact','rejected_below_exact_seed','greedy_and_error_flags_match'));score_values+=n*v['batch']
   pruning.extend(dict(model=model,**v) for v in current);checks+=34;fallbacks+=22;ownership+=34
  else:
   clock=[v for v in records if v['kind']=='timing'];assert len(clock)==6400;seen=set()
   ordinary={q['name']:q for q in queries if q['name'].startswith(('b1-step','b2-step'))};assert len(ordinary)==16
   for v in clock:
    assert v['query'] in ordinary and v['batch']==ordinary[v['query']]['batch'] and isinstance(v['null'],bool) and v['round'] in range(10) and v['position'] in range(4) and v['sample'] in range(5) and v['candidate']==(v['position'] in (1,2)) and v['actual_policy']==int(v['candidate'] and not v['null']) and all(math.isfinite(v[k]) and v[k]>0 for k in ('gpu_ms','wall_ms'))
    key=(v['query'],v['null'],v['round'],v['position'],v['sample']);assert key not in seen;seen.add(key)
   timing_count+=len(clock)
   for batch in (1,2):
    means={};fractions={}
    for null in (True,False):
     subset=[v for v in clock if v['batch']==batch and v['null']==null];assert len(subset)==1600 and len({v['query'] for v in subset})==8
     phase='null' if null else 'comparison';means[phase]={name:{key:statistics.mean(v[key] for v in subset if v['candidate']==candidate) for key in ('gpu_ms','wall_ms')} for name,candidate in (('control',False),('candidate',True))};fractions[phase]={key:means[phase]['candidate'][key]/means[phase]['control'][key] for key in ('gpu_ms','wall_ms')}
    stable=all(.95<=v<=1.05 for v in fractions['null'].values());preserved=all(v<=1.02 for v in fractions['comparison'].values());gain=all(v<=.95 for v in fractions['comparison'].values()) if batch==1 and model in meta['admission']['unmet_b1_models'] else None
    profiles.append(dict(model=model,batch=batch,means=means,fractions=fractions,own_null_stable=stable,preservation_pass=preserved,unmet_b1_gain=gain))
assert (raw_checks,checks,fallbacks,ownership,timing_count)==(51,102,66,102,19200)
admit=all(v['own_null_stable'] and v['preservation_pass'] and v['unmet_b1_gain'] is not False for v in profiles)
result=dict(verified_files=len(proof['files']),trial=trial,regenerated_sources=6,source_files=len(sources),original_full_captured_queries=raw_checks,numeric_policy_queries=checks,unsafe_policy_queries=fallbacks,ownership_profiles=ownership,original_score_values_checked=score_values,timing_samples=timing_count,all_models_numeric_before_timing=True,packing=packing,pruning=pruning,profiles=profiles,sampled_process_rss_bytes=peak_samples,unchanged_confirmation_admitted=admit,native_integration_admitted=False,statistic='Per-batch arithmetic means of all 800 intervals/arm/phase over eight ordinary captured steps; no filtering or null normalization',scope='Captured-query greedy head component only; index6/8 is rejection storage, final seed/refinement original BF16/FP32. No inference throughput/TTFT, universal-bound certification or fresh vLLM result.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','pruning','packing','sampled_process_rss_bytes')}))
