"""Audit complete learned QV with original K data, original FP32-key controls, source regeneration and clocks."""
from pathlib import Path
import hashlib,json,math,runpy,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert len(proof['jobs'])==1;trial,status=next(iter(proof['jobs'].items()));assert status['exit_code']==0
assert 'Apple M4' in proof['hardware']['machdep.cpu.brand_string'] and int(proof['hardware']['hw.memsize'])==16*2**30
jobs=root/'logs'/trial/'jobs';meta=json.loads((jobs/'metadata.json').read_text());snapshots={}
for i,(name,sha) in enumerate(meta['source_sha256'].items()):
 p=jobs/'source-snapshot'/str(i)/Path(name).name;assert digest(p)==sha and p.name not in snapshots;snapshots[p.name]=p
factory=runpy.run_path(str(snapshots['make_prepared_union_qv.py']))
for name,operation in [('metal_context.h','header'),('metal_context.mm','context')]:
 p=jobs/'generated'/name;assert p.read_bytes()==factory[operation](snapshots[name].read_text()).encode() and digest(p)==meta['generated_sha256'][name]
assert meta['reference_library_sha256']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
assert digest(snapshots['exact_half_codec.h'])=='e8800f4983bb2786a7e376ca7784a7817176a5e6dd2d46c7e353116f99b2a4c9'
cpu=jobs/'cpu-report.json';assert digest(cpu)==meta['cpu_report_sha256'];report=json.loads(cpu.read_text())
stages=json.loads((jobs/'statuses.json').read_text());assert [v['name'] for v in stages]==['compile','numeric','timing'] and all(v['exit_code']==0 for v in stages)
for stage in stages:
 monitor=json.loads((jobs/(stage['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
assert all(v in stages[0]['command'] for v in meta['defines']) and '-O3' in stages[0]['command'] and '-fobjc-arc' in stages[0]['command']
assert meta['timing_design']==dict(common_warmup_ms=2000,rounds=10,window_order='ABBA',samples_per_window=5,warm_calls_per_window=3,null_before_comparison=True,all_samples_retained=True,timed_layers={'Qwen3-0.6B':[0,27],'Qwen2.5-Coder-0.5B-Instruct':[0,23],'Qwen3.5-0.8B':[3,19]},endpoint_selection='First/last previously admitted triple, frozen before QV execution; original triple scale and eligibility retained')
assert proof['omitted']['logs/'+trial+'/jobs/prepared_union_qv_learned_probe']['sha256']==meta['binary_sha256']
expected={};endpoints=set()
for model in report['models']:
 name=model['model'];first={'Qwen3-0.6B':24,'Qwen2.5-Coder-0.5B-Instruct':26,'Qwen3.5-0.8B':80}[name];second=44 if first==24 else first*2
 assert meta['assets'][name]['checkpoint_sha256']==model['checkpoint_sha256']
 eligible=[g['layer'] for g in model['groups'] if g['union_admitted']];assert [min(eligible),max(eligible)]==meta['timing_design']['timed_layers'][name]
 for group in model['groups']:
  for rows in (first,second):
   key=(name,group['layer'],rows);expected[key]=group
   if group['layer'] in (min(eligible),max(eligible)):endpoints.add(key)
assert len(expected)==116 and len(endpoints)==12
def key(v):return (v['model'],v['layer'],v['rows'])
def finite(v):
 if isinstance(v,float):assert math.isfinite(v)
 elif isinstance(v,dict):
  for value in v.values():finite(value)
 elif isinstance(v,list):
  for value in v:finite(value)
summary={};all_records={}
for mode in ('numeric','timing'):
 records=[json.loads(line) for line in (jobs/(mode+'.log')).read_text().splitlines()];finite(records);all_records[mode]=records;assert records[-1]['kind']=='complete' and records[-1]['mode']==mode and sum(v['kind']=='complete' for v in records)==1
 wanted=set(expected) if mode=='numeric' else endpoints;numeric=[v for v in records if v['kind']=='numeric'];assert len(numeric)==len(wanted) and {key(v) for v in numeric}==wanted
 for v in numeric:
  group=expected[key(v)];n=[s['rows'] for s in group['sides']];total=sum(n);qv=n[0]+n[2]
  assert v['prepared']==group['union_admitted'] and v['variants']==4 and v['passed'] and v['mode']==mode and v['inner']==group['inner'] and [v[f+'_columns'] for f in ('q','k','v')]==n
  assert v['fp32_key']==group['sides'][1]['retained_float_key']
  for field in ('fp64_ratio','comparison_ratio','control_candidate_ratio'):assert 0<=v[field]<=1
  assert v['checked_values']==v['rows']*4*((2*total+qv) if v['prepared'] else total)
  assert v['scatter_bit_checks']==(v['rows']*qv*4 if v['prepared'] else 0) and v['original_key_bit_checks']==(v['rows']*n[1]*4 if v['prepared'] else 0)
  assert v['extra_union_weight_bytes']==((group['logical_extra_union_bytes']-group['sides'][1]['original_bytes']) if v['prepared'] else 0) and v['logical_joined_scratch_bytes']==v['rows']*qv*4
  assert v['prepare_ms']>0 and v['upload_ms']>0 and v['first_control_wall_ms']>0 and (v['first_candidate_wall_ms']>0)==v['prepared']
 done=[v for v in records if v['kind']=='profile_complete'];assert len(done)==len(wanted) and {key(v) for v in done}==wanted and all(v['passed'] for v in done)
 complete=records[-1];assert complete['profiles']==len(wanted) and complete['changed_input_cases']==len(wanted)*4 and complete['fallback_cases']==(16 if mode=='numeric' else 0) and complete['timed_profiles']==(0 if mode=='numeric' else 12) and complete['original_fp32_key_matrices']==(52 if mode=='numeric' else 4)
 assert complete['checked_values']==sum(v['checked_values'] for v in numeric) and complete['scatter_bit_checks']==sum(v['scatter_bit_checks'] for v in numeric)
 for field in ('worst_fp64_ratio','worst_comparison_ratio','worst_control_candidate_ratio'):assert 0<=complete[field]<=1
 summary[mode]=complete
windows=[v for v in all_records['timing'] if v['kind']=='timing'];assert len(windows)==960
warm=[v for v in all_records['timing'] if v['kind']=='warmup'];assert len(warm)==24 and all(v['wall_ms']>=2000 for v in warm)
for profile in endpoints:
 for null in (True,False):
  group=[v for v in windows if key(v)==profile and v['null']==null];assert len(group)==40 and {(v['round'],v['window']) for v in group}=={(r,w) for r in range(10) for w in range(4)}
  for v in group:
   assert v['label_b']==(v['window'] in (1,2)) and v['candidate']==(v['label_b'] and not null) and v['complete_output_checked_after_window']
   for field in ('encode_ms','wall_ms','gpu_ms'):assert len(v[field])==5 and all(x>0 for x in v[field])
result=dict(verified_files=len(proof['files']),source_context_regenerated=True,complete_learned_profiles=116,admitted_profiles=112,required_fallback_profiles=4,original_fp32_key_matrices=52,timed_endpoint_profiles=12,timing_windows=960,timing_samples=4800,summary=summary,scope='All separate learned QV with original K component numerical/source/clock coverage only; not native integration or full-model speed proof.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
