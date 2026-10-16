"""Audit complete learned original-bit coverage and every declared alternating timing window."""
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
factory=runpy.run_path(str(snapshots['make_paired_norm_rope.py']))
for name,operation in [('metal_context.h','header'),('metal_context.mm','context')]:
 p=jobs/'generated'/name;assert p.read_bytes()==factory[operation](snapshots[name].read_text()).encode() and digest(p)==meta['generated_sha256'][name]
assert meta['reference_library_sha256']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
report_path=jobs/'learned-report.json';assert digest(report_path)==meta['learned_report_sha256'];report=json.loads(report_path.read_text());expected={};endpoints=set();learned_norms=0
assert meta['timing_design']==dict(common_warmup_ms=2000,rounds=10,window_order='ABBA',samples_per_window=5,warm_calls_per_window=3,null_before_comparison=True,all_samples_retained=True,timed_layers={'Qwen3-0.6B':[0,27],'Qwen2.5-Coder-0.5B-Instruct':[0,23],'Qwen3.5-0.8B':[3,23]},endpoint_selection='First/last separate full-attention layer; all four decode/prefill row profiles, fixed before execution')
for model in report['models']:
 name=model['model'];c=model['rope'];assert meta['assets'][name]['checkpoint_sha256']==model['checkpoint_sha256'];raw=meta['assets'][name]['config'];config=raw.get('text_config',raw)
 assert c['query_heads']==config['num_attention_heads'] and c['key_heads']==config['num_key_value_heads'] and c['dim']==config.get('head_dim',config['hidden_size']//config['num_attention_heads'])
 layers=[g['layer'] for g in model['groups']];assert [min(layers),max(layers)]==meta['timing_design']['timed_layers'][name]
 for g in model['groups']:
  assert len(g['norms'])==2
  for item in g['norms']:
   assert item['ones']==(name=='Qwen2.5-Coder-0.5B-Instruct')
   if not item['ones']:assert item['dtype'] in ('BF16','F32') and item['bytes']==c['dim']*(2 if item['dtype']=='BF16' else 4) and len(item['raw_sha256'])==64;learned_norms+=1
  for rows in [1,2,*c['prefill_rows']]:
   key=(name,g['layer'],rows);assert key not in expected;expected[key]=c
   if g['layer'] in (min(layers),max(layers)):endpoints.add(key)
assert len(expected)==232 and len(endpoints)==24 and learned_norms==68
stages=json.loads((jobs/'statuses.json').read_text());assert [v['name'] for v in stages]==['compile','numeric','timing'] and all(v['exit_code']==0 for v in stages)
for stage in stages:
 monitor=json.loads((jobs/(stage['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
assert proof['omitted']['logs/'+trial+'/jobs/paired_norm_rope_learned_probe']['sha256']==meta['binary_sha256']
def key(v):return (v['model'],v['layer'],v['rows'])
def finite(v):
 if isinstance(v,float):assert math.isfinite(v)
 elif isinstance(v,dict):
  for x in v.values():finite(x)
 elif isinstance(v,list):
  for x in v:finite(x)
summary={};all_records={}
for mode in ('numeric','timing'):
 records=[json.loads(v) for v in (jobs/(mode+'.log')).read_text().splitlines()];finite(records);all_records[mode]=records;wanted=set(expected) if mode=='numeric' else endpoints;numeric=[v for v in records if v['kind']=='numeric'];assert len(numeric)==len(wanted) and {key(v) for v in numeric}==wanted
 for v in numeric:
  c=expected[key(v)];assert v['variants']==4 and v['original_bits_equal'] and all(v[a]==c[b] for a,b in [('query_heads','query_heads'),('key_heads','key_heads'),('dim','dim'),('rotary','rotary'),('normalize','normalize'),('weight_offset','weight_offset')])
  assert v['original_output_bits_checked']==v['rows']*(c['query_heads']+c['key_heads'])*c['dim']*4
  for field in ('input_sha256','output_sha256'):assert len(v[field])==4 and all(len(pair)==2 and all(len(h)==64 for h in pair) for pair in v[field])
 done=[v for v in records if v['kind']=='profile_complete'];assert len(done)==len(wanted) and {key(v) for v in done}==wanted and all(v['original_bits_equal'] for v in done)
 complete=records[-1];assert complete['kind']=='complete' and complete['mode']==mode and complete['profiles']==len(wanted) and complete['changed_input_cases']==len(wanted)*4 and complete['timed_profiles']==(24 if mode=='timing' else 0) and complete['original_output_bits_checked']==sum(v['original_output_bits_checked'] for v in numeric);summary[mode]=complete
windows=[v for v in all_records['timing'] if v['kind']=='timing'];assert len(windows)==1920
warm=[v for v in all_records['timing'] if v['kind']=='warmup'];assert len(warm)==48 and all(v['wall_ms']>=2000 for v in warm)
for profile in endpoints:
 for null in (True,False):
  group=[v for v in windows if key(v)==profile and v['null']==null];assert len(group)==40 and {(v['round'],v['window']) for v in group}=={(r,w) for r in range(10) for w in range(4)}
  for v in group:
   assert v['label_b']==(v['window'] in (1,2)) and v['candidate']==(v['label_b'] and not null) and v['complete_output_bits_checked_after_window']
   for field in ('encode_ms','wall_ms','gpu_ms'):assert len(v[field])==5 and all(x>0 for x in v[field])
result=dict(verified_files=len(proof['files']),source_context_regenerated=True,complete_learned_profiles=232,changed_input_cases=928,learned_norm_matrices=68,coder_ones_matrices=48,timed_endpoint_profiles=24,timing_windows=1920,timing_samples=9600,summary=summary,native_integration_admitted=False,scope='All learned component bit/source/clock coverage only. Kernel-only waits and host fixtures are not complete-model inference; no native/matched-vLLM speed proof.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
