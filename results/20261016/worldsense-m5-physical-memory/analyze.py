"""Diagnostic validity and memory categories only. No speed acceptance calculation."""
from pathlib import Path
import json,hashlib,re,collections,csv
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-physical-memory/source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(s/k)==v for k,v in json.loads((w/'starting-source-manifest.json').read_text()).items())
gold=json.loads((r/'Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json').read_text());refs={}
for item in gold['items']:
 if item['policy']=='candidate':
  for phase,c in item['result']['model_response'].items():
   if isinstance(c,dict)and c.get('status')=='ok':refs[(item['observation_id'],phase)]=(item,c)
env=json.loads((w/'environment.json').read_text());manif={m['label']:m for m in json.loads((w/'cohort-manifest.json').read_text())};observations=[];rows=[];calls=[]
for label,enabled in [('control-1',False),('candidate-1',True)]:
 d=json.loads((w/(label+'.json')).read_text());assert d['valid']and d['assignments_restored']and len(d['items'])==2 and not d['search_plans'];assert d['request']==json.loads((w/'cases.json').read_text())
 assert [(i['repetition'],i['warmup'],i['policy'])for i in d['items']]==[(0,True,'default'),(1,False,'default')]
 assert manif[label]['client_exit']==0 and manif[label]['memory_observer_records']==4 and manif[label]['library_sha256']==env['libraries']['on'if enabled else'off'] and manif[label]['serving_sha256']==sha(s/'runtime/metal_serving.py')
 for item in d['items']:
  for phase,c in item['result']['model_response'].items():
   if not isinstance(c,dict)or c.get('status')!='ok':continue
   ri,ref=refs[(item['observation_id'],phase)];stable=lambda x:{k:v for k,v in x.items()if k not in ['model_response','model_response_json']};assert stable(item['result'])==stable(ri['result'])
   assert all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])and c['timings']['input_tokens']==ref['timings']['input_tokens'] and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256']
   t=c['timings'];assert ('cached_matrix_metadata'in c['backend'])==enabled and (t['lazy_default_retired_bytes']>0)==enabled and t['lazy_default_rehydrated_bytes']==0 and Path(t['serving_source'])==s/'runtime/metal_serving.py'
   key=t['os_memory_observation']['capture_id'];folder=w/'observations'/label/key;o=json.loads((folder/'observation.json').read_text());inv=json.loads((folder/'native-inventory.json').read_text());assert o['native_boundary']['model_id']==c['model_id']and o['native_boundary']['capture_id']==key and o['native_boundary']['inventory']['sha256']==sha(folder/'native-inventory.json')==t['memory_inventory']['sha256']
   assert sorted(o['native_boundary']['loaded_model_ids'])==sorted(set(next(name for name in o['native_boundary']['loaded_model_ids'] if tag['tag'].startswith(name+'.'))for tag in inv['references']))
   native={b['id']:b for b in inv['buffers']};group=collections.defaultdict(list)
   for refrow in inv['references']:group[refrow['buffer_id']].append(refrow)
   assert set(group)==set(native)and sum(b['allocated_bytes']for b in native.values())==inv['unique_allocated_bytes']
   categories=collections.Counter()
   for bid,rr in group.items():
    if any(x['weight']for x in rr):kind='weights'
    elif any(x['role'].startswith('private_state')for x in rr):kind='session_private_states'
    elif any('prefix'in x['role']for x in rr):kind='prefix_cache'
    elif any(x['input']for x in rr):kind='graph_default_inputs'
    elif any(x['legacy_output']for x in rr):kind='graph_legacy_packed_outputs'
    elif any(x['output_part']for x in rr):kind='graph_output_parts'
    else:kind='other_graph_session_allocations'
    categories[kind]+=native[bid]['allocated_bytes']
   footprint=(folder/'footprint.stdout.txt').read_text();category_rows=[]
   pattern=r'^\s*(-?\d+) B\s+(-?\d+) B\s+(-?\d+) B\s+(-?\d+) B\s+(-?\d+) B\s+(\d+)\s+(.+)$'
   for line in footprint.splitlines():
    m=re.match(pattern,line)
    if m:
     values=list(map(int,m.groups()[:6]));category_rows.append(dict(zip(['dirty_bytes','swapped_subset_bytes','clean_bytes','reclaimable_bytes','wired_subset_bytes','regions'],values),category=m.group(7)))
   total=next(x for x in category_rows if x['category']=='TOTAL');non_total=[x for x in category_rows if x['category']!='TOTAL'];assert sum(x['dirty_bytes']for x in non_total)==total['dirty_bytes'];assert all(x['swapped_subset_bytes']<=x['dirty_bytes']for x in non_total)
   aux=re.search(r'^\s*phys_footprint: (\d+) B',footprint,re.M);assert aux
   for name,tool in o['tools'].items():assert tool['returncode']==0 and tool['stdout_sha256']==sha(folder/(name+'.stdout.txt')) and tool['stderr_sha256']==sha(folder/(name+'.stderr.txt'))
   observation=dict(worker=label,model=c['model_id'],warmup=item['warmup'],capture_id=key,loaded_model_ids=o['native_boundary']['loaded_model_ids'],logical_native_unique_bytes=inv['unique_allocated_bytes'],logical_native_categories=dict(categories),native_task_before=o['native_boundary']['native_before'],os_category_total=total,os_physical_footprint_bytes=int(aux.group(1)),os_categories=sorted(non_total,key=lambda x:x['dirty_bytes'],reverse=True),scope='OS Dirty includes Swapped subset; never add them. Native logical buffers and OS categories overlap, not additive. Diagnostic pauses and allocations prohibit speed or memory-improvement claims.');observations.append(observation)
   calls.append(dict(worker=label,phase=phase,model=c['model_id'],capture_id=key,exact=True))
   for cr in non_total:rows.append(dict(worker=label,model=c['model_id'],warmup=item['warmup'],capture_id=key,**cr))
assert len(calls)==8 and len({x['capture_id']for x in calls})==8
(w/'memory-analysis.json').write_text(json.dumps({'status':'valid_diagnostic_not_a_performance_comparison','calls':calls,'observations':observations,'scope':'All8realWorldSense calls exact/golden/native/identity valid. Global graph/session union in one native collection; no exact per-buffer physical attribution or causal inference.'},indent=2))
with(w/'os-categories.csv').open('w',newline='')as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({'status':'valid_diagnostic','calls':len(calls),'captures':len(observations),'no_performance_claim':True}))
