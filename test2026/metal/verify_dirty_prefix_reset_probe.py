"""Independent source/coverage/positive-zero/timing audit of a reset component."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text())
for name,info in proof['files'].items():
 p=root/name;assert p.resolve().is_relative_to(root);raw=p.read_bytes();assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256']
assert proof['selected_libraries']=={'build-resident-kv-inplace':'73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2','build-attention-prefix':'f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088','build-global-half-alpha':'887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'}
job=root/'evidence'/proof['job'];assert json.loads((job/'exit.json').read_text())['exit_code']==0
j=job/'jobs';meta=json.loads((j/'metadata.json').read_text());profiles=json.loads((j/'profiles.json').read_text());summary=json.loads((j/'summary.json').read_text())
assert meta['profiles']==profiles and meta['binary_sha256']==proof['binary']['sha256']
for name,sha in meta['source_sha256'].items():assert hashlib.sha256((j/'source-snapshot'/name).read_bytes()).hexdigest()==sha
assert {p['model'] for p in profiles}=={'Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B','synthetic'} and len(profiles)==25
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 c=meta['configs'][model]['config'];c=c.get('text_config',c)
 for batch in (1,2):
  for dirty in (0,1,64,256):
   p=next(p for p in profiles if p['model']==model and p['batch']==batch and p['dirty_tokens']==dirty);assert p['capacity']==256
   expected=[]
   for kind in c.get('layer_types',['full_attention']*c['num_hidden_layers']):
    if kind=='linear_attention':
     channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim']
     for col in (channels*c['linear_conv_kernel_dim'],c['linear_num_value_heads']*c['linear_key_head_dim']*c['linear_value_head_dim']):expected.append(dict(columns=col,groups=1,group_columns=col,dirty_columns=col))
    else:
     dim=c.get('head_dim',c['hidden_size']//c['num_attention_heads']);groups=c['num_key_value_heads']
     expected.extend([dict(columns=groups*256*dim,groups=groups,group_columns=256*dim,dirty_columns=dirty*dim)]*2)
   assert p['states']==expected
rows=[json.loads(line) for line in (j/'gpu.log').read_text().splitlines() if line.startswith('{')];checks=[r for r in rows if r['kind']=='check'];timings=[r for r in rows if r['kind']=='timing'];results=[]
assert len(checks)==300 and len(timings)==300 and summary['checks']==300 and summary['timed_resets']==3600
for p in profiles:
 name=p['name'];warm=next(r for r in rows if r['kind']=='warmup' and r['profile']==name);assert warm['elapsed_ms']>=250 and warm['executions']>=3
 pc=[r for r in checks if r['profile']==name];assert {(r['policy'],r['generation']) for r in pc}=={(policy,v) for policy in (0,1,2) for v in range(4)}
 assert all(r['all_cpu_gpu_positive_zero'] and r['padding_preserved'] for r in pc)
 pt=[r for r in timings if r['profile']==name];assert len(pt)==12
 logical_bytes=p['batch']*sum(s['columns'] for s in p['states'])*4;dirty_bytes=p['batch']*sum(s['groups']*s['dirty_columns'] for s in p['states'])*4
 for round in (0,1):
  windows=sorted((r for r in pt if r['round']==round),key=lambda r:r['window']);assert [r['window'] for r in windows]==list(range(6)) and [r['policy'] for r in windows]==[0,1,2,2,1,0]
 for r in pt:assert len(r['wall_ms'])==12 and all(v>=0 for v in r['wall_ms']) and r['logical_bytes']==logical_bytes and r['dirty_bytes']==dirty_bytes and r['all_cpu_gpu_positive_zero'] and r['padding_preserved']
 means={str(policy):statistics.mean(v for r in pt if r['policy']==policy for v in r['wall_ms']) for policy in (0,1,2)}
 claimed=next(r for r in summary['profiles'] if r['profile']==name)
 assert means==claimed['means_ms'] and claimed['prefix_over_legacy']==means['2']/means['0'] and claimed['prefix_over_full_direct']==means['2']/means['1']
 results.append(claimed)
statuses=json.loads((j/'statuses.json').read_text());assert [v['name'] for v in statuses]==['compile','gpu'] and all(v['exit_code']==0 for v in statuses)
for name in ('compile','gpu'):
 monitor=json.loads((j/(name+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
producer_observed=all(r.get('observed_gpu_dirty_pattern',False) for r in checks)
result=dict(verified_files=len(proof['files']),component_checks=300,timed_resets=3600,all_gpu_dirty_patterns_observed=producer_observed,scope='Synthetic bounded GPU writes and reset only; no native inference/TTFT/TPS gain proven',profiles=results,full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
