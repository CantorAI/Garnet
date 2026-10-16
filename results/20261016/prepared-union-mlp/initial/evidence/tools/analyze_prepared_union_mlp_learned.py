"""Predeclared mean component fractions, own-null gates and model endpoint selection."""
from pathlib import Path
from statistics import mean
import json,sys
base=Path(sys.argv[1]).resolve();audit=json.loads((base/'verification.json').read_text());assert audit['all_layer_profiles']==152 and audit['timing_samples']==4800
proof=json.loads((base/'evidence/manifest.json').read_text());trial=next(iter(proof['jobs']));jobs=base/'evidence/logs'/trial/'jobs'
records=[json.loads(line) for line in (jobs/'timing.log').read_text().splitlines()];windows=[v for v in records if v['kind']=='timing'];profiles=[]
for key in sorted({(v['model'],v['layer'],v['rows']) for v in windows}):
 record=dict(model=key[0],layer=key[1],rows=key[2],means={},fractions={})
 for null in (True,False):
  group=[v for v in windows if (v['model'],v['layer'],v['rows'])==key and v['null']==null];label='null' if null else 'comparison';record['means'][label]={};record['fractions'][label]={}
  for field in ('encode_ms','wall_ms','gpu_ms'):
   values={b:[x for v in group if v['label_b']==b for x in v[field]] for b in (False,True)};assert len(values[False])==len(values[True])==100
   a,b=mean(values[False]),mean(values[True]);record['means'][label][field]=dict(a=a,b=b);record['fractions'][label][field]=b/a
 record['null_pass']=all(.95<=record['fractions']['null'][f]<=1.05 for f in ('wall_ms','gpu_ms'))
 record['gain_pass']=all(record['fractions']['comparison'][f]<=.95 for f in ('wall_ms','gpu_ms'))
 record['profile_pass']=record['null_pass'] and record['gain_pass'];profiles.append(record)
models=[]
for model in sorted({v['model'] for v in profiles}):
 selected=[v for v in profiles if v['model']==model];assert len(selected)==4;models.append(dict(model=model,endpoint_profiles=4,component_screen_pass=all(v['profile_pass'] for v in selected)))
result=dict(profiles=profiles,models=models,unchanged_confirmation_admitted=any(v['component_screen_pass'] for v in models),native_integration_admitted=False,fresh_vllm_comparison=False,full_goal_met=False,scope='Arithmetic means of all ABBA samples; null is an exclusion gate, never a correction; learned component only.')
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(dict(models=models,unchanged_confirmation_admitted=result['unchanged_confirmation_admitted'],full_goal_met=False)))
