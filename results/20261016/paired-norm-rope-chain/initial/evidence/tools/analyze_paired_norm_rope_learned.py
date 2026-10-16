"""Describe all fixed endpoint means and own-null variation without inferring native speed."""
from pathlib import Path
from statistics import mean
import json,sys
base=Path(sys.argv[1]).resolve();audit=json.loads((base/'verification.json').read_text());assert audit['complete_learned_profiles']==232 and audit['timing_samples']==9600
proof=json.loads((base/'evidence/manifest.json').read_text());trial=next(iter(proof['jobs']));jobs=base/'evidence/logs'/trial/'jobs';windows=[json.loads(v) for v in (jobs/'timing.log').read_text().splitlines() if json.loads(v)['kind']=='timing'];profiles=[]
for key in sorted({(v['model'],v['layer'],v['rows']) for v in windows}):
 record=dict(model=key[0],layer=key[1],rows=key[2],means={},fractions={})
 for null in (True,False):
  group=[v for v in windows if (v['model'],v['layer'],v['rows'])==key and v['null']==null];label='null' if null else 'comparison';record['means'][label]={};record['fractions'][label]={}
  for field in ('encode_ms','wall_ms','gpu_ms'):
   values={b:[x for v in group if v['label_b']==b for x in v[field]] for b in (False,True)};assert len(values[False])==len(values[True])==100;a,b=mean(values[False]),mean(values[True]);record['means'][label][field]=dict(a=a,b=b);record['fractions'][label][field]=b/a
 record['null_within_five_percent']=all(.95<=record['fractions']['null'][f]<=1.05 for f in ('wall_ms','gpu_ms'));profiles.append(record)
result=dict(profiles=profiles,native_integration_admitted=False,fresh_vllm_comparison=False,full_goal_met=False,scope='Descriptive component results only. GPU kernel/CPU encode savings and single-submit wall times are distinct. Own-nulls are not gain normalization. Native admission requires separate repeated complete-model evidence.')
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(dict(profiles=len(profiles),stable_null_profiles=sum(p['null_within_five_percent'] for p in profiles),native_integration_admitted=False,full_goal_met=False)))
