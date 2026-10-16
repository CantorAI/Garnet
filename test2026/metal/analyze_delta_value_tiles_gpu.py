"""Fixed all-endpoint component screen: raw-sample arithmetic means, no null normalization or filtering."""
from pathlib import Path
import json,statistics,sys
base=Path(sys.argv[1]).resolve();audit=json.loads((base/'verification.json').read_text());assert audit['timing_samples']==6400 and audit['all_original_bits_equal'];jobs=base/'evidence/logs'/audit['trial']/'jobs'
records=[json.loads(line) for line in (jobs/'timing.log').read_text().splitlines()];records=[r for r in records if r['kind']=='timing'];cases=json.loads((jobs/'assets/manifest.json').read_text())['cases'];profiles=[]
for case in cases:
 if not case['name'].startswith('real-layer'):continue
 for width in (32,64):
  values=[r for r in records if r['name']==case['name'] and r['width']==width];assert len(values)==800//2;fractions={}
  for null,label in ((True,'null'),(False,'comparison')):
   selected=[r for r in values if r['null']==null];assert len(selected)==200
   fractions[label]={field:statistics.mean(r[field] for r in selected if r['candidate'])/statistics.mean(r[field] for r in selected if not r['candidate']) for field in ('gpu_ms','wall_ms')}
  stable=all(.95<=v<=1.05 for v in fractions['null'].values());passed=stable and all(v<=.95 for v in fractions['comparison'].values())
  profiles.append(dict(name=case['name'],layer=int(case['name'].split('-')[1].removeprefix('layer')),batch=case['batch'],chunk=case['chunk'],width=width,fractions=fractions,own_null_stable=stable,component_screen_pass=passed))
workloads=[]
for width in (32,64):
 for chunk,kind in ((1,'decode'),(80,'prefill')):
  selected=[p for p in profiles if p['width']==width and p['chunk']==chunk];assert len(selected)==4
  workloads.append(dict(width=width,kind=kind,component_screen_pass=all(p['component_screen_pass'] for p in selected)))
result=dict(profiles=profiles,workloads=workloads,unchanged_confirmation_admitted=any(w['component_screen_pass'] for w in workloads),statistic='Ratio of arithmetic means of all 100 candidate and all 100 control raw samples per comparison/null endpoint; no filtering or normalization.',native_integration_admitted=False,full_goal_met=False,scope='Initial component screen only; all four first/last layer x B1/B2 endpoints must have >=5% GPU and wall gain with stable own-nulls. Complete unchanged confirmation and native/full-model gates remain required.')
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
