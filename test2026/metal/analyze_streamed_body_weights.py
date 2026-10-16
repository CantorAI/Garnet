"""Fixed raw-mean component screen for both unmet B1 models and all-profile/null preservation."""
from pathlib import Path
import json,statistics,sys
base=Path(sys.argv[1]).resolve();audit=json.loads((base/'verification.json').read_text());assert audit['timing_samples']==2400 and audit['original_bits_and_bounds_pass'];jobs=base/'evidence/logs'/audit['trial']/'jobs';meta=json.loads((jobs/'metadata.json').read_text());profiles=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 records=[json.loads(line) for line in (jobs/(model+'-timing.log')).read_text().splitlines() if line.startswith('{')];records=[r for r in records if r['kind']=='timing']
 for batch in (1,2):
  rows=[r for r in records if r['batch']==batch];assert len(rows)==400;fractions={};means={}
  for null,label in ((True,'null'),(False,'comparison')):
   selected=[r for r in rows if r['null']==null];assert len(selected)==200
   means[label]={policy:{k:statistics.mean(r[k] for r in selected if r['candidate']==candidate) for k in ('gpu_ms','wall_ms')} for policy,candidate in [('control',False),('candidate',True)]}
   fractions[label]={k:means[label]['candidate'][k]/means[label]['control'][k] for k in ('gpu_ms','wall_ms')}
  profiles.append(dict(model=model,batch=batch,means=means,fractions=fractions,own_null_stable=all(.95<=v<=1.05 for v in fractions['null'].values()),within_preservation=all(v<=1.02 for v in fractions['comparison'].values()),unmet_b1_gain=all(v<=.95 for v in fractions['comparison'].values()) if batch==1 and model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct') else None))
unmet=[p for p in profiles if p['unmet_b1_gain'] is not None];assert len(unmet)==2
admitted=all(p['own_null_stable'] and p['within_preservation'] for p in profiles) and all(p['unmet_b1_gain'] for p in unmet)
result=dict(profiles=profiles,unchanged_confirmation_admitted=admitted,native_integration_admitted=False,all_numeric_and_original_bits_pass=True,statistic='Candidate/control arithmetic means of all 100 raw samples per policy and null/comparison profile; no filtering or null normalization.',full_goal_met=False,scope='Both unmet B1 models require >=5% GPU and wall gain; all six profiles preserve both clocks within2% and own-nulls within5%. One complete unchanged confirmation required before any native/model performance claim.')
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
