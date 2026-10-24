"""Describe raw pauses without excluding requests or inferring their cause."""
from pathlib import Path
import argparse,hashlib,json,zipfile

def observe(archive):
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None;e=json.loads(z.read('evidence-manifest.json'))
  for n,h in e['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
  names=[]
  for n in z.namelist():
   if n.endswith('-manifest.json'):
    d=json.loads(z.read(n))
    if isinstance(d,dict) and d.get('status')=='completed' and 'cohorts' in d:names.append((n,d))
  assert len(names)==1;n,m=names[0];base=n.rsplit('/',1)[0]+'/';samples=[json.loads(l) for l in z.read(base+m['run_id']+'-vram.jsonl').decode().splitlines()];cohorts={}
  for cohort in m['cohorts']:
   rows=[];phase=cohort['case']+'-c'+str(cohort['concurrency'])
   for key in cohort['request_keys']:
    d=json.loads(z.read(base+key+'-result.json'));assert d['submitted']<=d['first_token']<=d['last_token']<=d['completed'];rows.append({'key':key,'submitted':d['submitted'],'first_token':d['first_token'],'last_token':d['last_token'],'completed':d['completed'],'elapsed_s':d['completed']-d['submitted'],'ttft_ms':1000*(d['first_token']-d['submitted']),'decode_s':d['last_token']-d['first_token'],'tail_s':d['completed']-d['last_token']})
   rows.sort(key=lambda v:v['submitted']);gaps=[{'after':a['key'],'before':b['key'],'gap_s':b['submitted']-a['completed']} for a,b in zip(rows,rows[1:])]
   sg=sorted([{'left':a['monotonic'],'right':b['monotonic'],'gap_s':b['monotonic']-a['monotonic']} for a,b in zip(samples,samples[1:]) if a['phase']==b['phase']==phase],key=lambda v:v['gap_s'],reverse=True)[:5]
   cohorts[phase]={'requests':rows,'between_request_gaps':gaps,'largest_sample_gaps':sg}
 return {'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'cohorts':cohorts,'scope':'Raw request/sampling pauses; no requests excluded, no baseline/optimization attribution. Cause unproven; phase-boundary gaps excluded only from within-phase sampler-gap description.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args();assert not a.output.exists();a.output.write_text(json.dumps(observe(a.archive),indent=2));print('RAW_PAUSE_DESCRIPTION_SAVED_WITHOUT_EXCLUSIONS')
