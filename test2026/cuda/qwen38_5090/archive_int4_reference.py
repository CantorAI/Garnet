from pathlib import Path
import argparse,hashlib,json,sys,zipfile
p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('--engine',choices=['vllm','sglang'],required=True);a=p.parse_args();r=a.study;source=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(source))
from compare_http_runs import verify_run
engine=a.engine;names=[engine+'-int4-'+suffix for suffix in ('quality','http-pilot','quality-ledger','http-ledger','server-ledger')]
for label in ('quality','http','server'):
 events=[json.loads(line) for line in (r/('results/'+engine+'-int4-'+label+'-ledger/commands.jsonl')).read_text().splitlines()]
 assert events[-1]['event']=='finished','Wait for original terminal handle'
 if label!='server':assert events[-1]['exit_code']==0
assert json.loads((r/('results/'+engine+'-int4-intentional-stop.json')).read_text())['signal']=='SIGTERM'
pilot=r/('results/'+engine+'-int4-http-pilot');assert json.loads((pilot/'completed.json').read_text())['status']=='completed'
paths=list((pilot/'benchmark').glob('*-manifest.json'));assert len(paths)==1;m=json.loads(paths[0].read_text());replay=verify_run(m,lambda n:(paths[0].parent/n).read_bytes());assert len(replay['cohorts'])==8
quality=r/('results/'+engine+'-int4-quality');greedy=json.loads((quality/'greedy/equivalence.json').read_text());assert greedy['status']=='passed' and len(greedy['cases'])==16
files=[p for name in names for p in (r/'results'/name).rglob('*') if p.is_file()]
files += [r/x for x in ['run-int4-reference-pilot.py','archive-int4-reference.py','stop-'+engine+'-int4.py','results/'+engine+'-int4-intentional-stop.json','results/'+engine+'-int4-server.log','results/int4-reference-source-install.json','study-state.json','preparation-manifest.json','int4-reference-source.zip','models/Qwen3.8-27B-INT4/config.json','models/Qwen3.8-27B-INT4/preprocessor_config.json','models/Qwen3.8-27B-INT4/model.safetensors.index.json']]
files += [source/n for n in ['run_reference_greedy_candidate.py','prepare_vlm_reference.py','reference_mtp_equivalence.py','serve_'+engine+'_int4.sh','benchmark_stream.py','metrics.py','sample_vram.py','compare_http_runs.py']]
raw={p.relative_to(r).as_posix():p.read_bytes() for p in files};out=r/('results/'+engine+'-int4-http-evidence.zip');assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in raw.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps({'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'raw_event_replay':replay,'scope':'Pinned INT4 ordinary reference candidate, four VLM cases and sixteen full128 captures; eight B1/2/4/8 text/image cohorts. Not broad quality, fastest tuning, native batching or2x qualification.'},indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'entries':len(raw)+1,'cohorts':m['cohorts']}))
