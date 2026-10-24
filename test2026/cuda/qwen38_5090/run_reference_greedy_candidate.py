"""Wait for a reference, retain VLM/full greedy evidence; never auto-benchmark."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from urllib.request import urlopen
from run_logged import source_identity


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--study',type=Path,required=True);p.add_argument('--engine',choices=['vllm','sglang'],required=True)
 p.add_argument('--url',required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--launcher',type=Path,required=True);p.add_argument('--server-log',type=Path,required=True)
 p.add_argument('--ordinary-oracle',type=Path)
 a=p.parse_args();out=a.output;scripts=Path(__file__).parent
 if out.exists() and any(out.iterdir()):raise FileExistsError('Preserve prior reference attempt')
 out.mkdir(parents=True,exist_ok=True)
 report={'status':'waiting','source':source_identity(),'engine':a.engine,
  'mode_intent':'speculative_compare' if a.ordinary_oracle else 'ordinary_capture',
  'scope':'Readiness plus four retained VLM cases and16 actual full greedy sequences. Server precision/speculation/graph configuration still requires independent startup-log review; no timing or2x claim.'}
 path=out/'candidate.json'
 try:
  deadline=time.monotonic()+1800
  while True:
   try:
    with urlopen(a.url+'/health',timeout=5) as response:
     if response.status==200:health=response.read();break
   except Exception as error:
    if time.monotonic()>=deadline:raise TimeoutError('Reference readiness observation expired; inspect original server before restart') from error
   time.sleep(5)
  (out/'health.bin').write_bytes(health)
  with urlopen(a.url+'/v1/models',timeout=15) as response:models=response.read()
  (out/'models.json').write_bytes(models)
  if not any(m.get('id')=='qwen38-27b-nvfp4' for m in json.loads(models).get('data',[])):raise ValueError('Actual reference model ID mismatch')
  print('REFERENCE_READY_FOR_UNTIMED_GREEDY',a.engine,flush=True)
  subprocess.run([sys.executable,str(scripts/'prepare_vlm_reference.py'),'--engine',a.engine,'--url',a.url,'--output',str(out/'correctness')],check=True)
  command=[sys.executable,str(scripts/'reference_mtp_equivalence.py'),'--engine',a.engine,'--url',a.url,'--output',str(out/'greedy')]
  if a.ordinary_oracle:command+=['--ordinary-oracle',str(a.ordinary_oracle)]
  subprocess.run(command,check=True)
  snapshot=a.server_log.read_bytes();(out/'server-log-snapshot.log').write_bytes(snapshot)
  files=[a.launcher,a.study/'models/Qwen3.8-27B-NVFP4/config.json',scripts/'reference_mtp_equivalence.py',scripts/'prepare_vlm_reference.py']
  report['files']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
  report['server_log_snapshot_sha256']=hashlib.sha256(snapshot).hexdigest()
  report['status']='quality_capture_completed';print('REFERENCE_VLM_AND_FULL_GREEDY_CAPTURE_COMPLETE',flush=True)
 except Exception as error:
  report['status']='failed';report['error']=repr(error);raise
 finally:
  report['source_after']=source_identity();path.write_text(json.dumps(report,indent=2),encoding='utf8')


if __name__=='__main__':main()
