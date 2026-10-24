"""Untimed full-token greedy equivalence before real MTP streaming/timing."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from urllib.request import Request,urlopen
from benchmark_stream import payload,request
from native_http_gates import validate_saved_vlm
from run_logged import source_identity

def post(url,body,out,key):
 wire=json.dumps(body,separators=(',',':')).encode();(out/(key+'-request.json')).write_bytes(wire)
 with urlopen(Request(url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=300) as response:raw=response.read()
 (out/(key+'-response.json')).write_bytes(raw);return json.loads(raw)

def compare_greedy_pair(key,teacher,trial):
 left=teacher['choices'][0];right=trial['choices'][0];metrics=trial.get('study',{}).get('speculative_metrics')
 row={'key':key,'prompt_ids_match':left['prompt_token_ids']==right['prompt_token_ids'],'generated_ids_match':left['response_token_ids']==right['response_token_ids'],
  'teacher_mode':teacher.get('study',{}).get('generation_mode'),'trial_mode':trial.get('study',{}).get('generation_mode'),'actual_generated_tokens':len(right['response_token_ids']),'untimed_speculative_metrics':metrics}
 row['passed']=bool(right['response_token_ids']) and row['prompt_ids_match'] and row['generated_ids_match'] and row['teacher_mode']=='greedy_target' and row['trial_mode']=='verified_mtp' and isinstance(metrics,dict)
 return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.study;out=a.output;out.mkdir(exist_ok=True)
 url='http://127.0.0.1:18082';deadline=time.monotonic()+1800
 while True:
  try:
   with urlopen(url+'/health',timeout=5) as response:
    if response.status==200:health=json.load(response);break
  except Exception as error:
   if time.monotonic()>=deadline:raise TimeoutError('MTP readiness deadline; inspect live worker before restart') from error
  time.sleep(5)
 if health.get('native_worker',{}).get('mtp_depth') not in (1,3,7):raise ValueError('Actual MTP worker readiness required')
 (out/'health.json').write_text(json.dumps(health,indent=2));colors=validate_saved_vlm(root,out,url);rows=[]
 for case_index,case in enumerate(['text','image']):
  for i in range(8):
   key='s3-'+case+'-c1-r'+str(i);body,_=payload(case,key,300000+case_index*20000+1000+i,'qwen38-27b-nvfp4',128,'jpeg','image-first','binary-grid');body['stream']=False
   teacher=post(url,dict(body,study_target_only=True),out,key+'-greedy')
   trial=post(url,body,out,key+'-mtp');row=compare_greedy_pair(key,teacher,trial)
   rows.append(row);(out/'greedy-equivalence.json').write_text(json.dumps(rows,indent=2));print('MTP_GREEDY_SEQUENCE_CHECK',key,row['passed'],flush=True)
   if not row['passed']:raise ValueError('MTP greedy equivalence failed: '+key)
 for case,prompt_count in [('text',41),('image',104)]:
  body,_=payload(case,'s3-'+case+'-c1-r0',301000+(20000 if case=='image' else 0),'qwen38-27b-nvfp4',128,'jpeg','image-first','binary-grid')
  result=request(url,body,out,'mtp-stream-gate-'+case,128)
  if result['status']!='completed' or result['prompt_tokens']!=prompt_count:raise ValueError('Actual MTP stream/count gate failed')
 print('MTP_SIXTEEN_GREEDY_AND_REAL_STREAM_GATES_PASSED',flush=True)
 files=[root/'build/bin/garnet.dll',root/'build/bin/xlang3.exe',root/'serve-native-worker.bat',root/'models/Qwen3.8-27B-NVFP4/config.json']
 metadata={'source':source_identity(),'health':health,'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'color_gate':colors,'greedy_equivalence':rows,
  'scope':'Sixteen untimed self-greedy sequence checks plus four retained VLM checks only; not broad quality or full reference precision equivalence'}
 path=out/'metadata.json';path.write_text(json.dumps(metadata,indent=2))
 subprocess.run([sys.executable,str(Path(__file__).with_name('benchmark_stream.py')),'--url',url,'--engine','garnet','--output',str(out/'benchmark'),'--metadata',str(path),
  '--concurrency','1','--requests','8','--warmups','2','--max-tokens','128','--workload-seed','3','--image-format','jpeg','--image-order','image-first','--image-nonce-pattern','binary-grid'],check=True)
if __name__=='__main__':main()
