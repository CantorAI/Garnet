"""Pinned INT4 B1 HTTP qualification before a matched short-context pilot."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
from urllib.request import Request,urlopen
from benchmark_stream import payload,request
from run_logged import source_identity

GOLDEN_SHA='2607a4183044309698c2f5852578b6d6e2825187730e577670ec2efefca6f775'
MODEL='qwen38-27b-int4'

def main():
 root,out=map(Path,sys.argv[1:3]);out.mkdir(exist_ok=False)
 url='http://127.0.0.1:18082';deadline=time.monotonic()+1800
 while True:
  try:
   with urlopen(url+'/health',timeout=5) as response:health=json.load(response)
  except Exception:
   if time.monotonic()>deadline:raise TimeoutError('Inspect original worker handle; do not restart on this deadline')
   time.sleep(5);continue
  break
 worker=health['native_worker']
 assert (worker['model'],worker['batch'],worker['capacity'],worker['chunk'],worker['prefill_chunk'],worker['mtp_depth'])==(MODEL,1,4096,16,16,0)
 assert sum(g.get('native_int4_g128_projection_weights',0) for g in worker['memory']['graphs'])==496
 golden_path=root/'results/int4-packed-qualification/report.json';raw=golden_path.read_bytes()
 assert hashlib.sha256(raw).hexdigest()==GOLDEN_SHA
 golden=json.loads(raw);assert golden['status']=='smoke_passed' and len(golden['cases'])==12
 rows={row['key']:row for row in golden['cases'] if row['repeat']==0};records={}
 for key,expected in rows.items():
  original=root/'results/native-mtp3-http'/(key+'-request.json');raw=original.read_bytes()
  assert hashlib.sha256(raw).hexdigest()==expected['request_sha256']
  body=json.loads(raw);body.pop('study_target_only',None);body['model']=MODEL;body['stream']=False
  for repeat in range(2):
   stem=key+'-'+str(repeat);(out/(stem+'-request.json')).write_text(json.dumps(body,indent=2))
   with urlopen(Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=600) as response:result=json.load(response)
   (out/(stem+'-response.json')).write_text(json.dumps(result,indent=2))
   choice=result['choices'][0]
   assert choice['prompt_token_ids']==expected['prompt_ids'] and choice['response_token_ids']==expected['generated_ids'],key
   assert result['study']['generation_mode']=='greedy_target' and result['study']['speculative_metrics'] is None
   assert choice['message']['content']==expected['text']
   records[key]=result
 for index,case in enumerate(('text','image')):
  key='s3-'+case+'-c1-r0';body,_=payload(case,key,301000+index*20000,MODEL,128,'jpeg','image-first','binary-grid')
  stem='stream-gate-'+case;result=request(url,body,out,stem,128)
  assert result['status']=='completed' and result['generated_tokens']==128 and result['prompt_tokens']==(41 if case=='text' else 104)
  events=[json.loads(line)['data'] for line in (out/(stem+'-events.jsonl')).read_text().splitlines()]
  final=next(json.loads(value) for value in reversed(events) if value!='[DONE]')
  assert final['response_token_ids']==records[key+'-greedy']['choices'][0]['response_token_ids']
 files=[golden_path,root/'build/bin/garnet.dll',root/'build/bin/xlang3.exe',root/'models/Qwen3.8-27B-INT4/config.json']
 metadata={'source':source_identity(),'health':health,'golden_sha256':GOLDEN_SHA,'exact_http_golden_cases':12,'stream_gates':2,
  'file_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
  'scope':'Pinned INT4 B1/T16/cap4096 short-input pilot. Exact retained full IDs, prompt IDs, text and real SSE gates. Different quantization from mixed NVFP4; no cross-precision quality equivalence, native batching, broad quality, full-context or 2x claim.'}
 path=out/'metadata.json';path.write_text(json.dumps(metadata,indent=2));print('INT4_HTTP_EXACT_GOLDENS_AND_SSE_PASS',flush=True)
 subprocess.run([sys.executable,str(root/'Garnet/test2026/cuda/qwen38_5090/benchmark_stream.py'),'--url',url,'--engine','garnet','--model',MODEL,'--output',str(out/'benchmark'),'--metadata',str(path),'--concurrency','1','--requests','8','--warmups','2','--max-tokens','128','--workload-seed','3','--image-format','jpeg','--image-order','image-first','--image-nonce-pattern','binary-grid'],check=True)
 (out/'completed.json').write_text(json.dumps({'status':'completed','scope':metadata['scope']}))

if __name__=='__main__':main()
