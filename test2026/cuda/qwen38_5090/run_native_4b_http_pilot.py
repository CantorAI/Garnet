"""Strict native4B HTTP smoke/repeats/streaming before a B1 pilot; no invented lengths."""
from pathlib import Path
import base64,hashlib,json,subprocess,sys,time,zipfile
from urllib.request import urlopen
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p))
from benchmark_stream import payload,request
from validate_native_mtp_http import post
from run_logged import source_identity
out=r/'results/native-4b-http-pilot';out.mkdir(exist_ok=False);url='http://127.0.0.1:18082';model='qwen35-4b-bf16'
archive=r/'results/qwen35-4b-grid16-evidence.zip';assert hashlib.sha256(archive.read_bytes()).hexdigest()=='beb80c6e2aa69997eac5a96b03fa53ca8cd58674b654cb51a49652cceeb03c1b'
with zipfile.ZipFile(archive) as z:old=json.loads(z.read('results/qwen35-4b-grid16-smoke/report.json'))
assert old['checkpoint_revision']=='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a' and old['status']=='smoke_completed'
config=r/'models/Qwen3.5-4B/config.json';assert hashlib.sha256(config.read_bytes()).hexdigest()=='ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670'
deadline=time.monotonic()+1800
while True:
 try:
  with urlopen(url+'/health',timeout=5) as response:health=json.load(response)
 except Exception:
  if time.monotonic()>=deadline:raise TimeoutError('Inspect original4B server handle; never restart on this deadline')
  time.sleep(5);continue
 worker=health['native_worker'];assert worker['model']==model and worker['batch']==1 and worker['capacity']==4096 and worker['chunk']==16
 assert worker['precision']=='BF16 weights/projections; FP32 residuals/state';break
report={'status':'checking','health':health,'old_smoke_cases':[],'semantic_cases':[],'full128_pairs':[],'stream_gates':[],'scope':'4B B1 compatibility/repeat gates; no cross-model/cross-engine equivalence or broad-quality admission.'}
def save():(out/'quality.json').write_text(json.dumps(report,indent=2),encoding='utf8')
def same(left,right):return all(left[f]==right[f] for f in ('prompt_token_ids','response_token_ids'))
save()
try:
 for key in ('text','image'):
  teacher=next(c for c in old['cases'] if c['key']==key);result=teacher['result'];ids=list(result['generated_token_ids'][0])
  if result['stop_token_ids'][0] is not None:ids.append(result['stop_token_ids'][0])
  expected={'prompt_token_ids':result['prompt_token_ids'][0],'response_token_ids':ids};content=teacher['prompt']
  if key=='image':
   image=(r/'qwen35-4b-smoke.jpg').read_bytes();assert hashlib.sha256(image).hexdigest()==teacher['image_sha256']
   content=[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(image).decode()}},{'type':'text','text':content}]
  body={'model':model,'messages':[{'role':'user','content':content}],'max_tokens':16,'temperature':0,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}
  for repeat in range(2):
   response=post(url,body,out,'old-'+key+'-'+str(repeat));choice=response['choices'][0];item={'key':key,'repeat':repeat,'exact':same(choice,expected)};report['old_smoke_cases'].append(item);save();assert item['exact'],'Old4B full IDs changed; preserve failure before timing'
 for key,want in [('red','red'),('green','green'),('blue','blue'),('bottom-blue','blue')]:
  body=json.loads((r/'results/native-mtp3-http'/(key+'-request.json')).read_text(encoding='utf8'));body['model']=model;body['stream']=False;body.pop('stream_options',None)
  response=post(url,body,out,'semantic-'+key);text=response['choices'][0]['message']['content'].strip().lower().rstrip('.');item={'key':key,'semantic_pass':text==want};report['semantic_cases'].append(item);save();assert item['semantic_pass'],'4B VLM semantic gate failed'
 records={}
 for index,case in enumerate(('text','image')):
  for i in range(8):
   key='s3-'+case+'-c1-r'+str(i);body,_=payload(case,key,300000+index*20000+1000+i,model,128,'jpeg','image-first','binary-grid');body['stream']=False;body.pop('stream_options',None)
   captures=[post(url,body,out,key+'-repeat-'+str(repeat))['choices'][0] for repeat in range(2)];records[key]=captures[0]
   item={'key':key,'repeat_exact':same(*captures),'generated':len(captures[0]['response_token_ids']),'prompt':len(captures[0]['prompt_token_ids'])};report['full128_pairs'].append(item);save()
   assert item['repeat_exact'] and item['generated']==128 and item['prompt']==(41 if case=='text' else 104),'4B full128/same-profile repeat gate failed; do not force EOS length'
 for index,case in enumerate(('text','image')):
  key='s3-'+case+'-c1-r0';body,_=payload(case,key,301000+index*20000,model,128,'jpeg','image-first','binary-grid');stem='stream-gate-'+case;result=request(url,body,out,stem,128)
  assert result['status']=='completed' and result['generated_tokens']==128
  events=[json.loads(l)['data'] for l in (out/(stem+'-events.jsonl')).read_text(encoding='utf8').splitlines()];final=next(json.loads(v) for v in reversed(events) if v!='[DONE]');assert same(final,records[key]);report['stream_gates'].append(case);save()
 report['status']='passed';save()
except Exception as error:report['status']='failed';report['error']=repr(error);save();raise
files=[config,archive,r/'build/bin/garnet.dll',r/'build/bin/xlang3.exe',r/'serve-native-4b-http-worker.bat',r/'run-native-4b-http-pilot.py',p/'native_4b_stream_worker.py']
metadata={'source':source_identity(),'health':health,'checkpoint_revision':old['checkpoint_revision'],'quality_gates':{'old_smoke_repeats':4,'vlm_semantics':4,'full128_repeat_pairs':16,'actual_sse_gates':2},'file_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'scope':'4B BF16 target-only B1 short-prompt pilot after retained smoke/repeat/SSE gates. Not quality parity with27B, tuned references, GPU batching or2x admission.'}
path=out/'metadata.json';path.write_text(json.dumps(metadata,indent=2),encoding='utf8');print('NATIVE_4B_HTTP_SMOKE_FULL128_REPEATS_AND_SSE_PASS',flush=True)
subprocess.run([sys.executable,str(p/'benchmark_stream.py'),'--url',url,'--engine','garnet','--model',model,'--output',str(out/'benchmark'),'--metadata',str(path),'--concurrency','1','--requests','8','--warmups','2','--max-tokens','128','--workload-seed','3','--image-format','jpeg','--image-order','image-first','--image-nonce-pattern','binary-grid'],check=True)

