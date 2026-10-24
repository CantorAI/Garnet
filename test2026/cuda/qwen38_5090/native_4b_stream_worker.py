"""Research4B B1 worker using existing Model and synchronized token callbacks.

Correctness and real HTTP timing must be qualified before any comparison claim.
"""
import base64,hashlib,json,os,sys,time
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
from runtime.cuda_serving import Model
from test2026.cuda.qwen38_5090.native_worker_io import requests,emit
from test2026.cuda.qwen38_5090.native_http_bridge import validate_body
checkpoint,image_path=sys.argv[1:3]
profile={'batch':1,'maxInputTokens':3968,'maxOutputTokens':128,'imagePixels':65536,'imageMinPixels':65536,'imageGrid':[16,16],
 'prefillChunk':16,'decodeChunk':16,'fusedDelta':True,'fusedAttention':True,'cudaGraph':True}
model=None
try:
 model=Model(checkpoint,profile,'Qwen3.5-4B')
 emit({'event':'ready','research_only':True,'model':'qwen35-4b-bf16','batch':1,'capacity':4096,'chunk':16,
  'precision':'BF16 weights/projections; FP32 residuals/state','quality_gate':'pending','memory':model.memory_info()})
 for item in requests():
  key=item['id'];body=item['body'];started=time.monotonic()
  try:
   validate_body(body,'qwen35-4b-bf16');limit=body['max_tokens']
   if limit>128:raise ValueError('Loaded4B profile supports at most128 outputs')
   content=body['messages'][0]['content'];path='';image_hash=None
   if isinstance(content,str):prompt=content
   else:
    data=base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True)
    if data[:2]!=b'\xff\xd8':raise ValueError('JPEG signature required')
    image_hash=hashlib.sha256(data).hexdigest()
    with open(image_path,'wb') as f:f.write(data)
    prompt=content[1]['text'];path=image_path
   generated=[];state={'text':''}
   def notify(row,token,count,prompt_count):
    assert row==0 and count==len(generated)+1
    now=time.monotonic();generated.append(token)
    state['text']=model.tokenizer.decode([t for t in generated if t not in model.stops])
    emit({'event':'token','id':key,'token_id':token,'sampled_tokens':count,'prompt_tokens':prompt_count,
     'text':state['text'],'stop':token in model.stops,'native_monotonic_seconds':now})
   result=model.infer(prompt,path,limit,on_token=notify)
   expected=list(result['generated_token_ids'][0])
   if result['stop_token_ids'][0] is not None:expected.append(result['stop_token_ids'][0])
   assert expected==generated and result['sampled_tokens_including_stop']==[len(generated)]
   assert result['text']==state['text']
   emit({'event':'completed','id':key,'generated_ids':generated,'prompt_ids':result['prompt_token_ids'][0],
    'text':result['text'],'image_sha256':image_hash,'generation_mode':'greedy_target','speculative_metrics':None,
    'finish_reason':'stop' if result['finish_reasons']==['eos'] else 'length',
    'native_elapsed_seconds':time.monotonic()-started})
  except Exception as error:
   emit({'event':'error','id':key,'error':repr(error)});raise
finally:
 if model is not None:model.close()
