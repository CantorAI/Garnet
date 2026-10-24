"""First-use TensorRT layer diagnostics; requires qualified profiler lifecycle fix.

A retained profiler may add timing events to later diagnostic graph captures.
Never use this process for official serving benchmarks.
"""
import base64,hashlib,json,math,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_shared_compressed_decoder,create_cuda_vision,prepare_cuda_image,create_cuda_mtp_session
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.run_logged import source_identity
root,out=map(Path,sys.argv[1:3]);out.mkdir(exist_ok=True);path=out/'report.json'
if any(out.iterdir()):raise FileExistsError('Preserve prior layer profile attempt')
profile=out/'layers.tsv'
assert Path(os.environ.get('GARNET_PROFILE_DECODE_LAYERS',''))==profile,'Requires inherited native profiler flag'
checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_text())
diagnostic=root/'results/shared-mtp-sequence-diagnostic';keys=['red','green','blue','bottom-blue']
report={'status':'constructing','source':source_identity(),'cases':[],'stage_traces':[],
 'scope':'One first-use eager enqueue per engine with inherited TensorRT IProfiler; shared T4/T128 MTP3 head reuse OFF. Cold/first-use diagnostic, CUDA Graph bypassed on sampled calls. Not warm captured execution, isolated device-active time, HTTP or throughput.'}
decoder=vision=session=None;trace_decode_active=False
def save():path.write_text(json.dumps(report,indent=2))
save()
try:
 decoder=create_cuda_shared_compressed_decoder(str(checkpoint),config,4096,retain_linear_inputs=True)
 decoder.enable_cuda_graph(True);io=decoder.io
 vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
 tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
 # Keep verified-head reuse disabled, matching the faster measured pilot.
 decoder.reuse_verified_head=False
 session=create_cuda_mtp_session(decoder,str(checkpoint),config,stops,3,draft_chunk_size=4)
 assert not session.verifier.reuse_verified_head and session.verifier.native_prefix is None
 assert not decoder.register_delta_128 and not decoder.deferred_state_commit
 def execute(key,reuse):
  global trace_decode_active
  trace_decode_active=False
  request=diagnostic/(key+'-target-request.json');teacher=diagnostic/(key+'-target-response.json')
  body=json.loads(request.read_text());expected=json.loads(teacher.read_text())['choices'][0]
  session.reset();session.verifier.reuse_verified_head=reuse;ids=expected['prompt_token_ids'];visual=None;start=-1
  content=body['messages'][0]['content']
  if isinstance(content,list):
   encoded=next(item['image_url']['url'] for item in content if item['type']=='image_url')
   image=out/'input.jpg';image.write_bytes(base64.b64decode(encoded.split(',',1)[1],validate=True))
   prepared=prepare_cuda_image(io,str(checkpoint),str(image));assert [prepared[1],prepared[2]]==[16,16]
   visual=vision.encode_patches(prepared[0]);start=ids.index(config['image_token_id']);assert ids.count(config['image_token_id'])==64
   positions,current=image_positions(len(ids),start,64,8)
  else:positions=[[i]*3 for i in range(len(ids))];current=len(ids)
  embeddings=io.embedding(ids)
  hidden=[io.visual_token(visual,i-start) if visual is not None and start<=i<start+64 else io.row(embeddings,i) for i in range(len(ids))]
  session.prefill(hidden,[positions],[len(ids)],[current])
  trace_decode_active=True
  try:result=session.generate(body['max_tokens'])
  finally:trace_decode_active=False
  actual=list(result['generated_token_ids'][0])
  if result['stop_token_ids'][0] is not None:actual.append(result['stop_token_ids'][0])
  assert actual==expected['response_token_ids'],'Retained ordinary target mismatch: '+key
  text=tokenizer.decode([token for token in actual if token not in stops])
  if key in ('red','green','blue','bottom-blue'):assert text.strip().lower().rstrip('.')==('blue' if key=='bottom-blue' else key)
  else:assert len(actual)==128
  return {'key':key,'reuse_verified_head':reuse,'generated_ids':actual,'text':text,
   'request_sha256':hashlib.sha256(request.read_bytes()).hexdigest(),'teacher_sha256':hashlib.sha256(teacher.read_bytes()).hexdigest(),
   'metrics':result['speculative_metrics']}
 report['status']='profiling';save()
 selected=['s3-text-c1-r0','s3-image-c1-r0','red','green','blue','bottom-blue']
 for repeat in range(2):
  for key in selected:
   row=execute(key,False);row['repeat']=repeat;report['cases'].append(row);save()
   print('TRT_PROFILE_EXACT',repeat,key,flush=True)
 assert profile.exists(),'Inherited native profiler emitted no TSV'
 rows=[]
 for line in profile.read_text().splitlines():
  fields=line.split('\t')
  if fields==['engine','layer','milliseconds']:continue
  assert len(fields)==3
  value=float(fields[2]);assert math.isfinite(value) and value>=0
  rows.append({'engine':fields[0],'layer':fields[1],'milliseconds':value})
 assert rows
 report['layers']=rows;report['decoder_final']=decoder.memory_info();report['session_final']=session.memory_info()
 report['vision_graph']=json.loads(vision.graph.stats_json())
 assert all(graph['graph_capture_failures']==0 for graph in report['decoder_final']['graphs'])
 assert report['vision_graph']['graph_capture_failures']==0
 report['library_sha256']=hashlib.sha256((root/'build/bin/garnet.dll').read_bytes()).hexdigest()
 report['status']='passed';print('NATIVE_FULL_TRT_LAYER_PROFILE_PASS; diagnostic only',flush=True)
except Exception as error:report['status']='failed';report['error']=repr(error);raise
finally:
 save()
 if session is not None:session.close()
 if vision is not None:vision.close()
 if decoder is not None:decoder.close()
