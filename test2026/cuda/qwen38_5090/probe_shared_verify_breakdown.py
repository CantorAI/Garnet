"""Nested full-model target-verification breakdown; diagnostic wall tracing only."""
import base64,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_shared_compressed_decoder,create_cuda_vision,prepare_cuda_image,create_cuda_mtp_session
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.run_logged import source_identity
root,out=map(Path,sys.argv[1:3]);out.mkdir(exist_ok=True);path=out/'report.json'
if path.exists():raise FileExistsError('Preserve prior head reuse attempt')
checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_text())
diagnostic=root/'results/shared-mtp-sequence-diagnostic';keys=['red','green','blue','bottom-blue']+['s3-'+case+'-c1-r'+str(i) for case in ('text','image') for i in range(8)]
report={'status':'constructing','source':source_identity(),'cases':[],'stage_traces':[],
 'scope':'Full shared T4/T128 MTP3, head reuse off/on: twenty retained ordinary-target pairs, then nested instrumented native-stream API wall traces. Do not sum parent/child intervals. Not HTTP, isolated GPU kernels, speedup or2x qualification.'}
decoder=vision=session=None;trace_decode_active=False
def save():path.write_text(json.dumps(report,indent=2))
save()
try:
 decoder=create_cuda_shared_compressed_decoder(str(checkpoint),config,4096,retain_linear_inputs=True)
 decoder.enable_cuda_graph(True);io=decoder.io
 vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
 tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
 # Explicit opt-in is set before verifier construction, using graph replay,
 # without native prefix commit or register128 arithmetic changes.
 decoder.reuse_verified_head=True
 session=create_cuda_mtp_session(decoder,str(checkpoint),config,stops,3,draft_chunk_size=4)
 assert session.verifier.reuse_verified_head and session.verifier.native_prefix is None
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
 report['status']='checking';save()
 for key in keys:
  reference=execute(key,False);candidate=execute(key,True)
  assert reference['generated_ids']==candidate['generated_ids']
  a,b=reference['metrics'],candidate['metrics']
  for field in ('target_verification_calls','target_prefix_replays','accepted_draft_tokens','proposed_draft_tokens','mtp_draft_calls','mtp_correction_calls'):
   assert a[field]==b[field],'Execution work changed beyond head reuse: '+field
  assert b['target_head_evaluations']+b['target_head_reuses']==a['target_head_evaluations']
  if key not in ('red','green','blue','bottom-blue'):assert b['target_head_reuses']>0
  report['cases'].extend([reference,candidate]);save();print('SHARED_HEAD_REUSE_EXACT',key,flush=True)
 report['status']='correctness_passed_tracing';save()
 # Existing bounded native readback synchronizes the same execution stream.
 # A single-word retained view keeps diagnostic copy overhead explicit/small.
 barrier_tensor=io.storage_view(decoder.last_hidden,[1,1])
 def barrier():
  raw=bytes(io.finite_f32_bytes(barrier_tensor))
  if len(raw)!=4:raise RuntimeError('Diagnostic barrier must read exactly one F32 word')
 report['profiling_barrier']={'operation':'finite_f32_bytes on retained first-word view of decoder.last_hidden','readback_bytes':4,
  'scope':'Native current-execution-stream synchronization through four-byte diagnostic D2H readback before/after each disjoint stage. Wall duration includes CPU, GPU, readback and synchronization overhead; instrumentation alters scheduling. Not isolated CUDA kernel duration, device-wide synchronization or production throughput.'}
 original=[];current_trace=None;stack=[]
 def instrument(owner,name,label):
  method=getattr(owner,name);original.append((owner,name,method))
  def wrapped(*args,**kwargs):
   if not trace_decode_active:return method(*args,**kwargs)
   barrier();start=time.perf_counter();parent=stack[-1] if stack else None;depth=len(stack);stack.append(label)
   try:result=method(*args,**kwargs);barrier()
   finally:stack.pop()
   ended=time.perf_counter()
   current_trace['stages'].append({'stage':label,'parent_stage':parent,'depth':depth,'started':start,'ended':ended,'wall_seconds':ended-start})
   return result
  setattr(owner,name,wrapped)
 class Delegate:
  def __init__(self,value):self.value=value
  def __getattr__(self,name):return getattr(self.value,name)
 original_io=session.verifier.io;original_norm=session.verifier.graph
 session.verifier.io=Delegate(original_io);session.verifier.graph=Delegate(original_norm)
 report['nested_trace_scope']='Parent intervals include child intervals and synchronization overhead; never sum parent and child. Only generate/decode is traced, not prefill or image preprocessing. Native graph wall includes host bindings and GPU work; it is not isolated device time.'
 instrument(session.verifier,'_begin','verify_backup')
 instrument(decoder,'chunk_hidden','target_chunk')
 instrument(decoder.programs[decoder.decode_chunk_size],'run','target_program')
 # Proxy only the Python references after all graph construction. Actual native
 # tensor/graph objects and operations are delegated unchanged.
 instrument(session.verifier.graph,'run_device','verify_normalization')
 instrument(session.verifier.io,'predict_tokens','verify_vocab_head')
 instrument(session.verifier,'_read','verify_read')
 instrument(session,'_draft','draft');instrument(session.verifier,'verify','target_verify')
 instrument(session.verifier,'commit_prefix','prefix_commit');instrument(session,'_correct','draft_correct')
 try:
  # Warmed ABBA order per real input; all outputs must still match teacher.
  for case in ('text','image'):
   for index,reuse in enumerate((False,True,True,False)):
    current_trace={'case':case,'reuse_verified_head':reuse,'order':index,'stages':[]}
    row=execute('s3-'+case+'-c1-r0',reuse);current_trace['metrics']=row['metrics']
    current_trace['generated_ids']=row['generated_ids'];current_trace['teacher_sha256']=row['teacher_sha256'];current_trace['request_sha256']=row['request_sha256']
    report['stage_traces'].append(current_trace);save()
 finally:
  for owner,name,method in original:setattr(owner,name,method)
  session.verifier.io=original_io;session.verifier.graph=original_norm
 report['decoder_final']=decoder.memory_info();report['session_final']=session.memory_info()
 assert all(graph['graph_capture_failures']==0 for graph in report['decoder_final']['graphs'])
 report['status']='passed';print('SHARED_VERIFICATION_NESTED_BREAKDOWN_PASS; no HTTP benchmark',flush=True)
except Exception as error:report['status']='failed';report['error']=repr(error);raise
finally:
 save()
 if session is not None:session.close()
 if vision is not None:vision.close()
 if decoder is not None:decoder.close()
