"""Whole actual 4B allocation ownership, budget, errors, head generation. No timing claim."""
import sys,json,threading,hashlib,struct,math
from pathlib import Path
s='/Users/shawn/CantorAI/out/research/worldsense-packed-request-residency/source'
sys.path.insert(0,s);sys.path.insert(0,s+'/runtime');sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-packed-request-residency-on')
import garnet,metal_serving,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm
w=Path(__file__).resolve().parent;name='Qwen3.5-4B'
profile=dict(maxInputTokens=384,maxOutputTokens=128,prefillChunk=64,linearChunk=True,imagePixels=65536)
loaded=json.loads(garnet.serve_model('/Users/shawn/CantorAI/out/models/'+name,s+'/xModel/qwen35/vl_4b','/Users/shawn/CantorAI/out/cache/'+name,json.dumps(profile),name))
assert loaded['ready'];m=metal_serving._models[name];graphs=[v['graph']for v in m.decoder.layers];head=m.decoder.head
a=garnet.decoder_session();b=garnet.decoder_session();records=[]
def stats(session):return json.loads(session.packed_request_residency_stats_json)
def clean(session):
 q=stats(session)
 assert not q['scope_open'] and not q['attached'] and not q['requested']
 assert q['active_bytes']==q['global_used_bytes']==q['owner_count']==q['set_allocation_count']==0
 return q
def record(case,**data):
 records.append(dict(case=case,**data));(w/'packed-full-session-progress.json').write_text(json.dumps(records,indent=2))
def rejects_begin(session,value):
 rejected=False
 try:session.begin_packed_request_residency(value)
 except Exception:rejected=True
 assert rejected
try:
 assert m._packed_request_scope and all(bool(g.wide_primary_eligible)for g in graphs+[head])
 descriptors=[dict(object=g,tag='layer.'+str(i))for i,g in enumerate(graphs)]+[dict(object=head,tag='head')]
 inv=json.loads(garnet.memory_inventory_json(descriptors,None));(w/'packed-full-graph-inventory.json').write_text(json.dumps(inv,indent=2))
 refs=[v for v in inv['references']if v['representation']in ['primary_bf16_payload','primary_bf16_metadata']]
 ids={v['buffer_id']for v in refs};buffers={v['id']:v for v in inv['buffers']}
 payload={v['buffer_id']for v in refs if v['representation']=='primary_bf16_payload'};metadata={v['buffer_id']for v in refs if v['representation']=='primary_bf16_metadata'}
 assert len(payload)==len(metadata)==201 and len(ids)==402 and not(payload&metadata)
 actual_bytes=sum(buffers[i]['allocated_bytes']for i in ids);logical_bytes=sum(buffers[i]['length_bytes']for i in ids)
 assert logical_bytes==6402163712 and logical_bytes<=actual_bytes<=8589934592
 a.build(graphs,head);b.build(graphs,head)
 for value in [[],[graphs[0]]*32,None]:rejects_begin(a,value);clean(a)
 assert a.begin_packed_request_residency(graphs);q=stats(a)
 assert q['active_bytes']==q['last_bytes']==q['attempted_bytes']==actual_bytes
 assert q['last_count']==q['attempted_count']==q['set_allocation_count']==402 and q['owner_count']==201 and q['attached'] and q['requested'] and q['scope_open']
 rejects_begin(a,graphs);assert stats(a)==q
 assert not b.begin_packed_request_residency(graphs);qb=stats(b);assert qb['reason']==5 and qb['active_bytes']==0 and qb['global_used_bytes']==actual_bytes
 b.end_packed_request_residency();assert stats(a)==q
 record('whole_actual_201owners_402allocations_nested_and_global_budget',active=q,contender=qb,actual_bytes=actual_bytes,logical_bytes=logical_bytes)
 # Same runtime in a second thread: Session locks and the global budget remain authoritative.
 thread_records=[];errors=[]
 def contender():
  try:
   admitted=b.begin_packed_request_residency(graphs);row=stats(b);assert not admitted and row['reason']==5 and row['active_bytes']==0 and row['global_used_bytes']==actual_bytes
   b.end_packed_request_residency();thread_records.append(row)
  except Exception as e:errors.append(str(e))
 t=threading.Thread(target=contender);t.start();t.join();assert not errors and len(thread_records)==1 and stats(a)==q
 record('cross_thread_same_runtime_budget_contention',stats=thread_records[0])
 a.close();assert b.begin_packed_request_residency(graphs);assert stats(b)['active_bytes']==actual_bytes
 b.reset();clean(b);b.close();record('closed_owner_releases_global_budget_readmission_and_reset',clean=stats(b))
 a.build(graphs,head);assert a.begin_packed_request_residency(graphs);a.build(graphs,head);clean(a);record('session_rebuild_releases_lease',clean=stats(a))
 # Serving begins its scope before native image/input work; both error paths must release it.
 for bad_image in [str(w/'missing-image.jpg'),'/Users/shawn/CantorAI/out/models/Qwen3.5-4B/config.json']:
  before=stats(m.decoder.session);bad=json.loads(garnet.infer_json('Describe the visible scene.',bad_image,1,name));assert bad['status']=='error'
  after=clean(m.decoder.session);assert after['calls']==before['calls']+1 and after['admitted']==before['admitted']+1 and after['ends']==before['ends']+1 and after['fallback']==0
  record('native_image_error_outer_finally_cleanup',image=bad_image,error=bad,after=after)
 recovered=json.loads(garnet.infer_json('Describe the visible subject in one sentence.',str(w.parent/'skill-expansion/car.jpg'),16,name));assert recovered['status']=='ok';clean(m.decoder.session)
 recovered_states=[]
 for layer in range(32):
  parts=[]
  for value in m.decoder.session.snapshot(layer):
   values=value.tolist();assert all(math.isfinite(v)for v in values);h=hashlib.sha256()
   for i in range(0,len(values),4096):block=values[i:i+4096];h.update(struct.pack('<'+'f'*len(block),*block))
   parts.append(dict(values=len(values),sha256=h.hexdigest()))
  recovered_states.append(parts)
 reference=json.loads((w.parent/'worldsense-prefix-cache/state-guard.json').read_text())['records'][1]
 assert recovered_states==reference['states'] and recovered['generated_token_ids']==reference['generated_token_ids']
 record('same_model_after_image_errors_original32layers_states_tokens_exact',call=recovered,states=recovered_states)
 # Rebuilding the very same head object invalidates an old session generation.
 symbol=tensor.input('ws69_rebuilt_head',shape=[1,m.text_config['hidden_size']],dtype=tensor.float32)
 @T.fusion(name='ws69_exact_rebuilt_tied_head',boundary='required')
 def forward(hidden):return norm(hidden,'model.language_model.norm',m.text_config)*T.unary_op('qwen35_tied_head',bf16_weight=1)
 head.close();head.build(forward(symbol),[symbol],m.checkpoint);rejects_begin(a,graphs);clean(a)
 a.build(graphs,head);assert a.begin_packed_request_residency(graphs);assert stats(a)['last_count']==402;a.end_packed_request_residency();clean(a)
 head.close();rejects_begin(a,graphs);clean(a);record('rebuilt_head_requires_rebuild_and_closed_head_rejected',clean=stats(a))
 (w/'packed-full-session-guard.json').write_text(json.dumps(dict(all_cases_pass=True,records=records,actual_owner_count=201,actual_allocation_count=402,actual_allocation_bytes=actual_bytes,logical_packed_bytes=logical_bytes,scope='Actual declared allocations, ownership and API errors only; no physical residency or speed claim.'),indent=2)+'\n')
 print('Whole4B actual201owners/402allocations, global8GiB/thread contention, close/reset/rebuild/head/image-error cleanup PASS',flush=True)
finally:
 a.close();b.close();m.decoder.close()
 if m.vision is not None:m.vision.close()
 del metal_serving._models[name]
