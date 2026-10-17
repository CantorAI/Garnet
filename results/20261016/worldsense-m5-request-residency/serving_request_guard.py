"""Correctness-only serving errors/finally/metric-accounting; no speed evidence."""
import sys,json,time,pathlib
mode=sys.argv[1];w=pathlib.Path(__file__).resolve().parent;source='/Users/shawn/CantorAI/out/research/worldsense-request-residency/'+('fault-source'if mode=='fault'else'source');sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-request-residency-'+mode);sys.path.insert(0,source);sys.path.insert(0,source+'/runtime')
import garnet,metal_serving
name='Qwen3-1.7B';loaded=json.loads(garnet.serve_model('/Users/shawn/CantorAI/out/models/'+name,source+'/xModel/qwen/metal_dense','/Users/shawn/CantorAI/out/cache/'+name,json.dumps(dict(maxInputTokens=192,maxOutputTokens=64,prefillChunk=16,textPrefixCache=True)),name));assert loaded['ready'];model=metal_serving._models[name];enabled='request_weight_residency'in garnet.backend_info();records=[]
def clean(tag):
 r=json.loads(model.decoder.session.weight_residency_stats_json);assert r['active_bytes']==0 and r['request_active']==0
 if enabled:assert r['request_starts']==r['request_ends']
 if mode=='fault':assert all(r[k]==0 for k in ['test_owners','test_set_count','test_attached','test_requested','test_global_used'])
 records.append(dict(tag=tag,stats=r));return r
base='Decide whether the visual facts satisfy the required condition. Use only the observation as evidence. Missing or contradictory evidence means matched=false. Do not substitute a different subject or condition. A still image cannot prove a transition. Return only a compact JSON object with matched and reason. '*2
prompt=base+'Observation: a red car passes a stationary viewpoint. Condition: a passenger vehicle is visible.'
a=json.loads(garnet.infer_json(prompt,'',8,name));assert a['status']=='ok';clean('real_warm')
bad=json.loads(garnet.infer_json(prompt,'/nonexistent/request-residency-test.jpg',8,name));assert bad['status']=='error';clean('image_error')
saved=model.infer
def fail(prompt,image,limit):raise ValueError('Injected serving request exception')
try:
 model.infer=fail;bad=json.loads(garnet.infer_json(prompt,'',8,name));assert bad['status']=='error'and'Injected serving request exception'in bad['error_message'];clean('injected_model_error')
finally:model.infer=saved
def fail_after_decode(prompt,image,limit):
 result=saved(prompt,image,limit)
 before=json.loads(model.decoder.session.weight_residency_stats_json)
 if enabled:
  assert before['request_active']==1
  if mode=='on':assert before['active_bytes']>0 and before['request_reuses']>0
 raise ValueError('Injected error after learned decode')
try:
 model.infer=fail_after_decode;bad=json.loads(garnet.infer_json(prompt,'',8,name));assert bad['status']=='error'and'Injected error after learned decode'in bad['error_message'];clean('error_after_real_decode_with_live_set')
finally:model.infer=saved
# Synthetic serving result only verifies outer timing includes the body and cleanup.
# It is never treated as learned inference or a benchmark result.
if enabled:
 def fake(prompt,image,limit):
  time.sleep(.003)
  return dict(status='ok',ttft_seconds=.001,total_seconds=0,timings={})
 try:
  model.infer=fake;r=json.loads(garnet.infer_json('synthetic metrics correctness only','',1,name));assert r['status']=='ok'and r['total_seconds']>=.003 and r['ttft_seconds']>=.001 and r['total_seconds']>=r['timings']['request_weight_residency_cleanup_seconds'];clean('synthetic_outer_timing')
 finally:model.infer=saved
if mode=='fault':
 graph=model.decoder.prefill_graphs['16-0'];graph._test_graph_workspace_encode_op(1)
 bad=json.loads(garnet.infer_json(prompt,'',8,name));assert bad['status']=='error'and'Injected ordinary workspace encoder failure'in bad['error_message'];clean('native_encoder_error')
b=json.loads(garnet.infer_json(prompt,'',8,name));assert b['status']=='ok'and b['generated_token_ids']==a['generated_token_ids']and b['text']==a['text'];clean('real_recovery')
(w/('serving-request-'+mode+'.json')).write_text(json.dumps(dict(records=records,all_errors_cleaned_and_recovery_exact=True,synthetic_metric_check_not_speed_evidence=True),indent=2));print('Serving request errors/finally/outer metric accounting/learned recovery PASS '+mode,flush=True)
