"""Real native disabled/fault boundary checks; no performance claim."""
import sys,pathlib,json
mode=sys.argv[1];assert mode in ['disabled','fault'];w=pathlib.Path(__file__).resolve().parent
source='/Users/shawn/CantorAI/out/research/worldsense-decode-step-profile/'+('fault-source'if mode=='fault'else'source');build='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-decode-step-profile-'+mode
sys.path.insert(0,build);sys.path.insert(0,source);sys.path.insert(0,source+'/runtime')
import garnet,metal_serving
from step_profile_checks import validate
name='Qwen3-1.7B';result=json.loads(garnet.serve_model('/Users/shawn/CantorAI/out/models/'+name,source+'/xModel/qwen/metal_dense','/Users/shawn/CantorAI/out/cache/'+name,json.dumps(dict(maxInputTokens=192,maxOutputTokens=64,prefillChunk=16,textPrefixCache=True)),name));assert result['ready'];model=metal_serving._models[name]
base='Decide whether the visual facts satisfy the required condition. Use only the observation as evidence. Missing or contradictory evidence means matched=false. Do not substitute a different subject or condition. A still image cannot prove a transition. Return only a compact JSON object with matched and reason. '*2
prompt=base+'Observation: a red car passes a stationary viewpoint. Condition: a passenger vehicle is visible.'
reference=next(r for r in json.loads((w.parent/'worldsense-prefix-cache/prefix-model-guard.json').read_text())['records']if r['prompt_index']==0)
def infer():
 call=json.loads(garnet.infer_json(prompt,'',8,name));assert call['status']=='ok'and call['generated_token_ids']==reference['generated_token_ids']
 if mode=='fault':validate(call,model.stops)
 else:assert 'decode_step_profile'not in call['timings']and 'decode_step_profile'not in call['backend']
 return call
try:
 first=infer()
 if mode=='fault':
  model.decoder.layers[0]['graph']._test_graph_workspace_encode_op(1)
  bad=json.loads(garnet.infer_json(prompt,'',8,name));assert bad['status']=='error'and 'Injected ordinary workspace encoder failure'in bad['error_message']
 else:
  bad=json.loads(garnet.infer_json('word '*4096,'',8,name));assert bad['status']=='error'and 'capacity'in bad['error_message']
 assert 'decode_step_profile'not in bad.get('timings',{})
 recovered=infer();assert recovered['generated_token_ids']==first['generated_token_ids']
 (w/('profile-boundary-'+mode+'.json')).write_text(json.dumps(dict(first=first,error=bad,recovered=recovered,scope='Native correctness/error/no stale trace/recovery only; no timing claim'),indent=2))
 print('Native '+mode+' profile boundary/exact tokens/error/no stale trace/reset recovery PASS',flush=True)
finally:model.decoder.close();del metal_serving._models[name]
