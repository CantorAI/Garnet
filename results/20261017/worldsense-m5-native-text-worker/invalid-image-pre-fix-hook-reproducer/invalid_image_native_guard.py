"""FAULT-only exact Galaxy C++ default-invalid image boundary, no model loaded."""
import sys,json
from pathlib import Path
w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-native-text-worker');s=Path('/Users/shawn/CantorAI/out/research/worldsense-native-text-worker/source');b=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-native-text-worker-fault');sys.path.insert(0,str(s));sys.path.insert(0,str(b));import garnet
expected_failure=sys.argv[1]=='before';record={}
def save():(w/('invalid-image-native-'+sys.argv[1]+'.json')).write_text(json.dumps(record,indent=2)+'\n')
try:
 record['serve']=json.loads(garnet.serve_model(str(w/'invalid-text-fixture'),str(s/'xModel/qwen/metal_dense'),str(w/'invalid-text-cache'),json.dumps(dict(maxInputTokens=64,maxOutputTokens=8,textPrefixCache=False)),'bad-text'));save();assert not record['serve']['ready']
 record['info_before']=json.loads(garnet.native_text_worker_info_json(''));save();assert record['info_before']['children'][0]['infer_calls']==0
 record['original_local_control']=json.loads(garnet.native_invalid_image_test_json('test',2,'bad-text',False));save();assert record['original_local_control']['status']=='error'and 'Model is not loaded'in record['original_local_control']['error_message']
 failed=False
 try:record['remote']=json.loads(garnet.native_invalid_image_test_json('test',2,'bad-text',True));save()
 except Exception as e:record['exception']=str(e);save();failed='XLangStream IPC marshal does not support this value yet'in str(e);assert failed
 assert failed==expected_failure
 if not failed:
  assert record['remote']['status']=='error'and record['remote']['error_message']==record['original_local_control']['error_message']and record['remote']['native_worker_call_index']==1
  record['info_after']=json.loads(garnet.native_text_worker_info_json(''));save();assert record['info_after']['children'][0]['infer_calls']==1 and record['info_after']['children'][0]['birth_id']==record['info_before']['children'][0]['birth_id']
 record['expected_native_CXX_boundary_outcome_pass']=True;record['scope']='Exact default-invalid C++ image constructed inside native Garnet callback; no loaded model or performance inference.';save();print('Native VM/Garnet exact C++invalid-image boundary expected outcome PASS '+sys.argv[1],flush=True)
finally:garnet.shutdown()
