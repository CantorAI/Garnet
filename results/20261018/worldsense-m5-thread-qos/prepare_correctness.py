from pathlib import Path
import json,hashlib,shutil,ast
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-native-text-worker';names=['restore_normal.py','wait_cantor.py','native_controller.py','gateway_nomodel_guard.py','text_model_guard.py','text_guard_controller.py','invalid_image_native_guard.py','invalid_image_loaded_guard.py','full_correctness.py','verify_correctness_terminal.py','build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py']
for name in names:(w/name).write_text((old/name).read_text().replace('worldsense-native-text-worker','worldsense-thread-qos'))
for name in ['before-libgarnet.dylib','cases.json','primary_corpus_test.cpp','direct_float_corpus_test.cpp','direct-upload-selection.json']:shutil.copy2(old/name,w/name)
# Model guard additionally verifies the actual native scope for every success/error.
helper='''def qos_valid(call):
 q=call['native_worker_qos'];a=q['before'];b=q['after'];scope=q['scope'];assert q['mode']=='user_initiated'and scope['api_start_accepted']and scope['closed']and scope['api_end_status']==0 and scope['target_class']==25 and scope['relative_priority']==0
 assert a['get_status']==b['get_status']==a['thread_id_status']==b['thread_id_status']==0 and a['thread_id']==b['thread_id']==scope['before']['thread_id']==scope['during']['thread_id']==scope['after']['thread_id']
 assert a['requested_class']==b['requested_class']==scope['during']['requested_class']and b['active_scopes']==b['end_errors']==0 and scope['during']['active_scopes']==1 and b['starts']==a['starts']+1 and b['ends']==a['ends']+1
'''
(w/'qos_call_guard.py').write_text(helper)
p=w/'text_model_guard.py';t=p.read_text();needle="def infer(label,prompt):return capture(label,lambda:json.loads(garnet.infer_json(prompt,'',8,name)))";assert needle in t;t=t.replace(needle,"from qos_call_guard import qos_valid\ndef infer(label,prompt):\n value=capture(label,lambda:json.loads(garnet.infer_json(prompt,'',8,name)));qos_valid(value);return value");p.write_text(t)
p=w/'invalid_image_loaded_guard.py';t=p.read_text().replace('import garnet\n','import garnet\nfrom qos_call_guard import qos_valid\n');needle="row=dict(kind=kind,call=call);";assert needle in t;t=t.replace(needle,"qos_valid(call);row=dict(kind=kind,call=call);");p.write_text(t)
p=w/'api_state_guard.py';t=p.read_text().replace('import garnet,metal_serving\n','import garnet,metal_serving\nfrom qos_call_guard import qos_valid\n');needle=" if mode!='off':\n  gateway=call['timings']['native_gateway'];";assert t.count(needle)==1;t=t.replace(needle," if remote:qos_valid(call)\n"+needle,1);p.write_text(t)
p=w/'gateway_nomodel_guard.py';t=p.read_text().replace('import garnet\n','import garnet\nfrom qos_call_guard import qos_valid\n');needle="assert response['status']=='error'and response['native_worker_call_index']==1";assert needle in t;t=t.replace(needle,needle+';qos_valid(response)');p.write_text(t)
# Old source auditors preserve all numerical guards; new standalone source
# proof below accounts explicitly for6 changed platform/serving files.
for name in names+['qos_call_guard.py']:ast.parse((w/name).read_text())
print('Prepared native/no-model/full32VLM28text/corpus/vision/session guards with explicit per-call QoS scope verification')
