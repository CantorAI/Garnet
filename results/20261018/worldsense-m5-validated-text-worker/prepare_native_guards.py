from pathlib import Path
import ast,shutil
w=Path(__file__).resolve().parent;old=w.parent/'worldsense-native-text-worker'
for name in ['native_controller.py','gateway_nomodel_guard.py','restore_normal.py','wait_cantor.py']:
 t=(old/name).read_text().replace('worldsense-native-text-worker','worldsense-validated-text-worker');ast.parse(t);(w/name).write_text(t)
for name in ['before-libgarnet.dylib','cases.json']:shutil.copy2(old/name,w/name)
p=w/'gateway_nomodel_guard.py';t=p.read_text()
needle="def info():return json.loads(garnet.native_text_worker_info_json(''))";assert t.count(needle)==1
t=t.replace(needle,"""def info():
 value=json.loads(garnet.native_text_worker_info_json(''));(w/('gateway-nomodel-last-info-'+mode+'.json')).write_text(json.dumps(value,indent=2))
 for child in value['children']:
  assert Path(child['support']['library_path']).resolve()==build/'libgarnet_worker_host.dylib'and Path(child['support']['source_root']).resolve()==source
  assert child['model_backend']['library_path']=='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib'and child['model_backend']['library_sha256']=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
  assert child['model_backend']['loaded_garnet_images']==1 and child['raw_counter_policy']=='pinned_ws10_structural_absence'
  assert Path(child['serving_source']).resolve()==source/'runtime/validated_ws10/metal_serving.py'
  assert Path(child['decoder_source']).resolve()==Path('/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source/xModel/qwen/metal_dense/decoder.py')
 return value""")
needle="def serve():return json.loads(garnet.serve_model(str(root),str(source/'xModel/qwen/metal_dense'),str(w/'invalid-text-cache'),json.dumps(dict(maxInputTokens=64,maxOutputTokens=8,textPrefixCache=False)),'bad-text'))";assert t.count(needle)==1
t=t.replace(needle,"""def serve():
 value=json.loads(garnet.serve_model(str(root),str(source/'xModel/qwen/metal_dense'),str(w/'invalid-text-cache'),json.dumps(dict(maxInputTokens=64,maxOutputTokens=8,textPrefixCache=False)),'bad-text'));(w/('gateway-nomodel-last-serve-'+mode+'.json')).write_text(json.dumps(value,indent=2));return value""")
needle="response=json.loads(garnet.infer_json('test','',2,'bad-text'));assert";assert t.count(needle)==1
t=t.replace(needle,"response=json.loads(garnet.infer_json('test','',2,'bad-text'));(w/('gateway-nomodel-last-rejection-'+mode+'.json')).write_text(json.dumps(response,indent=2));assert")
ast.parse(t);p.write_text(t)
print('Four-role original native tests and real integrated no-model/error/ownership/no-retry guards prepared')
