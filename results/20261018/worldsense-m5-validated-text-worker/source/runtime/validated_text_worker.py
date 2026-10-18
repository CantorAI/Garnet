"""Owned native XLang3 WS10 text service; never execute with CPython."""
import sys,os,time,json,hashlib,struct,math
from pathlib import Path
port=int(sys.argv[1]);parent=int(sys.argv[2]);birth=sys.argv[3];raw=Path(sys.argv[4]).resolve();source=Path(sys.argv[5]).resolve();ready=Path(sys.argv[6]);host=Path(sys.argv[7]).resolve();original=Path(sys.argv[8]).resolve();adapter=source/'runtime/validated_ws10'
for p in [original,original/'runtime',host,raw,adapter]:sys.path.insert(0,str(p))
import garnet_worker_host,garnet,metal_serving
import xModel.qwen.metal_dense.decoder as dense_module
support=json.loads(garnet_worker_host.support_json());model_backend=json.loads(garnet_worker_host.legacy_json())
assert support['pid']==os.getpid()and support['parent_pid']==parent and os.getppid()==parent
assert Path(support['library_path']).resolve()==host/'libgarnet_worker_host.dylib'and Path(support['source_root']).resolve()==source
assert Path(model_backend['library_path']).resolve()==raw/'libgarnet.dylib'and Path(model_backend['expected_original_source_root']).resolve()==original
assert Path(metal_serving.__file__).resolve()==adapter/'metal_serving.py'and Path(dense_module.__file__).resolve()==original/'xModel/qwen/metal_dense/decoder.py'
backend=garnet.backend_info()
for marker in ['primary_bf16_blocks','wide_primary_storage','lazy_resident_defaults','cached_matrix_metadata','primary_vector_prefill','primary_prefill_isolated_library','direct_float_weight_upload','primary_vector_prefill_k32_only','primary_full_isolated_library','native_text_worker']:
 if marker in backend:raise ValueError('Validated text worker requires the exact original RAW native backend')
assert garnet_worker_host.watch_parent(parent)
class NativeTextService:
 def __init__(self):self.running=True;self.infer_calls=0
 def info_json(self):
  return json.dumps(dict(protocol='garnet-native-text-v1',birth_id=birth,support=json.loads(garnet_worker_host.support_json()),model_backend=json.loads(garnet_worker_host.legacy_json()),backend=backend,models=list(metal_serving._models),clock_epoch_metadata_enabled=True,infer_calls=self.infer_calls,serving_source=metal_serving.__file__,decoder_source=dense_module.__file__,raw_counter_policy='pinned_ws10_structural_absence',raw_counter_origin=json.loads(garnet_worker_host.raw_policies_json())['counter_origin']))
 def memory_json(self):return garnet_worker_host.memory_stats_json()
 def primary_stats_json(self):return garnet_worker_host.raw_primary_stats_json()
 def set_benchmark_prefill_policy(self,enabled):return garnet.set_benchmark_prefill_policy(enabled)
 def serve_model(self,root,xmodel,cache,profile,name):
  with open(os.path.join(root,'config.json'))as f:config=json.loads(f.read())
  if 'vision_config'in config or config.get('model_type')!='qwen3' or config.get('hidden_size')!=2048 or config.get('num_hidden_layers')!=28:raise ValueError('Validated RAW worker requires the original Qwen3-1.7B profile')
  return garnet.serve_model(root,xmodel,cache,profile,name)
 def serve_status_json(self,name):return garnet.serve_status_json(name)
 def infer_json(self,prompt,image,limit,name):
  self.infer_calls+=1
  before=json.loads(garnet_worker_host.raw_primary_stats_json())
  result=json.loads(garnet.infer_json(prompt,image,limit,name));result['native_worker_call_index']=self.infer_calls
  if result.get('status')=='ok':
   timings=result['timings'];timings['memory_after_decode']=json.loads(garnet_worker_host.memory_stats_json())
   policies=json.loads(garnet_worker_host.raw_policies_json());timings.update(policies)
   after=json.loads(garnet_worker_host.raw_primary_stats_json());timings['primary_storage_before']=before;timings['primary_storage_after']=after
   timings['primary_encoded_delta']={key:after[key]-before[key]for key in ['decode_encoded','prefill_encoded','head_encoded','vector_decode_encoded','vector_head_encoded']}
  return json.dumps(result)
 def state_json(self,name):
  if name not in metal_serving._models:raise ValueError('Native worker model not ready')
  model=metal_serving._models[name];layers=[]
  for i in range(len(model.decoder.layers)):
   row=[]
   for value in model.decoder.session.snapshot(i):
    values=value.tolist()
    if not all(math.isfinite(v)for v in values):raise ValueError('Nonfinite native worker model state')
    h=hashlib.sha256()
    for j in range(0,len(values),4096):
     block=values[j:j+4096];h.update(struct.pack('<'+'f'*len(block),*block))
    row.append(dict(values=len(values),sha256=h.hexdigest()))
   layers.append(row)
  return json.dumps(dict(model_id=name,states=layers,pid=os.getpid(),birth_id=birth))
 def close_model(self,name):
  if name in metal_serving._models:
   model=metal_serving._models.pop(name);model.decoder.close()
   if model.vision is not None:model.vision.close()
  return json.dumps(dict(closed=True,model_id=name))
 def stop(self):self.running=False;return True
service=NativeTextService();register_remote_object('garnet_text_worker',service);lrpc_listen(port,False)
if ready.exists():raise ValueError('Native worker readiness path already exists')
with open(str(ready)+'.tmp','w')as f:f.write(service.info_json())
os.replace(str(ready)+'.tmp',str(ready))
while service.running and not os.path.exists(str(ready.parent/'stop.request')):time.sleep(.05)
for model in list(metal_serving._models.values()):
 model.decoder.close()
 if model.vision is not None:model.vision.close()
metal_serving._models.clear()
