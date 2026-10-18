"""Owned native XLang3/Garnet RAW text service; never run with CPython."""
import sys,os,time,json,hashlib,struct,math
from pathlib import Path
port=int(sys.argv[1]);parent=int(sys.argv[2]);birth=sys.argv[3];raw=Path(sys.argv[4]).resolve();source=Path(sys.argv[5]).resolve();ready=Path(sys.argv[6])
sys.path.insert(0,str(source/'runtime'));sys.path.insert(0,str(source));sys.path.insert(0,str(raw))
import garnet,metal_serving
info=json.loads(garnet.native_worker_support_json());assert info['pid']==os.getpid()and info['parent_pid']==parent and os.getppid()==parent
assert Path(info['library_path']).resolve()==raw/'libgarnet.dylib'and Path(info['source_root']).resolve()==source
backend=garnet.backend_info()
for marker in ['primary_bf16_blocks','wide_primary_storage','lazy_resident_defaults','cached_matrix_metadata','primary_vector_prefill','primary_prefill_isolated_library','direct_float_weight_upload','primary_vector_prefill_k32_only','primary_full_isolated_library','native_text_worker']:
 if marker in backend:raise ValueError('Native text worker requires the frozen RAW non-worker library')
assert garnet.native_worker_watch_parent(parent)
metal_serving._native_gateway_clock=True
class NativeTextService:
 def __init__(self):self.running=True
 def info_json(self):
  return json.dumps(dict(protocol='garnet-native-text-v1',birth_id=birth,support=json.loads(garnet.native_worker_support_json()),backend=backend,models=list(metal_serving._models),clock_epoch_metadata_enabled=metal_serving._native_gateway_clock))
 def memory_json(self):return garnet.memory_stats_json()
 def primary_stats_json(self):return garnet.primary_storage_stats_json()
 def set_benchmark_prefill_policy(self,enabled):return garnet.set_benchmark_prefill_policy(enabled)
 def serve_model(self,root,xmodel,cache,profile,name):
  with open(os.path.join(root,'config.json'))as f:config=json.loads(f.read())
  if 'vision_config'in config:raise ValueError('Native RAW text worker rejects vision model routing')
  return garnet.serve_model(root,xmodel,cache,profile,name)
 def serve_status_json(self,name):return garnet.serve_status_json(name)
 def infer_json(self,prompt,image,limit,name):return garnet.infer_json(prompt,image,limit,name)
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
 def stop(self):self.running=False;return True
service=NativeTextService();register_remote_object('garnet_text_worker',service);lrpc_listen(port,False)
if ready.exists():raise ValueError('Native worker readiness path already exists')
with open(str(ready)+'.tmp','w')as f:f.write(service.info_json())
os.replace(str(ready)+'.tmp',str(ready))
while service.running:time.sleep(.05)
for model in list(metal_serving._models.values()):
 model.decoder.close()
 if model.vision is not None:model.vision.close()
metal_serving._models.clear()
