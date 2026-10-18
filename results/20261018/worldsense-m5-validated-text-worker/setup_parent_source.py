"""Implement prospectively reviewed WS61 in a new parent source closure."""
from pathlib import Path
import json,hashlib,shutil,ast,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI')
old=r/'out/research/worldsense-native-text-worker/source';s=r/'out/research/worldsense-validated-text-worker/source'
legacy=r/'out/research/worldsense-prefix-cache/source'
seed=json.loads((w.parent/'worldsense-native-text-worker/gateway-source-manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert len(seed)==1723 and all(sha(old/k)==v for k,v in seed.items())
assert json.loads((w/'native-host-guards-result.json').read_text())['source_library_runtime_drivers_unchanged']
assert json.loads((w/'structural-RAW-counter-source-proof.json').read_text())['logical_primary_and_lazy_domains_absent']
assert not s.exists();s.mkdir(parents=True)
for k in seed:
 p=s/k;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/k,p)
(w/'parent-seed-source-manifest.json').write_text(json.dumps(seed,indent=2)+'\n')
def replace(rel,a,b):
 p=s/rel;t=p.read_text();assert t.count(a)==1,(rel,a[:80],t.count(a));p.write_text(t.replace(a,b,1))
shutil.copy2(w/'native-host/worker_host.mm',s/'src/entry/validated_worker_host.mm')
# Metadata only: existing start/first clocks returned with original model output.
overlay=(legacy/'runtime/metal_serving.py').read_text()
needle='        return dict(input_prompt_sha256=';assert overlay.count(needle)==1
overlay=overlay.replace(needle,'        return dict(native_model_start_monotonic=start, native_first_token_monotonic=first, input_prompt_sha256=',1)
p=s/'runtime/validated_ws10/metal_serving.py';p.parent.mkdir();p.write_text(overlay);ast.parse(overlay)
replace('src/entry/metal_main.cpp','        return info;\n    }\n    // Graphs own','#ifdef GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW\n        info+="; native_text_worker_validated_ws10";\n#endif\n        return info;\n    }\n    // Graphs own')
rel='src/entry/native_text_worker.cpp'
needle=' J Info(){'
replace(rel,needle,'''#ifdef GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW
 void ValidateProvenance(const J& info){
  const auto host=info.at("support"),model=info.at("model_backend");
  const fs::path source=GARNET_METAL_SOURCE_ROOT,raw=fs::weakly_canonical(GARNET_METAL_TEXT_WORKER_RAW_ROOT),hostRoot=GARNET_METAL_TEXT_WORKER_HOST_ROOT,legacySource=GARNET_METAL_VALIDATED_RAW_SOURCE_ROOT;
  if(fs::weakly_canonical(host.at("library_path").get<std::string>())!=fs::weakly_canonical(hostRoot/"libgarnet_worker_host.dylib")||fs::weakly_canonical(host.at("source_root").get<std::string>())!=fs::weakly_canonical(source))throw std::runtime_error("Validated native worker host identity mismatch");
  if(fs::weakly_canonical(model.at("library_path").get<std::string>())!=raw/"libgarnet.dylib"||model.at("library_sha256")!="dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72"||model.at("loaded_garnet_images")!=1||model.at("package_abi")!=X3_ABI_VERSION||fs::weakly_canonical(model.at("expected_original_source_root").get<std::string>())!=fs::weakly_canonical(legacySource))throw std::runtime_error("Validated native worker model-backend identity mismatch");
  if(fs::weakly_canonical(info.at("serving_source").get<std::string>())!=fs::weakly_canonical(source/"runtime/validated_ws10/metal_serving.py")||fs::weakly_canonical(info.at("decoder_source").get<std::string>())!=fs::weakly_canonical(legacySource/"xModel/qwen/metal_dense/decoder.py")||info.at("raw_counter_policy")!="pinned_ws10_structural_absence")throw std::runtime_error("Validated native worker serving/decoder/counter provenance mismatch");
 }
#endif
 J Info(){''')
needle='throw std::runtime_error("Native text worker ownership changed");return info;'
replace(rel,needle,'''throw std::runtime_error("Native text worker ownership changed");
#ifdef GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW
   ValidateProvenance(info);
#endif
   return info;''')
needle='  std::vector<std::string> args={exe.string(),"-B",(source/"runtime/native_text_worker.py").string(),std::to_string(logical),std::to_string(getpid()),birth,raw.string(),source.string(),ready.string()};'
replace(rel,needle,'''#ifdef GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW
  std::vector<std::string> args={exe.string(),"-B",(source/"runtime/validated_text_worker.py").string(),std::to_string(logical),std::to_string(getpid()),birth,raw.string(),source.string(),ready.string(),GARNET_METAL_TEXT_WORKER_HOST_ROOT,GARNET_METAL_VALIDATED_RAW_SOURCE_ROOT};
#else
'''+needle+'''
#endif''')
needle='   if(fs::weakly_canonical(support.at("library_path").get<std::string>())!=raw/"libgarnet.dylib"||fs::weakly_canonical(support.at("source_root").get<std::string>())!=fs::weakly_canonical(source))throw std::runtime_error("Native worker library/source mismatch");'
replace(rel,needle,'#ifndef GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW\n'+needle+'\n#endif')
p=s/'cmake/GarnetMetal.cmake';p.write_text(p.read_text()+'''
option(GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW "Use the explicitly pinned validated native RAW text backend with native instrumentation" OFF)
set(GARNET_METAL_VALIDATED_RAW_SOURCE_ROOT "" CACHE PATH "Immutable original source of the pinned validated RAW backend")
if(GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW)
 if(NOT GARNET_METAL_NATIVE_TEXT_WORKER OR NOT GARNET_METAL_XMODEL OR NOT TARGET garnet)
  message(FATAL_ERROR "Validated RAW worker requires owned native worker and Metal xModel API")
 endif()
 if(NOT EXISTS "${GARNET_METAL_VALIDATED_RAW_SOURCE_ROOT}/runtime/metal_serving.py")
  message(FATAL_ERROR "Validated RAW worker requires its immutable original source closure")
 endif()
 file(SHA256 "${GARNET_METAL_TEXT_WORKER_RAW_ROOT}/libgarnet.dylib" _validated_raw_sha)
 if(NOT _validated_raw_sha STREQUAL "dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72")
  message(FATAL_ERROR "Validated RAW worker library identity mismatch")
 endif()
 add_library(garnet_worker_host SHARED src/entry/validated_worker_host.mm src/entry/native_worker_support.cpp)
 set_target_properties(garnet_worker_host PROPERTIES CXX_STANDARD 17 OBJCXX_STANDARD 17 CXX_STANDARD_REQUIRED YES OBJCXX_STANDARD_REQUIRED YES)
 target_include_directories(garnet_worker_host PRIVATE "${XLANG3_SDK_ROOT}" third_party src/entry)
 target_compile_definitions(garnet_worker_host PRIVATE XLANG3_USE_SHARED=1 WORKER_HOST_ROOT="${CMAKE_CURRENT_SOURCE_DIR}" WORKER_LEGACY_LIBRARY="${GARNET_METAL_TEXT_WORKER_RAW_ROOT}/libgarnet.dylib" WORKER_LEGACY_SOURCE="${GARNET_METAL_VALIDATED_RAW_SOURCE_ROOT}")
 target_compile_options(garnet_worker_host PRIVATE -fobjc-arc -fno-fast-math -ffp-contract=off -Wno-deprecated-declarations)
 target_link_libraries(garnet_worker_host PRIVATE ${_garnet_metal_runtime} "-framework Foundation" "-framework Metal")
 target_compile_definitions(garnet PRIVATE GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW=1 GARNET_METAL_TEXT_WORKER_HOST_ROOT="${CMAKE_CURRENT_BINARY_DIR}" GARNET_METAL_VALIDATED_RAW_SOURCE_ROOT="${GARNET_METAL_VALIDATED_RAW_SOURCE_ROOT}")
endif()
''')
# Child service preserves existing protocol/normalization; actual old native code
# supplies model results, and the distinct C++ host supplies measured OS data.
worker='''"""Owned native XLang3 WS10 text service; never execute with CPython."""
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
'''
(s/'runtime/validated_text_worker.py').write_text(worker);ast.parse(worker)
manifest={str(p.relative_to(s)):sha(p)for p in sorted(s.rglob('*'))if p.is_file()}
changed=sorted(k for k,v in manifest.items()if seed.get(k)!=v)
assert len(manifest)==1726 and changed==sorted(['cmake/GarnetMetal.cmake','src/entry/metal_main.cpp','src/entry/native_text_worker.cpp','src/entry/validated_worker_host.mm','runtime/validated_text_worker.py','runtime/validated_ws10/metal_serving.py'])
assert all(manifest[k]==v for k,v in seed.items()if k not in changed)
assert (s/'runtime/validated_ws10/metal_serving.py').read_text().replace('native_model_start_monotonic=start, native_first_token_monotonic=first, ','')==(legacy/'runtime/metal_serving.py').read_text()
(w/'parent-implementation-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(w/'parent-implementation-audit.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed_files=1723,parent_files=1726,changed_files=changed,all_other_parent_source_exact58=True,legacy_serving_overlay_inverse_exact=True,original1706_source_and_DLL_immutable=True,no_build_or_model_inference_yet=True),indent=2)+'\n')
print('WS61 parent1726/six-file delta implemented; model/shader/graph/session/CUDA exact58 and old1706 source unchanged')
