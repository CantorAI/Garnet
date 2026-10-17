from pathlib import Path
import shutil,json,datetime,hashlib,re
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');old=r/'out/research/worldsense-cached-matrix-metadata/source';s=r/'out/research/worldsense-physical-memory/source';assert not s.exists()
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-35 diagnostic','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Distinguish logical Metal allocation reduction from process footprint accounting after WS34 failed the footprint gate. No optimization promotion or performance decision.','source':'Frozen WS34 seed. OFF vs ON retains only lazy-retirement/cached-metadata policy differences. New diagnostic API/runtime instrumentation identical in both.','method':'At each of8realWorldSense model-call boundaries collect globally deduplicated native graph+all-session buffer identities/roles, native device/task snapshots, and OS vmmap-summary/footprint-byte-category outputs while the native request waits for an observer acknowledgement. Original video frame/Skill/prompts/config/weights remain. Native inference/image path unchanged.','observation':'Separate CPython controller thread runs only OS tools and stores metadata; no Python inference or image preparation. Validate launchd target PID and process birth identity before every capture; bounded tool timeouts, native ack timeout and explicit failure retention. No arbitrary process inspection or memory contents.','validation':'New API duplicate graph/session dedup, old-single-session parity, invalid object rejection, unchanged host/private values and actual trained layer fixtures; existing native references. Real WorldSense all8tokens/text/prompt/config/image/Skill and interpreted semantics match WS10 golden.','limits':'OS categories and global logical owner totals do not map exact physical pages to individual buffers. Native graph/session inventory omits driver/context scratch. OS suspension, inventory allocations and barrier waits invalidate all timing/performance claims. Memory categories are observations, not a causal effect estimate. Fresh workers; no inference retries for favorable memory.','restoration':'Restore WS10 normal service/four video files/no camera/benchmarkPOST404 after observation. Preserve all raw outputs, failures, source/build/runtime/model/request identities and decision.'},indent=2))
shutil.copytree(old,s,ignore=shutil.ignore_patterns('__pycache__','.git','results'))
p=s/'cmake/GarnetMetal.cmake';t=p.read_text();t+='''
option(GARNET_METAL_MULTI_INVENTORY "Correctness/diagnostic global graph/session inventory" OFF)
if(GARNET_METAL_MULTI_INVENTORY)
 target_compile_definitions(garnet PRIVATE GARNET_METAL_MULTI_INVENTORY=1)
endif()
''';p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text();a=t.index('    std::string MemoryInventoryJson(');b=t.index('    X::Value PrepareImage(',a);method=t[a:b];assert method.count('std::string MemoryInventoryJson(')==1
many=method.replace('MemoryInventoryJson(X::Value graphs,X::Value session)','MemoryInventoryManyJson(X::Value graphs,X::Value sessions)')
needle='        if(!graphs.IsList())throw X::Error("Inventory graph descriptors must be a list");';assert needle in many;many=many.replace(needle,needle+'\n        if(!sessions.IsList()||sessions.runtime()!=graphs.runtime())throw X::Error("Inventory sessions must be a same-runtime list");')
start=many.index('        if(session.IsValid()');end=many.index('\n        uint64_t bytes=',start)
many=many[:start]+'''        for(uint64_t i=0;i<sessions.Size();++i){auto descriptor=sessions.Get(i);if(!descriptor.IsDict())throw X::Error("Invalid inventory session descriptor");auto value=descriptor.Get("object");if(value.runtime()!=graphs.runtime())throw X::Error("Inventory runtime mismatch");auto native=value.NativeData<MetalDecoderSession>();if(!native)throw X::Error("Inventory requires native session objects");append(descriptor.Get("tag").ToString(),native->MemoryReferences());}
'''+many[end:]
many=many.replace('Collection-local opaque ids, no addresses.','Single global collection of supplied graphs and sessions, opaque ids, no addresses.')
t=t[:b]+'#ifdef GARNET_METAL_MULTI_INVENTORY\n'+many+'#endif\n'+t[b:]
needle='        APISET().AddFunc<2>("memory_inventory_json",&GarnetMetalAPI::MemoryInventoryJson);';assert needle in t;t=t.replace(needle,needle+'\n#ifdef GARNET_METAL_MULTI_INVENTORY\n        APISET().AddFunc<2>("memory_inventory_many_json",&GarnetMetalAPI::MemoryInventoryManyJson);\n#endif');p.write_text(t)
p=s/'runtime/metal_serving.py';t=p.read_text().replace('self.collect_inventory = bool(profile.get("collectNativeInventory", False))','self.collect_inventory = True  # Diagnostic-only frozen source; never promoted.')
a=t.index("            descriptors = [dict(object=layer['graph']");b=t.index('            inventory = json.loads(inventory_payload)',a)
t=t[:a]+'''            descriptors=[];sessions=[]
            for model_name in sorted(_models):
                model=_models[model_name]
                descriptors += [dict(object=layer['graph'],tag=model_name+'.decode.layer.'+str(i)) for i,layer in enumerate(model.decoder.layers)]
                descriptors.append(dict(object=model.decoder.head,tag=model_name+'.head'))
                for key in sorted(model.decoder.prefill_graphs):
                    descriptors.append(dict(object=model.decoder.prefill_graphs[key],tag=model_name+'.prefill.'+key))
                if model.vision is not None:descriptors.append(dict(object=model.vision.graph,tag=model_name+'.vision'))
                if model.decoder.session is not None:sessions.append(dict(object=model.decoder.session,tag=model_name+'.session'))
            inventory_payload = garnet.memory_inventory_many_json(descriptors,sessions)
'''+t[b:]
t=t.replace("capture_root = '/Users/shawn/CantorAI/out/research/worldsense-lazy-defaults/captures'","capture_root = '/Users/shawn/CantorAI/out/research/worldsense-physical-memory/captures'")
needle="            stages['memory_inventory'] = dict(path=capture_path, sha256=hashlib.sha256(inventory_bytes).hexdigest(),";assert needle in t
# Hook after the complete compact inventory descriptor statement.
a=t.index(needle);b=t.index("    \n        stages['decode_profile']",a)
t=t[:b]+'''            barrier_root='/Users/shawn/CantorAI/out/research/worldsense-physical-memory/barriers'
            os.makedirs(barrier_root,exist_ok=True)
            pending=os.path.join(barrier_root,capture_key+'.pending.json');ack=os.path.join(barrier_root,capture_key+'.ack.json')
            observation=dict(capture_id=capture_key,pid=os.getpid(),model_id=self.name,inventory=stages['memory_inventory'],native_before=json.loads(garnet.memory_stats_json()),loaded_model_ids=sorted(_models))
            with open(pending+'.tmp','w')as stream:stream.write(json.dumps(observation))
            os.replace(pending+'.tmp',pending)
            barrier_started=time.monotonic()
            while not os.path.exists(ack):
                if time.monotonic()-barrier_started>90:raise ValueError('Memory observer acknowledgement timed out')
                time.sleep(.01)
            with open(ack)as stream:observed=json.loads(stream.read())
            if not observed.get('ok') or observed.get('capture_id')!=capture_key:raise ValueError('Memory observer failed: '+str(observed.get('error','identity mismatch')))
            stages['os_memory_observation']=dict(capture_id=capture_key,barrier_wait_seconds=time.monotonic()-barrier_started,memory_after_observation=json.loads(garnet.memory_stats_json()),scope='Diagnostic only: OS observation pauses invalidate performance metrics')
            os.replace(pending,os.path.join(barrier_root,capture_key+'.complete.json'))
'''+t[b:];p.write_text(t)
# Complete static delta audit; no inference is dispatched here.
changed=[str(p.relative_to(s))for p in s.rglob('*')if p.is_file()and'__pycache__'not in p.parts and(not(old/p.relative_to(s)).exists()or p.read_bytes()!=(old/p.relative_to(s)).read_bytes())]
assert sorted(changed)==['cmake/GarnetMetal.cmake','runtime/metal_serving.py','src/entry/metal_main.cpp']
(w/'implementation-audit.json').write_text(json.dumps({'changed_files':changed,'seed':'WS34','all_native_execution_and_shader_CUDA_files_unchanged':True,'instrumentation_only':True},indent=2))
build=(w.parent/'worldsense-cached-matrix-metadata/build.py').read_text().replace('worldsense-cached-matrix-metadata','worldsense-physical-memory').replace("'-DGARNET_METAL=ON'","'-DGARNET_METAL_MULTI_INVENTORY=ON','-DGARNET_METAL=ON'");(w/'build.py').write_text(build)
for name in ['wait_cantor.py','restore_normal.py']:(w/name).write_text((w.parent/'worldsense-cached-matrix-metadata'/name).read_text().replace('worldsense-cached-matrix-metadata','worldsense-physical-memory'))
shutil.copy2(w.parent/'worldsense-cached-matrix-metadata/cases.json',w/'cases.json')
print('WS35 declared, three-file diagnostic delta prepared, no inference dispatched')
