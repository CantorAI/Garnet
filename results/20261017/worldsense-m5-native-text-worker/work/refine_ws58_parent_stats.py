from pathlib import Path
import json,hashlib,shutil,datetime
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-native-text-worker';s=Path('/Users/shawn/CantorAI/out/research/worldsense-native-text-worker/source');dest=w/'gateway-before-parent-primary-telemetry';dest.mkdir(exist_ok=False)
manifest=json.loads((w/'gateway-source-manifest.json').read_text());seed=json.loads((w.parent/'worldsense-isolated-primary/correctness-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
for k,v in manifest.items():
 if seed.get(k)!=v:p=dest/'source-overlay'/k;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/k,p)
for p in sorted(w.iterdir()):
 if p.is_file()and(p.suffix in ['.py','.json','.log','.txt']):shutil.copy2(p,dest/p.name)
for mode in ['off','on','primary','fault']:
 p=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-native-text-worker-'+mode)/'libgarnet.dylib';q=dest/('library-'+mode+'.dylib');shutil.copy2(p,q)
shutil.copy2(root/'work/worldsense-opt/ws58_parent_stats_refinement.txt',w/'parent-primary-telemetry-before-source.txt')
(dest/'SHA256.json').write_text(json.dumps({str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(dest.rglob('*'))if p.is_file()},indent=2)+'\n')
for name in ['native_text_worker.h','native_text_worker.cpp']:
 p=s/'src/entry'/name;t=p.read_text();needle='const std::function<std::string()>& parentMemory,const std::string& parentBackend';assert t.count(needle)==1;t=t.replace(needle,'const std::function<std::string()>& parentMemory,const std::function<std::string()>& parentPrimary,const std::string& parentBackend');p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text();needle='[&]{return MemoryStatsJson();},BackendInfo()';assert t.count(needle)==1;t=t.replace(needle,'[&]{return MemoryStatsJson();},[&]{return PrimaryStatsJson();},BackendInfo()');p.write_text(t)
p=s/'src/entry/native_text_worker.cpp';t=p.read_text();needle=' const auto route=impl_->routes.find(model.ToString());const bool remote=';assert t.count(needle)==1;t=t.replace(needle,' const auto parentPrimaryBefore=J::parse(parentPrimary());\n'+needle,1)
needle=' auto parent=J::parse(parentMemory());';assert t.count(needle)==1
add=''' const auto parentPrimaryAfter=J::parse(parentPrimary());J parentDelta=J::object();
 for(const char* key:{"decode_encoded","prefill_encoded","head_encoded","vector_decode_encoded","vector_head_encoded"}){const auto before=parentPrimaryBefore.at(key).get<uint64_t>(),after=parentPrimaryAfter.at(key).get<uint64_t>();if(after<before)throw std::runtime_error("Parent primary encoded counter decreased inside serialized request");parentDelta[key]=after-before;}
 meta["parent_primary_storage_before"]=parentPrimaryBefore;meta["parent_primary_storage_after"]=parentPrimaryAfter;meta["parent_primary_encoded_delta"]=parentDelta;
'''
t=t.replace(needle,add+needle,1);p.write_text(t)
print('Preserved prior gateway source/drivers/4libraries; added separately scoped actual parent primary telemetry')
