from pathlib import Path
import shutil,json,datetime
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-inventory/source')
shutil.copytree('/Users/shawn/CantorAI/out/research/worldsense-memory/source',s,ignore=shutil.ignore_patterns('__pycache__','.git','results','docs'))
p=s/'src/backends/metal/metal_context.h';t=p.read_text().replace('struct MetalMemoryStats {','''struct MetalBufferInfo {
    uintptr_t identity=0;uint64_t allocatedBytes=0,lengthBytes=0;std::string representation;
};
struct MetalResourceReference {
    MetalBufferInfo buffer;size_t slot=0;std::string role,name;
    std::vector<int64_t> shape;bool residentOperand=false,input=false,outputPart=false,legacyOutput=false,weight=false;
};
struct MetalMemoryStats {''',1);t=t.replace('    static MetalMemoryStats MemoryStats();','''    static MetalMemoryStats MemoryStats();
    static MetalBufferInfo MatrixBufferInfo(const MetalMatrix&);
    static std::vector<MetalBufferInfo> WeightBufferInfo(const MetalBF16Weight&);''');p.write_text(t)
p=s/'src/backends/metal/metal_context.mm';t=p.read_text();idx=t.index('MetalMemoryStats MetalContext::MemoryStats()');t=t[:idx]+'''MetalBufferInfo MetalContext::MatrixBufferInfo(const MetalMatrix& value) {
    @autoreleasepool {
        if(!value.impl_||!value.impl_->matrix)return {};
        id<MTLBuffer> buffer=value.impl_->matrix.data;if(!buffer)return {};
        return {reinterpret_cast<uintptr_t>((__bridge void*)buffer),uint64_t(buffer.allocatedSize),uint64_t(buffer.length),"matrix_fp32"};
    }
}
std::vector<MetalBufferInfo> MetalContext::WeightBufferInfo(const MetalBF16Weight& value) {
    @autoreleasepool {
        std::vector<MetalBufferInfo> result;if(!value.impl_)return result;auto w=value.impl_;
        auto add=[&](id<MTLBuffer> buffer,const char* kind){if(buffer)result.push_back({reinterpret_cast<uintptr_t>((__bridge void*)buffer),uint64_t(buffer.allocatedSize),uint64_t(buffer.length),kind});};
        add(w->buffer,"bf16");if(w->exactHalf)add(w->exactHalf.data,"exact_half");add(w->halfScales,"half_scales");
#ifdef GARNET_METAL_MLP_PREFILL_GRAPH
        add(w->graphFloat,"graph_float");
#endif
#ifdef GARNET_METAL_INTERLEAVED_BF16_BODY
        add(w->interleavedBody,"interleaved_body");
#ifdef GARNET_METAL_COALESCED_BF16_BODY
        add(w->coalescedPayload,"coalesced_body");add(w->coalescedMetadata,"coalesced_metadata");
#endif
#endif
#ifdef GARNET_METAL_SCREENED_HEAD
        add(w->screenIndex,"screen_index");add(w->screenMetadata,"screen_metadata");
#endif
        return result;
    }
}
'''+t[idx:];p.write_text(t)
p=s/'src/backends/metal/compiled_graph.h';t=p.read_text().replace('    bool Ready() const {return ready_;}', '    std::vector<MetalResourceReference> MemoryReferences();\n    bool Ready() const {return ready_;}');p.write_text(t)
p=s/'src/backends/metal/compiled_graph.cpp';t=p.read_text()+'''
namespace Garnet {
std::vector<MetalResourceReference> MetalCompiledGraph::MemoryReferences() {
    std::lock_guard<std::mutex> lock(mutex_);if(!ready_)throw std::logic_error("Inventory requires a ready graph");
    std::vector<MetalResourceReference> result;std::vector<bool> active(slots_.size(),false),input(slots_.size(),false),output(slots_.size(),false);
    auto mark=[&](size_t i){if(i<active.size())active[i]=true;};
    for(size_t i=0;i<ops_.size();++i)if(residentOps_.size()!=ops_.size()||residentOps_[i]) {
        const auto& op=ops_[i];mark(op.a);mark(op.b);mark(op.out);mark(op.auxiliaryOutput);for(auto j:op.arguments)mark(j);
    }
    for(auto i:inputs_)if(i<input.size())input[i]=true;for(auto i:outputParts_)if(i<output.size())output[i]=true;
    std::vector<std::string> names(slots_.size());std::vector<std::pair<std::string,size_t>> sorted(weights_.begin(),weights_.end());std::sort(sorted.begin(),sorted.end());
    for(const auto& entry:sorted)if(entry.second<names.size()){if(!names[entry.second].empty())names[entry.second]+=";";names[entry.second]+=entry.first;}
    auto append=[&](const Slot& slot,size_t i,const std::string& role,const std::string& name,bool weight,bool in,bool out,bool resident,bool legacy){
        auto emit=[&](const MetalBufferInfo& info){if(!info.identity)return;MetalResourceReference r;r.buffer=info;r.slot=i;r.role=role;r.name=name;r.shape=slot.shape;r.weight=weight;r.input=in;r.outputPart=out;r.residentOperand=resident;r.legacyOutput=legacy;result.push_back(std::move(r));};
        emit(MetalContext::MatrixBufferInfo(slot.matrix));for(const auto& info:MetalContext::WeightBufferInfo(slot.bf16))emit(info);
    };
    for(size_t i=0;i<slots_.size();++i)append(slots_[i],i,"slot",names[i],!names[i].empty(),input[i],output[i],active[i],i==output_);
    auto held=[&](const auto& records,const char* role){std::vector<std::string> keys;for(const auto& entry:records)keys.push_back(entry.first);std::sort(keys.begin(),keys.end());for(const auto& key:keys)append(records.at(key).slot,SIZE_MAX,role,key,true,false,false,false,false);};
    held(sharedWeights_,"shared_weight_record");held(weightRecords_,"weight_record");return result;
}
}
''';p.write_text(t)
p=s/'src/backends/metal/decoder_session.h';t=p.read_text().replace('    X::Value Snapshot(int64_t layer);','    std::vector<MetalResourceReference> MemoryReferences();\n    X::Value Snapshot(int64_t layer);');p.write_text(t)
p=s/'src/backends/metal/decoder_session.cpp';t=p.read_text()+'''
namespace Garnet {
std::vector<MetalResourceReference> MetalDecoderSession::MemoryReferences() {
    std::lock_guard<std::mutex> lock(mutex_);if(!gpu_)throw std::logic_error("Inventory requires an open session");std::vector<MetalResourceReference> result;
    auto append=[&](const MetalMatrix& matrix,size_t slot,const std::string& role,const std::vector<int64_t>& shape){auto info=MetalContext::MatrixBufferInfo(matrix);if(!info.identity)return;MetalResourceReference r;r.buffer=info;r.slot=slot;r.role=role;r.shape=shape;r.residentOperand=true;result.push_back(std::move(r));};
    for(size_t i=0;i<states_.size();++i)for(size_t k=0;k<2;++k)append(states_[i].matrix[k],i,"private_state"+std::to_string(k+1),states_[i].shape[k]);
    append(hidden_,SIZE_MAX,"hidden",hiddenShape_);append(prediction_,SIZE_MAX,"prediction",{});return result;
}
}
''';p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text().replace('    X::Value PrepareImage(','''    std::string MemoryInventoryJson(X::Value graphs,X::Value session) {
        if(!graphs.IsList())throw X::Error("Inventory graph descriptors must be a list");
        nlohmann::json references=nlohmann::json::array(),buffers=nlohmann::json::array();std::unordered_map<uintptr_t,size_t> ids;
        auto append=[&](const std::string& tag,const std::vector<MetalResourceReference>& refs){for(const auto& r:refs){
            auto found=ids.find(r.buffer.identity);size_t id;
            if(found==ids.end()){id=buffers.size();ids.emplace(r.buffer.identity,id);buffers.push_back({{"id",id},{"allocated_bytes",r.buffer.allocatedBytes},{"length_bytes",r.buffer.lengthBytes}});}
            else {id=found->second;if(buffers[id]["allocated_bytes"]!=r.buffer.allocatedBytes||buffers[id]["length_bytes"]!=r.buffer.lengthBytes)throw X::Error("Inventory resource identity changed");}
            references.push_back({{"buffer_id",id},{"tag",tag},{"slot",r.slot==SIZE_MAX?nlohmann::json(nullptr):nlohmann::json(r.slot)},{"role",r.role},{"name",r.name},{"shape",r.shape},{"representation",r.buffer.representation},{"resident_mask_operand",r.residentOperand},{"input",r.input},{"output_part",r.outputPart},{"legacy_output",r.legacyOutput},{"weight",r.weight}});
        }};
        for(uint64_t i=0;i<graphs.Size();++i){auto descriptor=graphs.Get(i);if(!descriptor.IsDict())throw X::Error("Invalid inventory graph descriptor");auto value=descriptor.Get("object");if(value.runtime()!=graphs.runtime())throw X::Error("Inventory runtime mismatch");auto graph=value.NativeData<MetalCompiledGraph>();if(!graph)throw X::Error("Inventory requires graph objects");append(descriptor.Get("tag").ToString(),graph->MemoryReferences());}
        if(session.IsValid()&&!session.IsNone()){if(session.runtime()!=graphs.runtime())throw X::Error("Inventory session runtime mismatch");auto native=session.NativeData<MetalDecoderSession>();if(!native)throw X::Error("Inventory requires a native session");append("session",native->MemoryReferences());}
        uint64_t bytes=0;for(const auto& b:buffers)bytes+=b["allocated_bytes"].get<uint64_t>();
        return nlohmann::json{{"buffers",buffers},{"references",references},{"unique_allocated_bytes",bytes},{"device_snapshot",nlohmann::json::parse(MemoryStatsJson())},{"scope","Collection-local opaque ids, no addresses. Declared graph/held-weight/session owners only; context scratch/driver resources may be absent. Resident mask references are conservative, not a safe-free proof."}}.dump();
    }
    X::Value PrepareImage(''',1);t=t.replace('        APISET().AddFunc<0>("memory_stats_json"','        APISET().AddFunc<2>("memory_inventory_json",&GarnetMetalAPI::MemoryInventoryJson);\n        APISET().AddFunc<0>("memory_stats_json"');p.write_text(t)
p=s/'runtime/metal_serving.py';t=p.read_text();needle="        stages['memory_after_decode'] = json.loads(garnet.memory_stats_json())";replacement=needle+'''
        descriptors = [dict(object=layer['graph'], tag='decode.layer.'+str(i)) for i,layer in enumerate(self.decoder.layers)]
        descriptors.append(dict(object=self.decoder.head, tag='head'))
        for key in sorted(self.decoder.prefill_graphs):
            descriptors.append(dict(object=self.decoder.prefill_graphs[key], tag='prefill.'+key))
        if self.vision is not None:
            descriptors.append(dict(object=self.vision.graph, tag='vision'))
        stages['memory_inventory'] = json.loads(garnet.memory_inventory_json(descriptors, self.decoder.session))
''';assert needle in t;t=t.replace(needle,replacement,1);p.write_text(t)
shutil.copy2('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/libgarnet.dylib',w/'before-libgarnet.dylib')
(w/'controller.py').write_text(Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-memory/controller.py').read_text().replace('worldsense-memory','worldsense-inventory'))
(w/'cases.json').write_bytes(Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-memory/cases.json').read_bytes())
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-14 diagnostic','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Identify retained graph/session GPU buffers behind20.4GiB sampled resource footprint after request-pool memory rejection. Read only: no math, admission, policy, precision or CUDA change.','method':'Native MTLBuffer identity dedup within each model collection, emit opaque ordinal IDs only. Record allocatedSize/length and graph-slot/held-weight/private-session owners, shape, input/output/weight roles and conservative residentOps operand references. No raw addresses, tensor values or extra GPU allocations.','scope':'Context scratch/driver resources not enumerated; compare residual with native device total. Ownership references are not ARC retain counts. A missing resident-mask operand does not prove a buffer safe to free. Separate model collections do not prove absence of cross-model sharing.','workload':'Real WorldSense fixed three frames/Skills plus query, two windows,12 expected complete model calls. Complete token/text/prompt/config/input parity required. Native identity/alias/read-only tests before execution. Diagnostic overhead precludes speed or memory-improvement claim.','restoration':'Restore exact validatedWS10 library/live assignments and normal service with benchmark404 and four files online; no camera.'},indent=2))
print('Prepared passive native buffer inventory')
