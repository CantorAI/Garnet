from pathlib import Path
import shutil,json,datetime
w=Path(__file__).resolve().parent;seed=Path('/Users/shawn/CantorAI/out/research/worldsense-storage-scopes/source');source=Path('/Users/shawn/CantorAI/out/research/worldsense-resident-admission/source');assert not source.exists();shutil.copytree(seed,source,ignore=shutil.ignore_patterns('__pycache__','.git','results','docs'))
p=source/'src/backends/metal/metal_context.h';s=p.read_text().replace('    static void StorageScope(', '    size_t MatrixStorageBytes(size_t,size_t) const;\n    static void StorageScope(',1);p.write_text(s)
p=source/'src/backends/metal/metal_context.mm';s=p.read_text();at=s.index('MetalMatrix MetalContext::Allocate(');s=s[:at]+'''size_t MetalContext::MatrixStorageBytes(size_t rows,size_t columns) const {
    @autoreleasepool {Elements(rows,columns);const auto stride=[MPSMatrixDescriptor rowBytesFromColumns:columns dataType:MPSDataTypeFloat32];
    if(stride>std::numeric_limits<size_t>::max()/rows)throw std::overflow_error("Metal matrix storage overflow");
    if(rows*stride>impl_->device.maxBufferLength)throw std::runtime_error("Metal matrix exceeds device buffer limit");return rows*stride;}
}
'''+s[at:];p.write_text(s)
p=source/'src/backends/metal/compiled_graph.h';s=p.read_text().replace('    void RetireResidentDefaults();','    size_t residentAvoidedStorageBytes_=0;\n    void RetireResidentDefaults();');s=s.replace('bool bf16Projections=false);','bool bf16Projections=false,bool residentDefaults=false);').replace('std::unordered_map<std::string,SharedWeight>,bool);','std::unordered_map<std::string,SharedWeight>,bool,bool);');s=s.replace('    X::Value RunUnscoped(X::Value);','    X::Value RunUnscoped(X::Value);\n    bool BuildSharedInternal(X::Value,X::Value,const std::string&,X::Value,bool);');s=s.replace('    bool BuildBF16(X::Value graph,X::Value symbols,const std::string& checkpoint);','    bool BuildBF16(X::Value graph,X::Value symbols,const std::string& checkpoint);\n    bool BuildBF16Resident(X::Value,X::Value,const std::string&);\n    bool BuildSharedResident(X::Value,X::Value,const std::string&,X::Value);\n    size_t ResidentAvoidedStorageBytes() const;');s=s.replace('        APISET().AddFunc<3>("build_bf16",&MetalCompiledGraph::BuildBF16);','        APISET().AddFunc<3>("build_bf16",&MetalCompiledGraph::BuildBF16);\n        APISET().AddFunc<3>("build_bf16_resident",&MetalCompiledGraph::BuildBF16Resident);\n        APISET().AddFunc<4>("build_shared_resident",&MetalCompiledGraph::BuildSharedResident);\n        APISET().AddProp("resident_avoided_storage_bytes",&MetalCompiledGraph::ResidentAvoidedStorageBytes);');p.write_text(s)
p=source/'src/backends/metal/compiled_graph.cpp';s=p.read_text();s=s.replace('void MetalCompiledGraph::Clear() {','void MetalCompiledGraph::Clear() {\n    residentAvoidedStorageBytes_=0;',1)
s=s.replace('bool MetalCompiledGraph::BuildShared(X::Value graph,X::Value symbols,const std::string& checkpoint,X::Value source) {','bool MetalCompiledGraph::BuildSharedInternal(X::Value graph,X::Value symbols,const std::string& checkpoint,X::Value source,bool residentDefaults) {',1)
s=s.replace('peer->weightRecords_,peer->bf16Projections_);','peer->weightRecords_,peer->bf16Projections_,residentDefaults);',1)
s=s.replace('std::unordered_map<std::string,SharedWeight> shared,bool bf16Projections) {','std::unordered_map<std::string,SharedWeight> shared,bool bf16Projections,bool residentDefaults) {')
s=s.replace('checkpoint,std::move(shared),bf16Projections);','checkpoint,std::move(shared),bf16Projections,residentDefaults);')
needle='        for(uint64_t i=0;i<symbols.Size();++i) {';assert needle in s
s=s.replace(needle,'''        std::vector<size_t> initiallyDeferred;
        auto defer=[&](size_t rows,size_t columns){auto bytes=gpu_->MatrixStorageBytes(rows,columns);if(bytes>std::numeric_limits<size_t>::max()-residentAvoidedStorageBytes_)throw std::overflow_error("Deferred storage accounting overflow");residentAvoidedStorageBytes_+=bytes;auto id=slots_.size();slots_.push_back({MetalMatrix(),rows,columns,{int64_t(rows),int64_t(columns)}});initiallyDeferred.push_back(id);return id;};
        if(residentDefaults&&symbols.Size()!=6)throw std::invalid_argument("Resident construction requires six layer inputs");
'''+needle,1)
s=s.replace('const auto id=Allocate(info.shape[0],columns);inputs_.push_back(id);','const bool deferInput=residentDefaults&&(i==1||i==2);\n            if(deferInput&&info.rank<3)throw std::invalid_argument("Resident construction requires state input rank at least three");\n            const auto id=deferInput?defer(info.shape[0],columns):Allocate(info.shape[0],columns);inputs_.push_back(id);',1)
needle='        TensorGraphCapture capture(graph);';assert needle in s;s=s.replace(needle,needle+'''
        auto rootValue=capture.Outputs();if((rootValue.IsList()||x3_value_object_kind(rootValue.raw())==X3_OBJECT_KIND_TUPLE)&&rootValue.Size()==1)rootValue=rootValue.Get(uint64_t(0));
        if(!IsTensor(rootValue))throw std::runtime_error("Initial Metal graph requires one tensor output");
        const auto rootTensorId=TensorId(rootValue);
        auto packedStorage=[&](const X::Value& output,size_t rows,size_t columns){return residentDefaults&&TensorId(output)==rootTensorId?defer(rows,columns):Allocate(rows,columns);};
''',1)
for old,new in [
('out=Allocate(slots_[a].rows,slots_[a].columns+slots_[b].columns);as[1]+=bs[1];','out=packedStorage(node.output,slots_[a].rows,slots_[a].columns+slots_[b].columns);as[1]+=bs[1];'),
('out=Allocate(slots_[a].rows,packedWidth);','out=packedStorage(node.output,slots_[a].rows,packedWidth);'),
('out=Allocate(batch,chunk*width+2*cacheWidth);','out=packedStorage(node.output,batch,chunk*width+2*cacheWidth);'),
('out=Allocate(batch,slots_[hidden].columns+slots_[convState].columns+slots_[state].columns);','out=packedStorage(node.output,batch,slots_[hidden].columns+slots_[convState].columns+slots_[state].columns);')]:
 assert old in s,old;s=s.replace(old,new)
needle='        resetKVPrefix_[0]=ComputeResetKVPrefix(1);';assert needle in s;s=s.replace(needle,'''#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
        if(residentDefaults) {
            if(initiallyDeferred!=std::vector<size_t>{inputs_[1],inputs_[2],output_}||outputParts_.size()!=3||ops_.empty()||ops_.back().kind!=Kind::Concat||ops_.back().out!=output_||residentOps_.size()!=ops_.size()||residentOps_.back())throw std::invalid_argument("Unsupported resident graph packing");
            for(auto id:initiallyDeferred) {
                if(canonical(id)!=id||std::find(outputParts_.begin(),outputParts_.end(),id)!=outputParts_.end())throw std::invalid_argument("Deferred graph slot aliases a live output");
                for(const auto& alias:aliases)if(canonical(alias.first)==id)throw std::invalid_argument("Deferred graph slot has a logical alias");
                if(id!=output_)for(const auto& op:ops_)if(op.out==id||op.auxiliaryOutput==id)throw std::invalid_argument("Deferred state input has a graph writer");
            }
            lazyDefaultSlots_=initiallyDeferred;lazyDefaultPlanReady_=true;
        }
#endif
'''+needle,1)
s=s.replace('if(!member(op.out)&&source.SharesStorage(slots_[op.out].matrix))','if(!member(op.out)&&(op.out==inputs_[part]||source.SharesStorage(slots_[op.out].matrix)))').replace('if(op.auxiliaryOutput!=SIZE_MAX&&!member(op.auxiliaryOutput)&&source.SharesStorage(slots_[op.auxiliaryOutput].matrix))','if(op.auxiliaryOutput!=SIZE_MAX&&!member(op.auxiliaryOutput)&&(op.auxiliaryOutput==inputs_[part]||source.SharesStorage(slots_[op.auxiliaryOutput].matrix)))')
s+='''\nnamespace Garnet {
namespace {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
constexpr bool kResidentConstruction=true;
#else
constexpr bool kResidentConstruction=false;
#endif
}
bool MetalCompiledGraph::BuildBF16Resident(X::Value graph,X::Value symbols,const std::string& checkpoint){std::lock_guard<std::mutex> lock(mutex_);return BuildInternal(std::move(graph),std::move(symbols),checkpoint,{},true,kResidentConstruction);}
bool MetalCompiledGraph::BuildSharedResident(X::Value graph,X::Value symbols,const std::string& checkpoint,X::Value source){return BuildSharedInternal(std::move(graph),std::move(symbols),checkpoint,std::move(source),kResidentConstruction);}
bool MetalCompiledGraph::BuildShared(X::Value graph,X::Value symbols,const std::string& checkpoint,X::Value source){return BuildSharedInternal(std::move(graph),std::move(symbols),checkpoint,std::move(source),false);}
size_t MetalCompiledGraph::ResidentAvoidedStorageBytes()const{std::lock_guard<std::mutex> lock(mutex_);return residentAvoidedStorageBytes_;}
}\n''';p.write_text(s)
p=source/'xModel/qwen35/vl_0_8b/metal_decoder.py';s=p.read_text().replace('if self.bf16_projections:graph.build_bf16(output,symbols,checkpoint)','if self.bf16_projections:graph.build_bf16_resident(output,symbols,checkpoint)').replace('else:graph.build_shared(output,symbols,self.checkpoint,layer_graph)','else:graph.build_shared_resident(output,symbols,self.checkpoint,layer_graph)');p.write_text(s)
p=source/'runtime/metal_serving.py';s=p.read_text().replace("        stages['lazy_default_retired_bytes']", "        stages['resident_avoided_storage_bytes'] = sum(graph.resident_avoided_storage_bytes for graph in memory_graphs)\n        stages['lazy_default_retired_bytes']",1);p.write_text(s)
p=source/'src/entry/metal_main.cpp';p.write_text(p.read_text().replace('; storage_lifetime_scopes','; storage_lifetime_scopes; resident_build_admission'))
shutil.copy2(w.parent/'worldsense-storage-scopes/before-libgarnet.dylib',w/'before-libgarnet.dylib')
(w/'implementation-status.json').write_text(json.dumps(dict(prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(source),native_admission=True,timed_samples=False),indent=2))
print('Prepared native resident admission prototype',flush=True)
