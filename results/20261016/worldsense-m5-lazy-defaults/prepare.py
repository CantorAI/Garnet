from pathlib import Path
import shutil,json,datetime
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-lazy-defaults/source')
shutil.copytree('/Users/shawn/CantorAI/out/research/worldsense-inventory/source',s,ignore=shutil.ignore_patterns('__pycache__','.git','results','docs'))
p=s/'src/backends/metal/metal_context.h';t=p.read_text().replace('    bool SharesStorage(const MetalMatrix&) const;', '    bool Valid() const {return bool(impl_);}\n    bool SharesStorage(const MetalMatrix&) const;').replace('    static MetalBufferInfo MatrixBufferInfo(const MetalMatrix&);','    static MetalBufferInfo MatrixBufferInfo(const MetalMatrix&);\n    bool IsDefaultLayout(const MetalMatrix&,size_t rows,size_t columns) const;');p.write_text(t)
p=s/'src/backends/metal/metal_context.mm';t=p.read_text()
# Bound buffer getters for retired defaults; disabled build preserves original behavior.
def pool_function(text,needle):
 a=text.index(needle);start=text.index('{',a);depth=1;i=start+1
 while depth:
  if text[i]=='{':depth+=1
  elif text[i]=='}':depth-=1
  i+=1
 return text[:start+1]+'\n#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n    @autoreleasepool {\n#endif'+text[start+1:i-1]+'\n#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n    }\n#endif\n'+text[i-1:]
for name in ['bool MetalMatrix::SharesStorage(', 'bool MetalMatrix::CompatibleLayout(', 'void MetalContext::Update(const MetalMatrix& matrix,const float*', 'void MetalContext::Zero(const MetalMatrix& matrix)', 'std::vector<float> MetalContext::Download(', 'std::vector<float> MetalContext::ReadPrefixGroups(', 'void MetalContext::WritePrefixGroups(']:t=pool_function(t,name)
idx=t.index('MetalBufferInfo MetalContext::MatrixBufferInfo(');t=t[:idx]+'''bool MetalContext::IsDefaultLayout(const MetalMatrix& value,size_t rows,size_t columns) const {
    @autoreleasepool {
        if(!value.impl_||!value.impl_->matrix)return false;auto matrix=value.impl_->matrix;
        const auto stride=[MPSMatrixDescriptor rowBytesFromColumns:columns dataType:MPSDataTypeFloat32];
        return matrix.device==impl_->device&&matrix.rows==rows&&matrix.columns==columns&&matrix.rowBytes==stride&&matrix.offset==0&&matrix.data.length==rows*stride;
    }
}
'''+t[idx:];p.write_text(t)
p=s/'src/backends/metal/compiled_graph.h';t=p.read_text().replace('    void Clear();','''#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    bool lazyDefaultPlanReady_=false;std::vector<size_t> lazyDefaultSlots_;
    size_t lazyRetiredBytes_=0,lazyRehydratedBytes_=0;
#endif
    void RetireResidentDefaults();
    void MaterializeDefault(size_t slot);
    void Clear();''',1);t=t.replace('    std::vector<MetalResourceReference> MemoryReferences();','''    std::vector<MetalResourceReference> MemoryReferences();
    size_t LazyRetiredBytes() const;
    size_t LazyRehydratedBytes() const;
    size_t LazyPlannedSlots() const;''');t=t.replace('        APISET().AddProp("ready"','''        APISET().AddProp("lazy_default_retired_bytes",&MetalCompiledGraph::LazyRetiredBytes);
        APISET().AddProp("lazy_default_rehydrated_bytes",&MetalCompiledGraph::LazyRehydratedBytes);
        APISET().AddProp("lazy_default_planned_slots",&MetalCompiledGraph::LazyPlannedSlots);
        APISET().AddProp("ready"''');p.write_text(t)
p=s/'src/backends/metal/compiled_graph.cpp';t=p.read_text().replace('void MetalCompiledGraph::Clear() {','''void MetalCompiledGraph::Clear() {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    lazyDefaultPlanReady_=false;lazyDefaultSlots_.clear();lazyRetiredBytes_=lazyRehydratedBytes_=0;
#endif''',1)
# Preserve source identity checks even when an input's default allocation is absent.
t=t.replace('if(source.SharesStorage(slots_[op.a].matrix)||source.SharesStorage(slots_[op.b].matrix))return false;', 'if(op.a==inputs_[part]||op.b==inputs_[part]||source.SharesStorage(slots_[op.a].matrix)||source.SharesStorage(slots_[op.b].matrix))return false;')
t=t.replace('for(auto id:op.arguments)if(source.SharesStorage(slots_[id].matrix))return false;', 'for(auto id:op.arguments)if(id==inputs_[part]||source.SharesStorage(slots_[id].matrix))return false;')
needle='        gpu_->Update(slot.matrix,static_cast<const float*>(info.data),slot.rows*slot.columns);';assert needle in t;t=t.replace(needle,'        MaterializeDefault(inputs_[i]);\n'+needle,1)
t=t.replace('    PrepareInputs(inputs);\n    gpu_->Begin();','    PrepareInputs(inputs);\n    MaterializeDefault(output_);\n    gpu_->Begin();',1)
t+='''
namespace Garnet {
void MetalCompiledGraph::RetireResidentDefaults() {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    if(!lazyDefaultPlanReady_) {
        if(!ready_||outputParts_.size()!=3||inputs_.size()<3||residentOps_.size()!=ops_.size())return;
        std::vector<MetalBufferInfo> infos;std::unordered_map<uintptr_t,size_t> counts;std::unordered_set<uintptr_t> protectedWeights;
        for(const auto& slot:slots_){auto info=MetalContext::MatrixBufferInfo(slot.matrix);infos.push_back(info);if(info.identity)++counts[info.identity];for(const auto& b:MetalContext::WeightBufferInfo(slot.bf16))protectedWeights.insert(b.identity);}
        for(const auto& record:weightRecords_){auto info=MetalContext::MatrixBufferInfo(record.second.slot.matrix);if(info.identity)protectedWeights.insert(info.identity);for(const auto& b:MetalContext::WeightBufferInfo(record.second.slot.bf16))protectedWeights.insert(b.identity);}
        auto eligible=[&](size_t id){const auto& slot=slots_[id];const auto& info=infos[id];return info.identity&&counts[info.identity]==1&&!protectedWeights.count(info.identity)&&gpu_->IsDefaultLayout(slot.matrix,slot.rows,slot.columns);};
        std::vector<size_t> plan;
        for(size_t part=1;part<=2;++part){auto id=inputs_[part];if(inputShapes_[part].size()<3||std::find(outputParts_.begin(),outputParts_.end(),id)!=outputParts_.end())continue;if(eligible(id))plan.push_back(id);}
        if(!ops_.empty()&&ops_.back().kind==Kind::Concat&&ops_.back().out==output_&&!residentOps_.back()&&std::find(inputs_.begin(),inputs_.end(),output_)==inputs_.end()&&std::find(outputParts_.begin(),outputParts_.end(),output_)==outputParts_.end()&&eligible(output_))plan.push_back(output_);
        lazyDefaultSlots_=std::move(plan);lazyDefaultPlanReady_=true;
    }
    for(auto id:lazyDefaultSlots_)if(slots_[id].matrix.Valid()) {auto info=MetalContext::MatrixBufferInfo(slots_[id].matrix);lazyRetiredBytes_+=info.allocatedBytes;slots_[id].matrix=MetalMatrix();}
#endif
}
void MetalCompiledGraph::MaterializeDefault(size_t id) {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    if(slots_[id].matrix.Valid())return;
    if(std::find(lazyDefaultSlots_.begin(),lazyDefaultSlots_.end(),id)==lazyDefaultSlots_.end())throw std::logic_error("Missing non-deferred graph storage");
    auto matrix=gpu_->Allocate(slots_[id].rows,slots_[id].columns);auto info=MetalContext::MatrixBufferInfo(matrix);slots_[id].matrix=std::move(matrix);lazyRehydratedBytes_+=info.allocatedBytes;
#else
    (void)id;
#endif
}
size_t MetalCompiledGraph::LazyRetiredBytes() const {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    return lazyRetiredBytes_;
#else
    return 0;
#endif
}
size_t MetalCompiledGraph::LazyRehydratedBytes() const {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    return lazyRehydratedBytes_;
#else
    return 0;
#endif
}
size_t MetalCompiledGraph::LazyPlannedSlots() const {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    return lazyDefaultSlots_.size();
#else
    return 0;
#endif
}
}
''';p.write_text(t)
p=s/'src/backends/metal/decoder_session.cpp';t=p.read_text().replace('    if(hiddenShape_!=inputShape[0]) {','    for(auto graph:layers)graph->RetireResidentDefaults();\n    if(hiddenShape_!=inputShape[0]) {',1);p.write_text(t)
p=s/'cmake/GarnetMetal.cmake';t=p.read_text().replace('add_library(garnet_metal STATIC src/backends/metal/metal_context.mm)','''add_library(garnet_metal STATIC src/backends/metal/metal_context.mm)
option(GARNET_METAL_LAZY_RESIDENT_DEFAULTS "Defer singleton resident-unused layer defaults" OFF)
if(GARNET_METAL_LAZY_RESIDENT_DEFAULTS)
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_LAZY_RESIDENT_DEFAULTS=1)
endif()''',1);p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text().replace('#ifdef GARNET_METAL_DENSE_PREFIX_CACHE','#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS\n        info+="; lazy_resident_defaults";\n#endif\n#ifdef GARNET_METAL_DENSE_PREFIX_CACHE',1);p.write_text(t)
p=s/'runtime/metal_serving.py';t=p.read_text();a=t.index('        descriptors = [dict(object=layer');b=t.index("        stages['decode_profile']",a);block=t[a:b];t=t[:a]+'''        memory_graphs = [layer['graph'] for layer in self.decoder.layers] + list(self.decoder.prefill_graphs.values())
        stages['lazy_default_retired_bytes'] = sum(graph.lazy_default_retired_bytes for graph in memory_graphs)
        stages['lazy_default_rehydrated_bytes'] = sum(graph.lazy_default_rehydrated_bytes for graph in memory_graphs)
        stages['lazy_default_planned_slots'] = sum(graph.lazy_default_planned_slots for graph in memory_graphs)
        if self.collect_inventory:
'''+''.join('    '+line+'\n'for line in block.splitlines())+t[b:];t=t.replace('        self.root, self.name = root, name','        self.collect_inventory = bool(profile.get("collectNativeInventory", False))\n        self.root, self.name = root, name',1).replace('/worldsense-inventory/captures','/worldsense-lazy-defaults/captures');p.write_text(t)
shutil.copy2('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/libgarnet.dylib',w/'before-libgarnet.dylib')
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-15','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hypothesis':'Defer singleton layer default state input1/2 buffers and inactive terminal packed concat output during native resident execution. Generic graph.Run re-materializes required defaults; full admission, private state, prefix cache and arithmetic unchanged. Bounded host buffer getters allow retired allocations to release rather than survive in autorelease roots.','eligibility':'Layer graph3outputs, unique owned standard-layout allocation, exclude weight/alias/output-part owners. Only state inputrank>=3 or terminal nonresidentConcat root. Preserve slot metadata/generation and identical-slot alias detection when defaults empty.','validation':'OFF/ON native references; mixed ordinary/resident calls, multiple private sessions, closure/rebuild, malformed-input/failure recovery, exact finite32VLM/28text states/tokens and actual retire/regrow counters before timing.','measurement':'WorldSense only for performance: fresh separate car control/candidate screen after correctness, then fresh full three-frame/query control/candidate and mirrored candidate/control confirmation pairs, each independently passing. Source/shared telemetry match, only default-OFF lazy flag differs. All first/warm/setup preserved.','gates':{'sampled_metal_allocation_min_reduction':.10,'both_model_ttft_total_decode_max_regression':.02,'native_skill_wall_max_regression':.02,'exact_outputs_states_inputs_prompts_budgets_and_native_counters':True},'tradeoffs':'Targets allocation headroom, not presumed speed/physical RAM gain. Generic/resident alternation may allocate again; first retirement costs and partial allocation failure/recovery must be logged. Getter-pool cost belongs to candidate. No lower-context shortcut, precision change, external CPython image helper or CUDA change.'},indent=2))
print('Prepared opt-in lazy default storage candidate; not built or tested')
