from pathlib import Path
import shutil,json,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI/out/research/worldsense-graph-workspace/source');seed=Path('/Users/shawn/CantorAI/out/research/worldsense-resident-workspace/source');assert not r.exists();shutil.copytree(seed,r,ignore=shutil.ignore_patterns('__pycache__','.git','results','docs'))
p=r/'src/backends/metal/compiled_graph.h';s=p.read_text().replace('    bool workspacePlanReady_=false;','    std::map<std::pair<size_t,size_t>,std::vector<MetalMatrix>> graphWorkspacePool_;\n    size_t graphWorkspaceBoundSlots_=0;\n    void EnsureGraphWorkspace();\n    bool workspacePlanReady_=false;');s=s.replace('    std::string WorkspacePlanJson();','    std::string WorkspacePlanJson();\n    size_t GraphWorkspaceBytes() const;\n    size_t GraphWorkspaceCount() const;\n    size_t GraphWorkspaceSlots() const;');s=s.replace('        APISET().AddFunc<0>("workspace_plan_json"','        APISET().AddProp("graph_workspace_allocated_bytes",&MetalCompiledGraph::GraphWorkspaceBytes);\n        APISET().AddProp("graph_workspace_buffer_count",&MetalCompiledGraph::GraphWorkspaceCount);\n        APISET().AddProp("graph_workspace_bound_slots",&MetalCompiledGraph::GraphWorkspaceSlots);\n        APISET().AddFunc<0>("workspace_plan_json"');p.write_text(s)
p=r/'src/backends/metal/compiled_graph.cpp';s=p.read_text().replace('void MetalCompiledGraph::Clear() {','void MetalCompiledGraph::Clear() {\n    graphWorkspacePool_.clear();graphWorkspaceBoundSlots_=0;',1)
s=s.replace('    if(outputParts_.size()!=3||residentOps_.size()!=ops_.size()){workspacePlanReady_=true;return;}', '''    const bool resident=outputParts_.size()==3;
    bool ordinary=false;
#ifdef GARNET_METAL_GRAPH_WORKSPACE
    ordinary=outputParts_.empty();
#endif
    if((!resident&&!ordinary)||(resident&&residentOps_.size()!=ops_.size())){workspacePlanReady_=true;return;}''',1)
s=s.replace('for(size_t i=0;i<ops_.size();++i)if(residentOps_[i]){const auto& op=ops_[i];','for(size_t i=0;i<ops_.size();++i)if(ordinary||residentOps_[i]){const auto& op=ops_[i];',1)
old='''#ifdef GARNET_METAL_RESIDENT_WORKSPACE
    for(const auto& entry:workspacePlan_)MaterializeDefault(entry.slot);
#endif
    gpu_->Begin();
    try {
        Encode(*gpu_);''';assert old in s
new='''    std::vector<std::pair<size_t,MetalMatrix>> scratch;
    graphWorkspaceBoundSlots_=0;
#ifdef GARNET_METAL_GRAPH_WORKSPACE
    if(outputParts_.empty()) {
        BuildWorkspacePlan();EnsureGraphWorkspace();scratch.reserve(workspacePlan_.size());
        for(const auto& entry:workspacePlan_)scratch.emplace_back(entry.slot,graphWorkspacePool_.at({entry.rows,entry.columns}).at(entry.color));
        graphWorkspaceBoundSlots_=scratch.size();
    } else
#endif
    {
#ifdef GARNET_METAL_RESIDENT_WORKSPACE
        for(const auto& entry:workspacePlan_)MaterializeDefault(entry.slot);
#endif
    }
    auto restore=[&](){for(const auto& entry:scratch)slots_[entry.first].matrix=MetalMatrix();};
    for(const auto& entry:scratch)slots_[entry.first].matrix=entry.second;
    try {
        gpu_->Begin();
        Encode(*gpu_);''';s=s.replace(old,new,1)
s=s.replace('    }catch(...){gpu_->Abort();throw;}\n    auto result=gpu_->Download(slots_[output_].matrix);','    }catch(...){gpu_->Abort();restore();throw;}\n    restore();\n    auto result=gpu_->Download(slots_[output_].matrix);',1)
s=s.replace('held(sharedWeights_,"shared_weight_record");held(weightRecords_,"weight_record");return result;','''held(sharedWeights_,"shared_weight_record");held(weightRecords_,"weight_record");
    for(const auto& entry:graphWorkspacePool_)for(size_t color=0;color<entry.second.size();++color){Slot slot;slot.matrix=entry.second[color];slot.rows=entry.first.first;slot.columns=entry.first.second;slot.shape={int64_t(slot.rows),int64_t(slot.columns)};append(slot,SIZE_MAX,"graph_workspace","",false,false,false,true,false);}
    return result;''',1)
s+='''\nnamespace Garnet {
void MetalCompiledGraph::EnsureGraphWorkspace() {
#ifdef GARNET_METAL_GRAPH_WORKSPACE
    std::map<std::pair<size_t,size_t>,size_t> required;for(const auto& e:workspacePlan_)required[{e.rows,e.columns}]=std::max(required[{e.rows,e.columns}],e.color+1);
    for(const auto& e:required){auto& buffers=graphWorkspacePool_[e.first];while(buffers.size()<e.second){auto matrix=gpu_->Allocate(e.first.first,e.first.second);if(!gpu_->HazardTracked(matrix))throw std::runtime_error("Ordinary workspace requires tracked resources");buffers.push_back(std::move(matrix));}}
#endif
}
size_t MetalCompiledGraph::GraphWorkspaceBytes()const{std::lock_guard<std::mutex> lock(mutex_);size_t n=0;for(const auto&e:graphWorkspacePool_)for(const auto&m:e.second)n+=MetalContext::MatrixBufferInfo(m).allocatedBytes;return n;}
size_t MetalCompiledGraph::GraphWorkspaceCount()const{std::lock_guard<std::mutex> lock(mutex_);size_t n=0;for(const auto&e:graphWorkspacePool_)n+=e.second.size();return n;}
size_t MetalCompiledGraph::GraphWorkspaceSlots()const{std::lock_guard<std::mutex> lock(mutex_);return graphWorkspaceBoundSlots_;}
}\n''';p.write_text(s)
p=r/'src/backends/metal/decoder_session.cpp';s=p.read_text().replace('    EnsureWorkspaces(layers);','''#ifdef GARNET_METAL_GRAPH_WORKSPACE
    // A head may previously have run through its ordinary graph-owned pool.
    // Direct head encoding needs independent scratch before this batch starts.
    if(head_)for(const auto& entry:head_->workspacePlan_)head_->MaterializeDefault(entry.slot);
#endif
    EnsureWorkspaces(layers);''',1);p.write_text(s)
p=r/'cmake/GarnetMetal.cmake';s=p.read_text()+'''\noption(GARNET_METAL_GRAPH_WORKSPACE "Graph-owned all-operation scratch reuse for ordinary graphs" OFF)
if(GARNET_METAL_GRAPH_WORKSPACE)
 if(NOT GARNET_METAL_RESIDENT_WORKSPACE)
  message(FATAL_ERROR "Ordinary workspace requires admitted workspace/deferred storage support")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_GRAPH_WORKSPACE=1)
endif()
''';p.write_text(s)
p=r/'src/entry/metal_main.cpp';s=p.read_text().replace('        info+="; resident_scratch_workspace";','        info+="; resident_scratch_workspace";\n#ifdef GARNET_METAL_GRAPH_WORKSPACE\n        info+="; ordinary_graph_workspace";\n#endif');p.write_text(s)
p=r/'runtime/metal_serving.py';s=p.read_text().replace("        stages['workspace_allocated_bytes']", "        stages['vision_workspace_allocated_bytes'] = self.vision.graph.graph_workspace_allocated_bytes if self.vision is not None else 0\n        stages['vision_workspace_buffer_count'] = self.vision.graph.graph_workspace_buffer_count if self.vision is not None else 0\n        stages['vision_workspace_bound_slots'] = self.vision.graph.graph_workspace_bound_slots if self.vision is not None else 0\n        stages['workspace_allocated_bytes']",1);p.write_text(s)
shutil.copy2(w.parent/'worldsense-resident-workspace/before-libgarnet.dylib',w/'before-libgarnet.dylib');(w/'implementation-status.json').write_text(json.dumps(dict(prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prototype_written=True,tests_passed=False,performance_samples=False),indent=2))
p=w/'build.py';s=(w.parent/'worldsense-resident-workspace/build.py').read_text().replace('worldsense-resident-workspace','worldsense-graph-workspace');s=s.replace("'-DGARNET_METAL_RESIDENT_WORKSPACE='+mode.upper()", "'-DGARNET_METAL_RESIDENT_WORKSPACE='+mode.upper(),'-DGARNET_METAL_GRAPH_WORKSPACE='+mode.upper()");p.write_text(s)
print('Ordinary graph workspace prototype prepared; unvalidated',flush=True)
