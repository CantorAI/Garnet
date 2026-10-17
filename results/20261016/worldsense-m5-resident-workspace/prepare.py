from pathlib import Path
import shutil,json,datetime
w=Path(__file__).resolve().parent;seed=Path('/Users/shawn/CantorAI/out/research/worldsense-resident-admission/source');r=Path('/Users/shawn/CantorAI/out/research/worldsense-resident-workspace/source');assert not r.exists();shutil.copytree(seed,r,ignore=shutil.ignore_patterns('__pycache__','.git','results','docs'))
p=r/'src/backends/metal/resident_workspace_plan.h';p.write_text('''#pragma once
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <utility>
namespace Garnet {
struct WorkspaceInterval {size_t slot,rows,columns,first,last;};
struct WorkspaceBinding {size_t slot,rows,columns,first,last,color;};
inline std::vector<WorkspaceBinding> ColorWorkspace(std::vector<WorkspaceInterval> intervals) {
    std::sort(intervals.begin(),intervals.end(),[](const auto&a,const auto&b){return std::make_pair(a.first,a.slot)<std::make_pair(b.first,b.slot);});
    std::map<std::pair<size_t,size_t>,std::vector<size_t>> ends;std::set<size_t> seen;std::vector<WorkspaceBinding> result;
    for(const auto&i:intervals){if(!i.rows||!i.columns||i.first>i.last||!seen.insert(i.slot).second)throw std::invalid_argument("Invalid scratch interval");
        auto& colors=ends[{i.rows,i.columns}];size_t color=0;while(color<colors.size()&&colors[color]>=i.first)++color;
        if(color==colors.size())colors.push_back(i.last);else colors[color]=i.last;
        result.push_back({i.slot,i.rows,i.columns,i.first,i.last,color});}
    return result;
}
}
''')
p=r/'src/backends/metal/compiled_graph.h';s=p.read_text().replace('#include "metal_context.h"','#include "metal_context.h"\n#include "resident_workspace_plan.h"');s=s.replace('    void RetireResidentDefaults();','    bool workspacePlanReady_=false;\n    std::vector<WorkspaceBinding> workspacePlan_;\n    void BuildWorkspacePlan();\n    void RetireResidentDefaults();');p.write_text(s)
p=r/'src/backends/metal/compiled_graph.cpp';s=p.read_text().replace('void MetalCompiledGraph::Clear() {','void MetalCompiledGraph::Clear() {\n    workspacePlanReady_=false;workspacePlan_.clear();',1)
s=s.replace('    MaterializeDefault(output_);','    MaterializeDefault(output_);\n#ifdef GARNET_METAL_RESIDENT_WORKSPACE\n    for(const auto& entry:workspacePlan_)MaterializeDefault(entry.slot);\n#endif',1)
s=s.replace('if(std::find(lazyDefaultSlots_.begin(),lazyDefaultSlots_.end(),id)==lazyDefaultSlots_.end())throw std::logic_error("Missing non-deferred graph storage");','if(std::find(lazyDefaultSlots_.begin(),lazyDefaultSlots_.end(),id)==lazyDefaultSlots_.end()&&std::none_of(workspacePlan_.begin(),workspacePlan_.end(),[&](const auto& entry){return entry.slot==id;}))throw std::logic_error("Missing non-deferred graph storage");')
s+='''\nnamespace Garnet {
void MetalCompiledGraph::BuildWorkspacePlan() {
#ifdef GARNET_METAL_RESIDENT_WORKSPACE
    if(workspacePlanReady_)return;if(!ready_)throw std::logic_error("Closed graph has no workspace plan");
    if(outputParts_.size()!=3||residentOps_.size()!=ops_.size()){workspacePlanReady_=true;return;}
    const auto n=slots_.size();std::vector<size_t> writers(n,0),first(n,SIZE_MAX),readFirst(n,SIZE_MAX),last(n,0);std::vector<bool> full(n,false),blocked(n,false);
    std::vector<MetalBufferInfo> infos;std::unordered_map<uintptr_t,size_t> counts;std::unordered_set<uintptr_t> protectedBuffers;
    for(const auto& slot:slots_){auto info=MetalContext::MatrixBufferInfo(slot.matrix);infos.push_back(info);if(info.identity)++counts[info.identity];for(const auto& b:MetalContext::WeightBufferInfo(slot.bf16))protectedBuffers.insert(b.identity);}
    auto protect=[&](size_t id){if(id<n){blocked[id]=true;if(infos[id].identity)protectedBuffers.insert(infos[id].identity);}};
    for(auto id:inputs_)protect(id);for(auto id:outputParts_)protect(id);protect(output_);for(const auto& group:outputStorageGroups_)for(auto id:group)protect(id);
    for(const auto& name:weights_)protect(name.second);
    for(const auto* records:{&sharedWeights_,&weightRecords_})for(const auto& record:*records){auto info=MetalContext::MatrixBufferInfo(record.second.slot.matrix);if(info.identity)protectedBuffers.insert(info.identity);for(const auto& b:MetalContext::WeightBufferInfo(record.second.slot.bf16))protectedBuffers.insert(b.identity);}
    auto overwrite=[](const Op& op){switch(op.kind){case Kind::MatMul:case Kind::BF16MatVec:case Kind::Norm:case Kind::Silu:case Kind::Add:case Kind::Bias:case Kind::Mask:case Kind::Slice:return true;case Kind::Layout:return op.dimensions.size()==1&&op.dimensions[0]==0;default:return false;}};
    for(size_t i=0;i<ops_.size();++i)if(residentOps_[i]){const auto& op=ops_[i];
        auto read=[&](size_t id){if(id<n){readFirst[id]=std::min(readFirst[id],i);last[id]=std::max(last[id],i);}};
        read(op.a);read(op.b);for(auto id:op.arguments)read(id);
        auto write=[&](size_t id,bool complete){if(id<n){++writers[id];first[id]=std::min(first[id],i);last[id]=std::max(last[id],i);full[id]=complete;}};
        write(op.out,overwrite(op));write(op.auxiliaryOutput,false);
    }
    std::vector<WorkspaceInterval> candidates;
    for(size_t id=0;id<n;++id){const auto& info=infos[id];const auto& slot=slots_[id];
        if(blocked[id]||!info.identity||counts[info.identity]!=1||protectedBuffers.count(info.identity)||writers[id]!=1||!full[id]||readFirst[id]<=first[id]||!gpu_->IsDefaultLayout(slot.matrix,slot.rows,slot.columns))continue;
        candidates.push_back({id,slot.rows,slot.columns,first[id],last[id]});}
    auto plan=ColorWorkspace(std::move(candidates));workspacePlan_=std::move(plan);workspacePlanReady_=true;
#endif
}
}\n''';p.write_text(s)
p=r/'src/backends/metal/decoder_session.h';s=p.read_text();needle='    bool poisoned_=false;';assert needle in s;s=s.replace(needle,'''    std::map<std::pair<size_t,size_t>,std::vector<MetalMatrix>> workspacePool_;
    size_t workspaceBoundSlots_=0;
    void EnsureWorkspaces(const std::vector<MetalCompiledGraph*>&);
'''+needle);s=s.replace('public:', 'public:\n    size_t WorkspaceAllocatedBytes() const;\n    size_t WorkspaceBufferCount() const;\n    size_t WorkspaceBoundSlots() const;',1);s=s.replace('        APISET().AddFunc<6>("run",&MetalDecoderSession::Run);','        APISET().AddFunc<6>("run",&MetalDecoderSession::Run);\n        APISET().AddProp("workspace_allocated_bytes",&MetalDecoderSession::WorkspaceAllocatedBytes);\n        APISET().AddProp("workspace_buffer_count",&MetalDecoderSession::WorkspaceBufferCount);\n        APISET().AddProp("workspace_bound_slots",&MetalDecoderSession::WorkspaceBoundSlots);');s=s.replace('    std::mutex mutex_;','    mutable std::mutex mutex_;');p.write_text(s)
p=r/'src/backends/metal/decoder_session.cpp';s=p.read_text().replace('void MetalDecoderSession::Clear() {','void MetalDecoderSession::Clear() {\n    workspacePool_.clear();workspaceBoundSlots_=0;',1)
s=s.replace('    for(auto graph:layers)graph->RetireResidentDefaults();','    EnsureWorkspaces(layers);\n    for(auto graph:layers)graph->RetireResidentDefaults();',1)
needle='            auto graph=layers[i];std::array<MetalMatrix,3> saved;';assert needle in s;s=s.replace(needle,needle+'''
            std::vector<std::pair<size_t,MetalMatrix>> scratch;
#ifdef GARNET_METAL_RESIDENT_WORKSPACE
            scratch.reserve(graph->workspacePlan_.size());
            for(const auto& entry:graph->workspacePlan_)scratch.emplace_back(entry.slot,workspacePool_.at({entry.rows,entry.columns}).at(entry.color));
#endif
''',1)
s=s.replace('            auto restore=[&](){','            for(const auto& entry:scratch)graph->slots_[entry.first].matrix=entry.second;\n            auto restore=[&](){',1)
s=s.replace('                for(auto& entry:savedOutputs)graph->slots_[entry.first].matrix=entry.second;','                for(auto& entry:savedOutputs)graph->slots_[entry.first].matrix=entry.second;\n                for(const auto& entry:scratch)graph->slots_[entry.first].matrix=MetalMatrix();',1)
s+='''\nnamespace Garnet {
void MetalDecoderSession::EnsureWorkspaces(const std::vector<MetalCompiledGraph*>& graphs) {
    workspaceBoundSlots_=0;
#ifdef GARNET_METAL_RESIDENT_WORKSPACE
    std::map<std::pair<size_t,size_t>,size_t> required;
    for(auto graph:graphs){graph->BuildWorkspacePlan();workspaceBoundSlots_+=graph->workspacePlan_.size();for(const auto& entry:graph->workspacePlan_)required[{entry.rows,entry.columns}]=std::max(required[{entry.rows,entry.columns}],entry.color+1);}
    for(const auto& entry:required){auto& buffers=workspacePool_[entry.first];while(buffers.size()<entry.second)buffers.push_back(gpu_->Allocate(entry.first.first,entry.first.second));}
#endif
}
size_t MetalDecoderSession::WorkspaceAllocatedBytes() const {std::lock_guard<std::mutex> lock(mutex_);size_t bytes=0;for(const auto& entry:workspacePool_)for(const auto& matrix:entry.second)bytes+=MetalContext::MatrixBufferInfo(matrix).allocatedBytes;return bytes;}
size_t MetalDecoderSession::WorkspaceBufferCount() const {std::lock_guard<std::mutex> lock(mutex_);size_t count=0;for(const auto& entry:workspacePool_)count+=entry.second.size();return count;}
size_t MetalDecoderSession::WorkspaceBoundSlots() const {std::lock_guard<std::mutex> lock(mutex_);return workspaceBoundSlots_;}
}\n''';p.write_text(s)
p=r/'cmake/GarnetMetal.cmake';s=p.read_text();s+='''\noption(GARNET_METAL_RESIDENT_WORKSPACE "Session-private exact-layout resident scratch reuse" OFF)
if(GARNET_METAL_RESIDENT_WORKSPACE)
 if(NOT GARNET_METAL_LAZY_RESIDENT_DEFAULTS)
  message(FATAL_ERROR "Resident workspace requires native deferred storage support")
 endif()
 foreach(policy GARNET_METAL_PACKED_PREFILL_NORM GARNET_METAL_RESIDUAL_NORM GARNET_METAL_MLP_PREFILL_GRAPH GARNET_METAL_JOINED_GLOBAL_MLP GARNET_METAL_JOINED_PREFILL)
  if(${policy})
   message(FATAL_ERROR "Resident workspace has not admitted multi-operation fusion: ${policy}")
  endif()
 endforeach()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_RESIDENT_WORKSPACE=1)
endif()
''';p.write_text(s)
p=r/'src/entry/metal_main.cpp';s=p.read_text().replace('        info+="; lazy_resident_defaults;', '        info+="; lazy_resident_defaults;');needle='        info+="; lazy_resident_defaults; storage_lifetime_scopes; resident_build_admission";';assert needle in s;s=s.replace(needle,needle+'\n#ifdef GARNET_METAL_RESIDENT_WORKSPACE\n        info+="; resident_scratch_workspace";\n#endif');p.write_text(s)
p=r/'runtime/metal_serving.py';s=p.read_text();needle="        stages['resident_avoided_storage_bytes']";assert needle in s;s=s.replace(needle,"        stages['workspace_allocated_bytes'] = self.decoder.session.workspace_allocated_bytes\n        stages['workspace_buffer_count'] = self.decoder.session.workspace_buffer_count\n        stages['workspace_bound_slots'] = self.decoder.session.workspace_bound_slots\n"+needle,1);p.write_text(s)
shutil.copy2(w.parent/'worldsense-resident-admission/before-libgarnet.dylib',w/'before-libgarnet.dylib')
(w/'implementation-status.json').write_text(json.dumps(dict(prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prototype_written=True,tests_run=False,performance_samples=False),indent=2));print('WS18 prototype prepared; unvalidated',flush=True)
