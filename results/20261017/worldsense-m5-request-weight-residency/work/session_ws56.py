from pathlib import Path
s=Path('/Users/shawn/CantorAI/out/research/worldsense-request-weight-residency/source');p=s/'src/backends/metal/decoder_session.h';t=p.read_text();needle='    void Clear();';assert t.count(needle)==1;t=t.replace(needle,'''#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
    bool requestScopeOpen_=false;
    std::map<MetalCompiledGraph*,uint64_t> requestGraphs_;
    std::vector<X::Value> requestGraphOwners_;
    uint64_t requestCacheHits_=0,requestCoverageChecks_=0,requestCoverageFallbacks_=0,requestLastGraphs_=0;
    void CheckRequestCoverage(const std::vector<MetalCompiledGraph*>&,X::Value graphs);
    void CancelRequestUnlocked() noexcept;
#endif
'''+needle,1)
needle='    bool Build(X::Value graphs,X::Value head);';assert t.count(needle)==1;t=t.replace(needle,'''    bool BeginRequestResidency(X::Value graphs);
    bool EndRequestResidency();
    std::string RequestResidencyStatsJSON() const;
'''+needle,1)
# Stats locks the same mutex as all other session operations.
t=t.replace('    std::mutex mutex_;','    mutable std::mutex mutex_;',1)
needle='        APISET().AddFunc<0>("close",&MetalDecoderSession::Close);';assert t.count(needle)==1;t=t.replace(needle,'''        APISET().AddFunc<1>("begin_request_residency",&MetalDecoderSession::BeginRequestResidency);
        APISET().AddFunc<0>("end_request_residency",&MetalDecoderSession::EndRequestResidency);
        APISET().AddProp("request_residency_stats_json",&MetalDecoderSession::RequestResidencyStatsJSON);
'''+needle,1);p.write_text(t)
p=s/'src/backends/metal/decoder_session.cpp';t=p.read_text().replace('#include <limits>','#include <limits>\n#include <exception>',1)
for anchor in ['void MetalDecoderSession::Clear() {','void MetalDecoderSession::ResetUnlocked() {']:
 assert t.count(anchor)==1;t=t.replace(anchor,anchor+'\n#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY\n    CancelRequestUnlocked();\n#endif',1)
anchor='''    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_)throw std::logic_error("Reset a closed/failed Metal session before running");''';assert t.count(anchor)==1
t=t.replace(anchor,'''    std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
    const int requestExceptions=std::uncaught_exceptions();int requestGuardToken=0;
    auto requestCleanup=[&](int*) noexcept {if(std::uncaught_exceptions()>requestExceptions)CancelRequestUnlocked();};
    std::unique_ptr<int,decltype(requestCleanup)> requestGuard(&requestGuardToken,requestCleanup);
#endif
    if(!gpu_||poisoned_)throw std::logic_error("Reset a closed/failed Metal session before running");''',1)
anchor='#ifdef GARNET_METAL_DIRTY_PREFIX_RESET\n    // Host controls were validated above.';assert t.count(anchor)==1;t=t.replace(anchor,'''#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
    CheckRequestCoverage(locked,graphs);
#endif
'''+anchor,1)
t+=r'''
namespace Garnet {
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
void MetalDecoderSession::CancelRequestUnlocked() noexcept {
    if(gpu_){try{gpu_->Abort();}catch(...){}}
    requestLastGraphs_=requestGraphs_.size();requestGraphs_.clear();requestGraphOwners_.clear();requestScopeOpen_=false;
}
void MetalDecoderSession::CheckRequestCoverage(const std::vector<MetalCompiledGraph*>& graphs,X::Value owners) {
    if(!requestScopeOpen_||!gpu_->RequestWeightResidencyActive())return;
    for(auto graph:graphs){
        auto found=requestGraphs_.find(graph);
        if(found!=requestGraphs_.end()&&found->second==graph->generation_){++requestCacheHits_;continue;}
        ++requestCoverageChecks_;
        std::vector<MetalBF16Weight> weights;for(const auto& slot:graph->slots_)if(slot.bf16.Valid())weights.push_back(slot.bf16);
        if(requestGraphs_.size()>=256||!gpu_->RequestWeightsCovered(weights)){
            ++requestCoverageFallbacks_;gpu_->CancelRequestWeightResidency(8);
            requestLastGraphs_=requestGraphs_.size();requestGraphs_.clear();requestGraphOwners_.clear();requestScopeOpen_=false;return;
        }
        bool retained=false;
        for(uint64_t i=0;i<owners.Size();++i){auto owner=owners.Get(i);if(Graph(owner,Host())==graph){requestGraphOwners_.push_back(owner);retained=true;break;}}
        if(!retained&&graph==head_&&!owners_.empty()){requestGraphOwners_.push_back(owners_.back());retained=true;}
        if(!retained)throw std::logic_error("Request graph owner missing");
        requestGraphs_[graph]=graph->generation_;
    }
}
#endif
bool MetalDecoderSession::BeginRequestResidency(X::Value graphs) {
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||requestScopeOpen_)throw std::logic_error("Request scope requires open reset non-nested session");
    if(graphs.runtime()!=Host()->runtime||!graphs.IsList()||graphs.Size()!=states_.size()||graphs.Size()>255)throw std::invalid_argument("Request layer list mismatch");
    std::vector<MetalCompiledGraph*> locked;
    for(uint64_t i=0;i<graphs.Size();++i)locked.push_back(Graph(graphs.Get(i),Host()));
    if(head_)locked.push_back(head_);std::sort(locked.begin(),locked.end(),std::less<MetalCompiledGraph*>());
    if(std::adjacent_find(locked.begin(),locked.end())!=locked.end())throw std::invalid_argument("Request graphs must be distinct");
    std::vector<std::unique_lock<std::mutex>> locks;for(auto graph:locked)locks.emplace_back(graph->mutex_);
    if(head_&&(!head_->ready_||head_->generation_!=headGeneration_))throw std::logic_error("Request head closed/rebuilt");
    std::vector<MetalBF16Weight> weights;
    for(auto graph:locked){if(!graph->ready_)throw std::invalid_argument("Request graph closed");for(const auto& slot:graph->slots_)if(slot.bf16.Valid())weights.push_back(slot.bf16);}
    requestCacheHits_=requestCoverageChecks_=requestCoverageFallbacks_=requestLastGraphs_=0;
    try {
        const bool admitted=gpu_->BeginRequestWeightResidency(weights);requestScopeOpen_=true;
        if(admitted){
            for(uint64_t i=0;i<graphs.Size();++i){auto owner=graphs.Get(i);auto graph=Graph(owner,Host());requestGraphOwners_.push_back(owner);requestGraphs_[graph]=graph->generation_;}
            if(head_){requestGraphOwners_.push_back(owners_.back());requestGraphs_[head_]=head_->generation_;}
        }
        return admitted;
    }catch(...){gpu_->EndRequestWeightResidency();requestGraphs_.clear();requestGraphOwners_.clear();requestScopeOpen_=false;throw;}
#else
    (void)graphs;return false;
#endif
}
bool MetalDecoderSession::EndRequestResidency() {
    std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
    if(gpu_)gpu_->EndRequestWeightResidency();
    if(requestScopeOpen_)requestLastGraphs_=requestGraphs_.size();
    requestGraphs_.clear();requestGraphOwners_.clear();requestScopeOpen_=false;
#endif
    return true;
}
std::string MetalDecoderSession::RequestResidencyStatsJSON() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto json=gpu_?gpu_->RequestWeightResidencyStatsJSON():std::string("{\"enabled\":false,\"active_bytes\":0,\"global_used_bytes\":0}");
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
    if(!json.empty()&&json.back()=='}')json.pop_back();
    json+=",\"coverage_cache_hits\":"+std::to_string(requestCacheHits_)+",\"coverage_checks\":"+std::to_string(requestCoverageChecks_)+",\"coverage_fallbacks\":"+std::to_string(requestCoverageFallbacks_)+",\"last_coverage_graphs\":"+std::to_string(requestScopeOpen_?requestGraphs_.size():requestLastGraphs_)+"}";
#endif
    return json;
}
}
''';p.write_text(t)
# Feature and test-only bounded hooks, both default OFF.
p=s/'cmake/GarnetMetal.cmake';t=p.read_text();t+='''
option(GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY "Bound raw BF16 residency across one complete native serving request" OFF)
option(GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS "Correctness-only 64KiB request residency budget and failure hooks" OFF)
if(GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY)
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY=1)
endif()
if(GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS)
 if(NOT GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY)
  message(FATAL_ERROR "Request residency test hooks require request scope residency")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS=1)
endif()
add_executable(garnet_request_residency_test test2026/metal/request_residency_test.cpp)
target_link_libraries(garnet_request_residency_test PRIVATE garnet_metal)
add_test(NAME metal_request_residency_reference COMMAND garnet_request_residency_test)
''';p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text();needle='        return info;';assert t.count(needle)==1;t=t.replace(needle,'#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY\n        info+="; request_scope_weight_residency";\n#endif\n'+needle,1);p.write_text(t)
# Preserve the exact inference expressions; add native begin/finally/end only.
p=s/'runtime/metal_serving.py';t=p.read_text();t=t.replace('_models = {}',"_request_scope_residency = 'request_scope_weight_residency' in garnet.backend_info()\n_models = {}",1)
a=t.index('        prediction = self._prefill_with_prefix(hidden, positions, token_ids, stages)');b=t.index('\n\n\ndef serve_model',a);old=t[a:b];assert old.count('return dict(input_prompt_sha256=')==1
body=old.replace('        return dict(input_prompt_sha256=', '        result = dict(input_prompt_sha256=',1)+'\n        return result'
wrapped='''        request_started = False
        result = None
        try:
            if _request_scope_residency:
                self.decoder.session.begin_request_residency([layer['graph'] for layer in self.decoder.layers])
                request_started = True
'''+''.join('    '+line+'\n'for line in body.splitlines())+'''        finally:
            if request_started:
                self.decoder.session.end_request_residency()
                stages['request_weight_residency'] = json.loads(self.decoder.session.request_residency_stats_json)
                if result is not None:
                    result['total_seconds'] = time.monotonic() - start
'''
t=t[:a]+wrapped.rstrip()+t[b:];p.write_text(t)
import ast;ast.parse(t)
print('Session coverage/lifetime API, native option guards and XLang3 request lifecycle written; not built/tested yet')
