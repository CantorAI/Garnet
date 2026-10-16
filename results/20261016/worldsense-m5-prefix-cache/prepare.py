from pathlib import Path
import shutil,json,datetime
w=Path(__file__).resolve().parent;s=Path('/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source');shutil.copytree('/Users/shawn/CantorAI/out/research/worldsense-mixed-chunk/source',s)
p=s/'src/backends/metal/metal_context.h';t=p.read_text().replace('    void ZeroPrefixGroups(','''    std::vector<float> ReadPrefixGroups(const MetalMatrix&,size_t groups,size_t capacity,size_t width,size_t tokens);
    void WritePrefixGroups(const MetalMatrix&,size_t groups,size_t capacity,size_t width,size_t tokens,const std::vector<float>&);
    void ZeroPrefixGroups(''');p.write_text(t)
p=s/'src/backends/metal/metal_context.mm';t=p.read_text();needle='void MetalContext::ZeroPrefixGroups(';idx=t.index(needle)
code='''std::vector<float> MetalContext::ReadPrefixGroups(const MetalMatrix& matrix,size_t groups,size_t capacity,size_t width,size_t tokens) {
    if(impl_->command)throw std::logic_error("Prefix read requires a closed command batch");
    if(!matrix.impl_||matrix.impl_->matrix.device!=impl_->device)throw std::invalid_argument("Invalid prefix read matrix/device");
    if(matrix.impl_->pending||!matrix.impl_->ready)throw std::logic_error("Prefix read requires completed initialized state");
    const auto& native=matrix.impl_->matrix;
    if(!groups||!capacity||!width||!tokens||tokens>capacity||capacity>SIZE_MAX/width||
       groups>SIZE_MAX/(capacity*width)||groups*capacity*width!=native.columns||!native.data.contents)
        throw std::invalid_argument("Prefix read layout/bounds mismatch");
    std::vector<float> result(Elements(native.rows,groups*tokens*width));
    for(size_t row=0;row<native.rows;++row)for(size_t group=0;group<groups;++group)
        std::memcpy(result.data()+(row*groups+group)*tokens*width,
            static_cast<const char*>(native.data.contents)+row*native.rowBytes+group*capacity*width*sizeof(float),tokens*width*sizeof(float));
    return result;
}
void MetalContext::WritePrefixGroups(const MetalMatrix& matrix,size_t groups,size_t capacity,size_t width,size_t tokens,const std::vector<float>& values) {
    if(impl_->command)throw std::logic_error("Prefix write requires a closed command batch");
    if(!matrix.impl_||matrix.impl_->matrix.device!=impl_->device)throw std::invalid_argument("Invalid prefix write matrix/device");
    if(matrix.impl_->pending||!matrix.impl_->ready)throw std::logic_error("Prefix write requires completed initialized state");
    const auto& native=matrix.impl_->matrix;
    if(!groups||!capacity||!width||!tokens||tokens>capacity||capacity>SIZE_MAX/width||
       groups>SIZE_MAX/(capacity*width)||groups*capacity*width!=native.columns||!native.data.contents||
       values.size()!=Elements(native.rows,groups*tokens*width))throw std::invalid_argument("Prefix write layout/bounds mismatch");
    for(size_t row=0;row<native.rows;++row)for(size_t group=0;group<groups;++group)
        std::memcpy(static_cast<char*>(native.data.contents)+row*native.rowBytes+group*capacity*width*sizeof(float),
            values.data()+(row*groups+group)*tokens*width,tokens*width*sizeof(float));
    matrix.impl_->ready=true;
}
''';t=t[:idx]+code+t[idx:];p.write_text(t)
p=s/'src/backends/metal/decoder_session.h';t=p.read_text().replace('#include <array>','#include <array>\n#include <map>');t=t.replace('        size_t rows;','''        size_t rows;
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
        bool prefixEligible=false;uint64_t prefixOwnerGeneration=0;
#endif''')
needle='    void Clear();';t=t.replace(needle,'''#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    struct PrefixSnapshot {size_t tokens=0,bytes=0;uint64_t lastUse=0;std::vector<std::array<std::vector<float>,2>> layers;};
    std::map<int64_t,PrefixSnapshot> prefixCache_;
    size_t prefixBytes_=0,prefixRestoreBytes_=0,prefixCaptureBytes_=0;
    uint64_t prefixClock_=0;bool prefixFreshReset_=false;
    void ValidatePrefixOwners(std::vector<std::unique_lock<std::mutex>>& locks);
#endif
'''+needle)
t=t.replace('    void Reset();','''    void Reset();
    int64_t CapturePrefix(int64_t tokens);
    int64_t RestorePrefix(int64_t id);
    bool ClearPrefixCache();
    size_t PrefixCacheBytes();
    size_t PrefixCacheEntries();
    size_t PrefixRestoreBytes();
    size_t PrefixCaptureBytes();''')
t=t.replace('        APISET().AddFunc<0>("reset",&MetalDecoderSession::Reset);','''        APISET().AddFunc<0>("reset",&MetalDecoderSession::Reset);
        APISET().AddFunc<1>("capture_prefix",&MetalDecoderSession::CapturePrefix);
        APISET().AddFunc<1>("restore_prefix",&MetalDecoderSession::RestorePrefix);
        APISET().AddFunc<0>("clear_prefix_cache",&MetalDecoderSession::ClearPrefixCache);
        APISET().AddProp("prefix_cache_bytes",&MetalDecoderSession::PrefixCacheBytes);
        APISET().AddProp("prefix_cache_entries",&MetalDecoderSession::PrefixCacheEntries);
        APISET().AddProp("prefix_restore_bytes",&MetalDecoderSession::PrefixRestoreBytes);
        APISET().AddProp("prefix_capture_bytes",&MetalDecoderSession::PrefixCaptureBytes);''');p.write_text(t)
p=s/'src/backends/metal/decoder_session.cpp';t=p.read_text().replace('#include <stdexcept>','#include <stdexcept>\n#include <atomic>\n#include <limits>')
t=t.replace('void MetalDecoderSession::Clear() {','''void MetalDecoderSession::Clear() {
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    prefixCache_.clear();prefixBytes_=prefixRestoreBytes_=prefixCaptureBytes_=0;prefixClock_=0;prefixFreshReset_=false;
#endif''')
t=t.replace('void MetalDecoderSession::ResetUnlocked() {','''void MetalDecoderSession::ResetUnlocked() {
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    prefixFreshReset_=false;prefixRestoreBytes_=prefixCaptureBytes_=0;
#endif''')
needle='    hidden_=MetalMatrix();hiddenShape_.clear();kvInplaceUpdates_=0;poisoned_=false;inferenceMs_=0;prepareMs_=encodeMs_=gpuMs_=waitMs_=0;';assert needle in t;t=t.replace(needle,needle+'''
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    prefixFreshReset_=true;
#endif''')
t=t.replace('            states_.push_back(std::move(state));owners_.push_back(std::move(owner));','''#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
            state.prefixEligible=batch==1&&graph->CanResetKVPrefix(1)&&graph->CanResetKVPrefix(2);
            state.prefixOwnerGeneration=graph->generation_;
#endif
            states_.push_back(std::move(state));owners_.push_back(std::move(owner));''')
needle='    kvInplaceUpdates_=0;constantHeadDispatches_=';assert t.count(needle)==1;t=t.replace(needle,'''#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    prefixFreshReset_=false;
#endif
'''+needle)
# Cache implementation remains session-private and never exposes state vectors to XLang3.
t+='''
namespace Garnet {
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
namespace {
constexpr size_t kPrefixCacheBytes=64*1024*1024;
std::atomic<int64_t> nextPrefixId{1};
int64_t NewPrefixId() {
    auto id=nextPrefixId.load();
    for(;;){if(id<=0||id==std::numeric_limits<int64_t>::max())throw std::overflow_error("Prefix cache IDs exhausted");
        if(nextPrefixId.compare_exchange_weak(id,id+1))return id;}
}
}
void MetalDecoderSession::ValidatePrefixOwners(std::vector<std::unique_lock<std::mutex>>& locks) {
    std::vector<MetalCompiledGraph*> graphs;
    for(size_t i=0;i<states_.size();++i)graphs.push_back(Graph(owners_.at(i),Host()));
    auto ordered=graphs;if(head_)ordered.push_back(head_);std::sort(ordered.begin(),ordered.end(),std::less<MetalCompiledGraph*>());
    if(std::adjacent_find(ordered.begin(),ordered.end())!=ordered.end())throw std::invalid_argument("Prefix owners must be distinct");
    for(auto graph:ordered)locks.emplace_back(graph->mutex_);
    for(size_t i=0;i<states_.size();++i){const auto& state=states_[i];auto graph=graphs[i];
        if(!state.prefixEligible||!graph->ready_||graph->generation_!=state.prefixOwnerGeneration)
            throw std::logic_error("Prefix cache requires unchanged dense-attention graph owners");
        for(size_t k=0;k<2;++k){const auto& shape=state.shape[k];
            if(shape.size()!=4||shape[0]!=1||shape[1]<=0||shape[2]<64||shape[3]<=0||state.rows!=1)
                throw std::invalid_argument("Prefix cache requires batch-one dense KV layout");}}
    if(head_&&(!head_->ready_||head_->generation_!=headGeneration_))throw std::logic_error("Prefix head owner changed");
}
#endif
int64_t MetalDecoderSession::CapturePrefix(int64_t tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
#ifndef GARNET_METAL_DENSE_PREFIX_CACHE
    (void)tokens;throw std::logic_error("Dense prefix cache is not enabled");
#else
    if(!gpu_||poisoned_||prefixFreshReset_||states_.empty())throw std::logic_error("Capture requires a completed nonempty inference prefix");
    if(tokens!=64)throw std::invalid_argument("Only the validated64-token prefix is admitted");
    std::vector<std::unique_lock<std::mutex>> locks;ValidatePrefixOwners(locks);
    PrefixSnapshot snapshot;snapshot.tokens=64;
    for(const auto& state:states_)for(size_t k=0;k<2;++k){const auto& shape=state.shape[k];
        const auto count=size_t(shape[1])*64*size_t(shape[3]);
        if(count>kPrefixCacheBytes/sizeof(float)||snapshot.bytes>kPrefixCacheBytes-count*sizeof(float))
            throw std::invalid_argument("Prefix cache byte budget exceeded");
        snapshot.bytes+=count*sizeof(float);}
    for(const auto& state:states_){std::array<std::vector<float>,2> layer;
        for(size_t k=0;k<2;++k){const auto& shape=state.shape[k];layer[k]=gpu_->ReadPrefixGroups(state.matrix[k],size_t(shape[1]),size_t(shape[2]),size_t(shape[3]),64);
            if(!std::all_of(layer[k].begin(),layer[k].end(),[](float v){return std::isfinite(v);}))throw std::runtime_error("Cannot cache nonfinite KV state");}
        snapshot.layers.push_back(std::move(layer));}
    const auto id=NewPrefixId();snapshot.lastUse=++prefixClock_;const auto bytes=snapshot.bytes;
    prefixCache_.emplace(id,std::move(snapshot));prefixBytes_+=bytes;
    while(prefixCache_.size()>2||prefixBytes_>kPrefixCacheBytes){auto oldest=std::min_element(prefixCache_.begin(),prefixCache_.end(),[](const auto& a,const auto& b){return a.second.lastUse<b.second.lastUse;});prefixBytes_-=oldest->second.bytes;prefixCache_.erase(oldest);}
    prefixCaptureBytes_+=bytes;return id;
#endif
}
int64_t MetalDecoderSession::RestorePrefix(int64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
#ifndef GARNET_METAL_DENSE_PREFIX_CACHE
    (void)id;throw std::logic_error("Dense prefix cache is not enabled");
#else
    if(!gpu_||poisoned_||!prefixFreshReset_)throw std::logic_error("Restore requires a fresh successful full reset");
    auto found=prefixCache_.find(id);if(found==prefixCache_.end())throw std::invalid_argument("Unknown, evicted or foreign prefix handle");
    std::vector<std::unique_lock<std::mutex>> locks;ValidatePrefixOwners(locks);auto& snapshot=found->second;
    if(snapshot.layers.size()!=states_.size()||snapshot.tokens!=64)throw std::logic_error("Prefix metadata changed");
    for(size_t i=0;i<states_.size();++i)for(size_t k=0;k<2;++k){const auto& shape=states_[i].shape[k];
        if(snapshot.layers[i][k].size()!=size_t(shape[1])*64*size_t(shape[3]))throw std::logic_error("Prefix shape changed");}
    // All owners/layouts/counts are proved before any session state is written.
    prefixFreshReset_=false;poisoned_=true;
    try {for(size_t i=0;i<states_.size();++i)for(size_t k=0;k<2;++k){const auto& shape=states_[i].shape[k];
        gpu_->WritePrefixGroups(states_[i].matrix[k],size_t(shape[1]),size_t(shape[2]),size_t(shape[3]),64,snapshot.layers[i][k]);}
        poisoned_=false;snapshot.lastUse=++prefixClock_;prefixRestoreBytes_+=snapshot.bytes;return 64;
    }catch(...){throw;}
#endif
}
bool MetalDecoderSession::ClearPrefixCache() {std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    prefixCache_.clear();prefixBytes_=prefixRestoreBytes_=prefixCaptureBytes_=0;
#endif
    return true;
}
size_t MetalDecoderSession::PrefixCacheBytes(){std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    return prefixBytes_;
#else
    return 0;
#endif
}
size_t MetalDecoderSession::PrefixCacheEntries(){std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    return prefixCache_.size();
#else
    return 0;
#endif
}
size_t MetalDecoderSession::PrefixRestoreBytes(){std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    return prefixRestoreBytes_;
#else
    return 0;
#endif
}
size_t MetalDecoderSession::PrefixCaptureBytes(){std::lock_guard<std::mutex> lock(mutex_);
#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
    return prefixCaptureBytes_;
#else
    return 0;
#endif
}
}
''';p.write_text(t)
p=s/'cmake/GarnetMetal.cmake';t=p.read_text().replace('option(GARNET_METAL_MLP_PREFILL_GRAPH', '''option(GARNET_METAL_DENSE_PREFIX_CACHE "Bounded native dense KV prefix snapshots" OFF)
if(GARNET_METAL_DENSE_PREFIX_CACHE)
    if(GARNET_METAL_DEFERRED_STATE_ZERO OR GARNET_METAL_DIRTY_PREFIX_RESET)
        message(FATAL_ERROR "Prefix snapshot prototype requires full immediate reset")
    endif()
    target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_DENSE_PREFIX_CACHE=1)
endif()
option(GARNET_METAL_MLP_PREFILL_GRAPH''');p.write_text(t)
p=s/'src/entry/metal_main.cpp';t=p.read_text();needle='#ifdef GARNET_METAL_BODY_SHAPE';idx=t.index(needle);t=t[:idx]+'''#ifdef GARNET_METAL_DENSE_PREFIX_CACHE
        info+="; dense_prefix_cache";
#endif
'''+t[idx:];p.write_text(t)
for name in ['controller.py','cases.json']:
 (w/name).write_text((w.parent/'worldsense-mixed-chunk'/name).read_text().replace('worldsense-mixed-chunk','worldsense-prefix-cache'))
shutil.copy2(w.parent/'worldsense-mixed-chunk/before-libgarnet.dylib',w/'before-libgarnet.dylib')
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-10','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Combined policy:4B VLM prefill64;1.7B reasoner prefill16 with exact native64-token KV prefix reuse, versus16/16 uncached. Two bounded session-private compact CPU snapshot entries,64MiB retained maximum. Every suffix/answer and image freshly inferred.','tradeoffs':'Cold prefix computation and native copy/finite scan; warm reset+restore; retained compact native CPU vectors, transient capture allocation and key/LRU overhead. Both policies remain warm in same process; no claim of cold speed or GPU-resident/zero-copy cache. No quantization/CPython image helper/CUDA change.','validation':'Native memory bounds/read-write bits/pending/abort plus session layout, fresh reset, eviction, ownership, stale owners, finite-state and failure recovery. All-state/token equality with changing suffixes and cache hits required before timing.','stage':'Fixed-car WorldSense ABBA screen four measured calls/policy/model after both modes warm. All prompts/image-Skill hashes/non-policy budgets exact; native restore positive on candidate text hits, zero on control, no VLM prefix reuse. Fresh full three-frame/query ABBA and BAAB workers required before promotion.','gates':{'4b_ttft_min_gain':.10,'4b_total_min_gain':.03,'both_decode_max_regression':.02,'text_total_ttft_max_regression':.02,'exact_outputs_states_prompts_nonpolicy_budgets':True,'actual_native_prefix_counters':True}},indent=2))
print('Native prefix prototype prepared, not compiled or measured')
