from pathlib import Path
s=Path('/Users/shawn/CantorAI/out/research/worldsense-request-weight-residency/source');p=s/'src/backends/metal/metal_context.h';t=p.read_text();needle='    void Begin();';assert t.count(needle)==1;t=t.replace(needle,'''    bool BeginRequestWeightResidency(const std::vector<MetalBF16Weight>&);
    void EndRequestWeightResidency();
    bool RequestWeightResidencyActive() const;
    bool RequestWeightsCovered(const std::vector<MetalBF16Weight>&) const;
    void CancelRequestWeightResidency(uint64_t reason);
    std::string RequestWeightResidencyStatsJSON() const;
#ifdef GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS
    static void RequestResidencyTestFailAfter(int);
#endif
'''+needle,1);p.write_text(t)
p=s/'src/backends/metal/metal_context.mm';t=p.read_text();t=t.replace('#include "metal_context.h"','#include "metal_context.h"\n#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY\n#include "weight_residency_budget.h"\n#include <atomic>\n#endif',1)
needle='    double gpuMs=0;';assert t.count(needle)==1;t=t.replace(needle,'''#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
    id<MTLResidencySet> requestResidencySet=nil;
    bool requestAttached=false,requestRequested=false,requestScopeOpen=false;
    uint64_t requestCalls=0,requestAdmitted=0,requestFallback=0,requestEnds=0;
    uint64_t requestLastBytes=0,requestLastCount=0,requestActiveBytes=0,requestAttemptedBytes=0,requestAttemptedCount=0,requestReason=0;
    std::vector<MetalBF16Weight> requestOwners;
    std::unordered_set<const void*> requestIdentities;
#endif
'''+needle,1)
needle='    impl_->outputs.clear();impl_->pendingOutputs.clear();impl_->command=nil;\n}\nstd::vector<float> MetalContext::Download';assert t.count(needle)==1;t=t.replace(needle,'''    impl_->outputs.clear();impl_->pendingOutputs.clear();impl_->command=nil;
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
    if(impl_->requestScopeOpen)CancelRequestWeightResidency(9);
#endif
}
std::vector<float> MetalContext::Download''',1)
t+=r'''
namespace Garnet {
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
namespace {
#ifdef GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS
constexpr uint64_t kRequestResidencyLimit=64*1024;
std::atomic<int> requestFailureStage{0};
void RequestMaybeFail(int stage){int expected=stage;if(requestFailureStage.compare_exchange_strong(expected,0))throw std::runtime_error("Injected request residency failure");}
#else
constexpr uint64_t kRequestResidencyLimit=uint64_t(4)*1024*1024*1024;
void RequestMaybeFail(int){}
#endif
WeightResidencyBudget& RequestBudget(){static WeightResidencyBudget value(kRequestResidencyLimit);return value;}
}
#endif
void MetalContext::EndRequestWeightResidency() {
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
 @autoreleasepool {
  if(impl_->command||impl_->submitted.count)throw std::logic_error("Drain or Abort before ending request residency");
  if(@available(macOS 15.0,*)) {
   if(impl_->requestResidencySet){
    if(impl_->requestAttached){[impl_->queue removeResidencySet:impl_->requestResidencySet];impl_->requestAttached=false;}
    if(impl_->requestRequested){[impl_->requestResidencySet endResidency];impl_->requestRequested=false;}
    [impl_->requestResidencySet removeAllAllocations];[impl_->requestResidencySet commit];
   }
  }
  impl_->requestOwners.clear();impl_->requestIdentities.clear();
  if(impl_->requestActiveBytes){RequestBudget().Release(impl_->device.registryID,impl_->requestActiveBytes);impl_->requestActiveBytes=0;}
  if(impl_->requestScopeOpen){++impl_->requestEnds;impl_->requestScopeOpen=false;}
 }
#endif
}
bool MetalContext::BeginRequestWeightResidency(const std::vector<MetalBF16Weight>& weights) {
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
 @autoreleasepool {
  if(impl_->command||impl_->submitted.count||impl_->requestScopeOpen)throw std::logic_error("Request residency requires a closed drained non-nested scope");
  ++impl_->requestCalls;impl_->requestScopeOpen=true;
  impl_->requestLastBytes=impl_->requestLastCount=impl_->requestAttemptedBytes=impl_->requestAttemptedCount=impl_->requestReason=0;
  auto fallback=[&](uint64_t reason){++impl_->requestFallback;impl_->requestReason=reason;};
  try {
   if(@available(macOS 15.0,*)) {
    NSMutableArray<id<MTLAllocation>>* allocations=[NSMutableArray new];std::unordered_set<const void*> seen;std::vector<MetalBF16Weight> owners;uint64_t bytes=0;
    for(const auto& weight:weights){
     if(weight.PrimaryReady()){fallback(7);return false;}
     if(!weight.impl_||!weight.impl_->buffer)throw std::invalid_argument("Invalid raw BF16 request weight");
     auto buffer=weight.impl_->buffer;if(buffer.device!=impl_->device)throw std::invalid_argument("Request weight device mismatch");
     auto identity=(__bridge const void*)buffer;if(!seen.insert(identity).second)continue;
     auto n=buffer.allocatedSize;if(n>UINT64_MAX-bytes)throw std::overflow_error("Request residency byte overflow");bytes+=n;
     impl_->requestAttemptedBytes=bytes;impl_->requestAttemptedCount=seen.size();
     if(seen.size()>2048||bytes>kRequestResidencyLimit){fallback(4);return false;}
     [allocations addObject:buffer];owners.push_back(weight);
    }
    if(!bytes){fallback(2);return false;}
    if(!RequestBudget().Acquire(impl_->device.registryID,bytes)){fallback(5);return false;}
    impl_->requestActiveBytes=bytes;
    RequestMaybeFail(1);
    impl_->requestOwners=std::move(owners);impl_->requestIdentities=std::move(seen);
    if(!impl_->requestResidencySet){auto descriptor=[MTLResidencySetDescriptor new];descriptor.initialCapacity=allocations.count;descriptor.label=@"Garnet bounded request BF16 weights";NSError* error=nil;impl_->requestResidencySet=[impl_->device newResidencySetWithDescriptor:descriptor error:&error];}
    if(!impl_->requestResidencySet){EndRequestWeightResidency();fallback(6);return false;}
    for(id<MTLAllocation> allocation in allocations)[impl_->requestResidencySet addAllocation:allocation];
    [impl_->requestResidencySet commit];[impl_->queue addResidencySet:impl_->requestResidencySet];impl_->requestAttached=true;
    RequestMaybeFail(2);
    [impl_->requestResidencySet requestResidency];impl_->requestRequested=true;
    ++impl_->requestAdmitted;impl_->requestLastBytes=bytes;impl_->requestLastCount=allocations.count;impl_->requestReason=1;return true;
   }else{fallback(3);return false;}
  }catch(...){EndRequestWeightResidency();throw;}
 }
#else
 (void)weights;return false;
#endif
}
bool MetalContext::RequestWeightResidencyActive() const {
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
 return impl_->requestActiveBytes!=0;
#else
 return false;
#endif
}
bool MetalContext::RequestWeightsCovered(const std::vector<MetalBF16Weight>& weights) const {
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
 if(!impl_->requestActiveBytes)return false;
 for(const auto& weight:weights){if(!weight.impl_||weight.PrimaryReady()||!weight.impl_->buffer||weight.impl_->buffer.device!=impl_->device||!impl_->requestIdentities.count((__bridge const void*)weight.impl_->buffer))return false;}
 return true;
#else
 (void)weights;return false;
#endif
}
void MetalContext::CancelRequestWeightResidency(uint64_t reason) {
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
 if(impl_->requestScopeOpen){++impl_->requestFallback;impl_->requestReason=reason;}
 EndRequestWeightResidency();
#else
 (void)reason;
#endif
}
std::string MetalContext::RequestWeightResidencyStatsJSON() const {
#ifdef GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY
 return "{\"enabled\":true,\"calls\":"+std::to_string(impl_->requestCalls)+",\"admitted\":"+std::to_string(impl_->requestAdmitted)+",\"fallback\":"+std::to_string(impl_->requestFallback)+",\"ends\":"+std::to_string(impl_->requestEnds)+",\"last_bytes\":"+std::to_string(impl_->requestLastBytes)+",\"last_count\":"+std::to_string(impl_->requestLastCount)+",\"active_bytes\":"+std::to_string(impl_->requestActiveBytes)+",\"global_used_bytes\":"+std::to_string(RequestBudget().Used(impl_->device.registryID))+",\"limit_bytes\":"+std::to_string(kRequestResidencyLimit)+",\"attempted_bytes\":"+std::to_string(impl_->requestAttemptedBytes)+",\"attempted_count\":"+std::to_string(impl_->requestAttemptedCount)+",\"reason\":"+std::to_string(impl_->requestReason)+",\"scope_open\":"+(impl_->requestScopeOpen?"true":"false")+"}";
#else
 return "{\"enabled\":false,\"calls\":0,\"admitted\":0,\"fallback\":0,\"ends\":0,\"active_bytes\":0,\"global_used_bytes\":0,\"scope_open\":false}";
#endif
}
#ifdef GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS
void MetalContext::RequestResidencyTestFailAfter(int stage){if(stage<0||stage>2)throw std::invalid_argument("Invalid request failure stage");requestFailureStage.store(stage);}
#endif
}
''';p.write_text(t)
print('Native bounded request-context implementation written; not built/tested yet')
