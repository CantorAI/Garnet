from pathlib import Path
import shutil
w=Path(__file__).resolve().parent;old=Path('/Users/shawn/CantorAI/out/research/worldsense-private-resident-defaults/source');new=old.parent.parent/'worldsense-weight-residency/source';assert not new.exists();shutil.copytree(old,new)
p=new/'src/backends/metal/weight_residency_budget.h';p.write_text('''#pragma once
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
namespace Garnet {
class WeightResidencyBudget {
 std::mutex mutex_;std::unordered_map<uint64_t,uint64_t> used_;uint64_t limit_;
public:
 explicit WeightResidencyBudget(uint64_t limit):limit_(limit){}
 bool Acquire(uint64_t device,uint64_t bytes){std::lock_guard<std::mutex> lock(mutex_);if(!bytes||bytes>limit_)return false;auto& n=used_[device];if(n>limit_-bytes)return false;n+=bytes;return true;}
 void Release(uint64_t device,uint64_t bytes) noexcept {std::lock_guard<std::mutex> lock(mutex_);auto i=used_.find(device);if(i!=used_.end()&&i->second>=bytes){i->second-=bytes;if(!i->second)used_.erase(i);}}
 uint64_t Used(uint64_t device){std::lock_guard<std::mutex> lock(mutex_);auto i=used_.find(device);return i==used_.end()?0:i->second;}
};
}
''')
p=new/'src/backends/metal/metal_context.h';s=p.read_text();needle='    void Abort();';assert needle in s;s=s.replace(needle,'''    void BeginWeightResidency(const std::vector<MetalBF16Weight>&);
    void EndWeightResidency();
    std::string WeightResidencyStatsJSON() const;
'''+needle);p.write_text(s)
p=new/'src/backends/metal/metal_context.mm';s=p.read_text();s='#include "weight_residency_budget.h"\n'+s
needle='    id<MTLCommandQueue> queue;';assert needle in s;s=s.replace(needle,needle+'''
    uint64_t weightResidencyAdmitted=0,weightResidencyFallback=0,weightResidencyLastBytes=0,weightResidencyLastCount=0,weightResidencyActiveBytes=0;
    uint64_t weightResidencyAttemptedBytes=0,weightResidencyAttemptedCount=0,weightResidencyReason=0;
#ifdef GARNET_METAL_WEIGHT_RESIDENCY
    id<MTLResidencySet> weightResidencySet=nil;
    std::vector<MetalBF16Weight> weightResidencyOwners;
#endif
''')
needle='    impl_->outputs.clear();impl_->pendingOutputs.clear();impl_->command=nil;impl_->submitted=nil;';assert needle in s;s=s.replace(needle,needle+'\n    EndWeightResidency();')
needle='    impl_->outputs.clear();impl_->pendingOutputs.clear();impl_->command=nil;\n}';assert needle in s;s=s.replace(needle,'    impl_->outputs.clear();impl_->pendingOutputs.clear();impl_->command=nil;\n    EndWeightResidency();\n}')
s+='''
namespace Garnet {
namespace {constexpr uint64_t kWeightResidencyLimit=uint64_t(4)*1024*1024*1024;WeightResidencyBudget& ResidencyBudget(){static WeightResidencyBudget b(kWeightResidencyLimit);return b;}}
void MetalContext::EndWeightResidency() {
#ifdef GARNET_METAL_WEIGHT_RESIDENCY
 if(@available(macOS 15.0,*)) {
  if(impl_->weightResidencySet){[impl_->queue removeResidencySet:impl_->weightResidencySet];[impl_->weightResidencySet endResidency];[impl_->weightResidencySet removeAllAllocations];[impl_->weightResidencySet commit];}
 }
 impl_->weightResidencyOwners.clear();
 if(impl_->weightResidencyActiveBytes){ResidencyBudget().Release(impl_->device.registryID,impl_->weightResidencyActiveBytes);impl_->weightResidencyActiveBytes=0;}
#endif
}
void MetalContext::BeginWeightResidency(const std::vector<MetalBF16Weight>& weights) {
 if(impl_->command||impl_->submitted.count||impl_->weightResidencyActiveBytes)throw std::logic_error("Weight residency requires a closed drained batch");
 impl_->weightResidencyLastBytes=impl_->weightResidencyLastCount=impl_->weightResidencyAttemptedBytes=impl_->weightResidencyAttemptedCount=impl_->weightResidencyReason=0;
#ifdef GARNET_METAL_WEIGHT_RESIDENCY
 if(@available(macOS 15.0,*)) {
  NSMutableArray<id<MTLAllocation>>* allocations=[NSMutableArray new];std::unordered_set<const void*> seen;uint64_t bytes=0;
  for(const auto& weight:weights){if(!weight.impl_||!weight.impl_->buffer)throw std::invalid_argument("Invalid BF16 residency weight");auto buffer=weight.impl_->buffer;if(buffer.device!=impl_->device)throw std::invalid_argument("Residency weight device mismatch");if(!seen.insert((__bridge const void*)buffer).second)continue;auto n=buffer.allocatedSize;if(n>UINT64_MAX-bytes)throw std::overflow_error("Residency byte overflow");bytes+=n;[allocations addObject:buffer];}
  impl_->weightResidencyAttemptedBytes=bytes;impl_->weightResidencyAttemptedCount=seen.size();
  auto fallback=[&](uint64_t reason){++impl_->weightResidencyFallback;impl_->weightResidencyReason=reason;};
  if(!bytes){fallback(2);return;}if(seen.size()>2048||bytes>kWeightResidencyLimit){fallback(4);return;}
  if(!ResidencyBudget().Acquire(impl_->device.registryID,bytes)){fallback(5);return;}
  impl_->weightResidencyActiveBytes=bytes;
  try {
   impl_->weightResidencyOwners=weights;
   if(!impl_->weightResidencySet){auto desc=[MTLResidencySetDescriptor new];desc.initialCapacity=seen.size();desc.label=@"Garnet bounded BF16 masters";NSError* error=nil;impl_->weightResidencySet=[impl_->device newResidencySetWithDescriptor:desc error:&error];}
   if(!impl_->weightResidencySet){EndWeightResidency();fallback(6);return;}
   for(id<MTLAllocation> a in allocations)[impl_->weightResidencySet addAllocation:a];
   [impl_->weightResidencySet commit];[impl_->queue addResidencySet:impl_->weightResidencySet];[impl_->weightResidencySet requestResidency];
   ++impl_->weightResidencyAdmitted;impl_->weightResidencyLastBytes=bytes;impl_->weightResidencyLastCount=seen.size();impl_->weightResidencyReason=1;
  }catch(...){EndWeightResidency();throw;}
 }else{++impl_->weightResidencyFallback;impl_->weightResidencyReason=3;}
#else
 (void)weights;
#endif
}
std::string MetalContext::WeightResidencyStatsJSON() const {
 return "{\\"admitted\\":"+std::to_string(impl_->weightResidencyAdmitted)+",\\"fallback\\":"+std::to_string(impl_->weightResidencyFallback)+",\\"last_bytes\\":"+std::to_string(impl_->weightResidencyLastBytes)+",\\"last_count\\":"+std::to_string(impl_->weightResidencyLastCount)+",\\"active_bytes\\":"+std::to_string(impl_->weightResidencyActiveBytes)+",\\"attempted_bytes\\":"+std::to_string(impl_->weightResidencyAttemptedBytes)+",\\"attempted_count\\":"+std::to_string(impl_->weightResidencyAttemptedCount)+",\\"reason\\":"+std::to_string(impl_->weightResidencyReason)+"}";
}
}
''';p.write_text(s)
p=new/'src/backends/metal/decoder_session.cpp';s=p.read_text();needle='poisoned_=true;gpu_->Begin();\n    try {';assert needle in s;s=s.replace(needle,'''poisoned_=true;
    try {
#ifdef GARNET_METAL_WEIGHT_RESIDENCY
        std::vector<MetalBF16Weight> residencyWeights;
        for(auto graph:locked)for(const auto& slot:graph->slots_)if(slot.bf16.Valid())residencyWeights.push_back(slot.bf16);
        gpu_->BeginWeightResidency(residencyWeights);
#endif
        gpu_->Begin();''');s+='\nnamespace Garnet {std::string MetalDecoderSession::WeightResidencyStatsJSON() const {std::lock_guard<std::mutex> lock(mutex_);return gpu_?gpu_->WeightResidencyStatsJSON():"{}";}}\n';p.write_text(s)
p=new/'src/backends/metal/decoder_session.h';s=p.read_text();s=s.replace('    size_t DenseDecode1024EncodedTotal() const;','    size_t DenseDecode1024EncodedTotal() const;\n    std::string WeightResidencyStatsJSON() const;');needle='        APISET().AddProp("dense_decode1024_encoded_total",&MetalDecoderSession::DenseDecode1024EncodedTotal);';assert needle in s;s=s.replace(needle,needle+'\n        APISET().AddProp("weight_residency_stats_json",&MetalDecoderSession::WeightResidencyStatsJSON);');p.write_text(s)
p=new/'runtime/metal_serving.py';s=p.read_text();needle="        stages['dense_decode1024_encoded_total'] = self.decoder.session.dense_decode1024_encoded_total";assert needle in s;s=s.replace(needle,needle+"\n        stages['weight_residency'] = json.loads(self.decoder.session.weight_residency_stats_json)");p.write_text(s)
p=new/'src/entry/metal_main.cpp';s=p.read_text().replace('#ifdef GARNET_METAL_DENSE_PREFIX_CACHE','#ifdef GARNET_METAL_WEIGHT_RESIDENCY\n        info+="; weight_residency_4gib";\n#endif\n#ifdef GARNET_METAL_DENSE_PREFIX_CACHE',1);p.write_text(s)
p=new/'cmake/GarnetMetal.cmake';p.write_text(p.read_text()+'''\noption(GARNET_METAL_WEIGHT_RESIDENCY "Bounded per-batch BF16 master residency preparation" OFF)
if(GARNET_METAL_WEIGHT_RESIDENCY)
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_WEIGHT_RESIDENCY=1)
endif()
''')
s=(w.parent/'worldsense-private-resident-defaults/build.py').read_text().replace('worldsense-private-resident-defaults','worldsense-weight-residency');needle="'-DGARNET_METAL_PRIVATE_RESIDENT_DEFAULTS='+mode.upper(),";assert needle in s;s=s.replace(needle,needle+"'-DGARNET_METAL_WEIGHT_RESIDENCY='+mode.upper(),");(w/'build.py').write_text(s)
print('IsolatedWS32implementedunbuilt/untested')
