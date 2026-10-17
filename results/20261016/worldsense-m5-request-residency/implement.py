from pathlib import Path
import shutil
w=Path(__file__).resolve().parent;old=Path('/Users/shawn/CantorAI/out/research/worldsense-weight-residency/source');new=old.parent.parent/'worldsense-request-residency/source';assert not new.exists();shutil.copytree(old,new)
p=new/'src/backends/metal/metal_context.h';s=p.read_text();needle='    void BeginWeightResidency(const std::vector<MetalBF16Weight>&);';assert needle in s;s=s.replace(needle,'''    bool BeginWeightResidencyRequest();
    void EndWeightResidencyRequest();
    void CheckWeightResidencyRequestOwner() const;
'''+needle);p.write_text(s)
p=new/'src/backends/metal/metal_context.mm';s=p.read_text();s='#include <thread>\n'+s;needle='    uint64_t weightResidencyAttemptedBytes=0';at=s.index(needle);s=s[:at]+'''    bool weightResidencyRequest=false;
    std::thread::id weightResidencyRequestOwner;
    uint64_t weightResidencyRequestStarts=0,weightResidencyRequestEnds=0,weightResidencyReuses=0;
    std::vector<std::pair<uintptr_t,uint64_t>> weightResidencyFingerprint;
'''+s[at:]
needle='    EndWeightResidency();';assert s.count(needle)==2;s=s.replace(needle,'    if(!impl_->weightResidencyRequest)EndWeightResidency();',1)
# Abort remains unconditional. Clearing a cache does not end the logical request.
needle='void MetalContext::EndWeightResidency() {\n';s=s.replace(needle,needle+' impl_->weightResidencyFingerprint.clear();\n')
needle='if(impl_->command||impl_->submitted.count||impl_->weightResidencyActiveBytes)throw';assert needle in s;s=s.replace(needle,'CheckWeightResidencyRequestOwner();\n if(impl_->command||impl_->submitted.count||(!impl_->weightResidencyRequest&&impl_->weightResidencyActiveBytes))throw')
needle='NSMutableArray<id<MTLAllocation>>* allocations=[NSMutableArray new];std::unordered_set<const void*> seen;uint64_t bytes=0;';assert needle in s;s=s.replace(needle,needle+'std::vector<std::pair<uintptr_t,uint64_t>> fingerprint;')
needle='bytes+=n;[allocations addObject:buffer];';assert needle in s;s=s.replace(needle,'bytes+=n;fingerprint.emplace_back(reinterpret_cast<uintptr_t>((__bridge const void*)buffer),n);[allocations addObject:buffer];')
needle='  auto fallback=[&](uint64_t reason)';at=s.index(needle);s=s[:at]+'''  std::sort(fingerprint.begin(),fingerprint.end());
  if(impl_->weightResidencyRequest&&impl_->weightResidencyActiveBytes&&fingerprint==impl_->weightResidencyFingerprint){++impl_->weightResidencyReuses;impl_->weightResidencyLastBytes=bytes;impl_->weightResidencyLastCount=seen.size();impl_->weightResidencyReason=1;return;}
  if(impl_->weightResidencyActiveBytes)EndWeightResidency();
'''+s[at:]
needle='++impl_->weightResidencyAdmitted;impl_->weightResidencyLastBytes=bytes;';assert needle in s;s=s.replace(needle,'impl_->weightResidencyFingerprint=std::move(fingerprint);++impl_->weightResidencyAdmitted;impl_->weightResidencyLastBytes=bytes;')
start=s.index('std::string MetalContext::WeightResidencyStatsJSON() const {');a=s[:start];b=s[start:];b=b.replace(' return ',' auto result=',1);ending='\n}\n}\n';assert b.endswith(ending);b=b[:-len(ending)]+'''
 result.pop_back();result+=",\\"request_active\\":"+std::to_string(impl_->weightResidencyRequest)+",\\"request_starts\\":"+std::to_string(impl_->weightResidencyRequestStarts)+",\\"request_ends\\":"+std::to_string(impl_->weightResidencyRequestEnds)+",\\"request_reuses\\":"+std::to_string(impl_->weightResidencyReuses)+"}";
 return result;
'''+ending;s=a+b
s+='''
namespace Garnet {
void MetalContext::CheckWeightResidencyRequestOwner() const {if(impl_->weightResidencyRequest&&impl_->weightResidencyRequestOwner!=std::this_thread::get_id())throw std::logic_error("Weight residency request belongs to another thread");}
bool MetalContext::BeginWeightResidencyRequest() {
#ifdef GARNET_METAL_REQUEST_WEIGHT_RESIDENCY
 if(impl_->weightResidencyRequest)throw std::logic_error("Nested weight residency request");
 if(impl_->command||impl_->submitted.count)throw std::logic_error("Drain before request begin");
 EndWeightResidency();impl_->weightResidencyRequest=true;impl_->weightResidencyRequestOwner=std::this_thread::get_id();++impl_->weightResidencyRequestStarts;return true;
#else
 return false;
#endif
}
void MetalContext::EndWeightResidencyRequest() {
#ifdef GARNET_METAL_REQUEST_WEIGHT_RESIDENCY
 CheckWeightResidencyRequestOwner();if(!impl_->weightResidencyRequest)throw std::logic_error("No weight residency request");
 EndWeightResidency();impl_->weightResidencyRequest=false;impl_->weightResidencyRequestOwner=std::thread::id();++impl_->weightResidencyRequestEnds;
#endif
}
}
''';p.write_text(s)
p=new/'src/backends/metal/decoder_session.h';s=p.read_text();needle='    std::string WeightResidencyStatsJSON() const;';assert needle in s;s=s.replace(needle,needle+'\n    bool BeginWeightResidencyRequest();\n    void EndWeightResidencyRequest();');needle='        APISET().AddProp("weight_residency_stats_json",&MetalDecoderSession::WeightResidencyStatsJSON);';assert needle in s;s=s.replace(needle,needle+'\n        APISET().AddFunc<0>("begin_weight_residency_request",&MetalDecoderSession::BeginWeightResidencyRequest);\n        APISET().AddFunc<0>("end_weight_residency_request",&MetalDecoderSession::EndWeightResidencyRequest);');p.write_text(s)
p=new/'src/backends/metal/decoder_session.cpp';s=p.read_text();needle='void MetalDecoderSession::ResetUnlocked() {\n';assert needle in s;s=s.replace(needle,needle+'    if(gpu_){gpu_->CheckWeightResidencyRequestOwner();gpu_->EndWeightResidency();}\n');needle='    if(!gpu_||poisoned_)throw std::logic_error("Reset a closed/failed Metal session before running");';assert needle in s;s=s.replace(needle,'    if(gpu_)gpu_->CheckWeightResidencyRequestOwner();\n'+needle);s+='''
namespace Garnet {
bool MetalDecoderSession::BeginWeightResidencyRequest(){std::lock_guard<std::mutex> lock(mutex_);if(!gpu_||poisoned_)throw std::logic_error("Reset closed/failed session before residency request");return gpu_->BeginWeightResidencyRequest();}
void MetalDecoderSession::EndWeightResidencyRequest(){std::lock_guard<std::mutex> lock(mutex_);if(gpu_)gpu_->EndWeightResidencyRequest();}
}
''';p.write_text(s)
p=new/'runtime/metal_serving.py';s=p.read_text();needle="            return json.dumps(_models[name].infer(prompt, image, int(limit)))";assert needle in s;s=s.replace(needle,'''            model = _models[name]
            if 'request_weight_residency' not in garnet.backend_info():
                return json.dumps(model.infer(prompt, image, int(limit)))
            outer_start = time.monotonic()
            begun = model.decoder.session.begin_weight_residency_request()
            before_infer = time.monotonic()
            try:
                result = model.infer(prompt, image, int(limit))
            finally:
                cleanup_start = time.monotonic()
                if begun:
                    model.decoder.session.end_weight_residency_request()
                cleanup_seconds = time.monotonic() - cleanup_start
            result['ttft_seconds'] += before_infer - outer_start
            result['timings']['weight_residency'] = json.loads(model.decoder.session.weight_residency_stats_json)
            result['timings']['request_weight_residency_cleanup_seconds'] = cleanup_seconds
            result['timings']['memory_after_request_cleanup'] = json.loads(garnet.memory_stats_json())
            result['total_seconds'] = time.monotonic() - outer_start
            return json.dumps(result)''');p.write_text(s)
p=new/'src/entry/metal_main.cpp';s=p.read_text().replace('#ifdef GARNET_METAL_DENSE_PREFIX_CACHE','#ifdef GARNET_METAL_REQUEST_WEIGHT_RESIDENCY\n        info+="; request_weight_residency";\n#endif\n#ifdef GARNET_METAL_DENSE_PREFIX_CACHE',1);p.write_text(s)
p=new/'cmake/GarnetMetal.cmake';p.write_text(p.read_text()+'''\noption(GARNET_METAL_REQUEST_WEIGHT_RESIDENCY "Explicit bounded request lifetime for weight residency" OFF)
if(GARNET_METAL_REQUEST_WEIGHT_RESIDENCY)
 if(NOT GARNET_METAL_WEIGHT_RESIDENCY)
  message(FATAL_ERROR "Request weight residency requires weight residency")
 endif()
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_REQUEST_WEIGHT_RESIDENCY=1)
endif()
''')
s=(w.parent/'worldsense-weight-residency/build.py').read_text().replace('worldsense-weight-residency','worldsense-request-residency');needle="'-DGARNET_METAL_WEIGHT_RESIDENCY='+mode.upper(),";assert needle in s;s=s.replace(needle,needle+"'-DGARNET_METAL_REQUEST_WEIGHT_RESIDENCY='+mode.upper(),");(w/'build.py').write_text(s)
print('WS33isolatedimplementationwritten;unbuilt/untested')
