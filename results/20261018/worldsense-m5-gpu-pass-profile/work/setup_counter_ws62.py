"""Isolate WS54 and add only bounded GPU-counter metadata instrumentation."""
from pathlib import Path
import json,hashlib,shutil,re,ast
base=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=base/'work/worldsense-gpu-counter-profile';r=Path('/Users/shawn/CantorAI');old=r/'out/research/worldsense-isolated-primary/source';s=r/'out/research/worldsense-gpu-counter-profile/source'
assert json.loads((w/'probe-verification.json').read_text())['native_M5_stage_boundary_timestamp_support']
seed=json.loads((w.parent/'worldsense-isolated-primary/correctness-source-manifest.json').read_text());assert len(seed)==1718
assert all(hashlib.sha256((old/k).read_bytes()).hexdigest()==v for k,v in seed.items())
s.mkdir(parents=True,exist_ok=False)
for rel in seed:
 p=s/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/rel,p)
changes={}
def replace(name,before,after,count=1):
 p=s/name;text=p.read_text();assert text.count(before)==count,(name,before,text.count(before));p.write_text(text.replace(before,after));changes.setdefault(name,[]).append(dict(before=before,after=after,count=count))
header='src/backends/metal/metal_context.h';mm='src/backends/metal/metal_context.mm';entry='src/entry/metal_main.cpp';serving='runtime/metal_serving.py'
scope='''class MetalCounterTraceRequest {
    std::string priorModel_;uint64_t priorRequest_=0,priorStep_=0;
public:
    explicit MetalCounterTraceRequest(const std::string&);
    ~MetalCounterTraceRequest() noexcept;
    MetalCounterTraceRequest(const MetalCounterTraceRequest&)=delete;
    MetalCounterTraceRequest& operator=(const MetalCounterTraceRequest&)=delete;
};
'''
replace(header,'class MetalContext {',scope+'class MetalContext {')
api='''    static void CounterTraceDecodeStep(int64_t);
    static uint64_t CounterTraceRequestId();
    static bool CounterTraceTimestampPairValid(uint64_t,uint64_t);
#if defined(GARNET_METAL_GPU_PASS_COUNTER_PROFILE) && defined(GARNET_METAL_PRIMARY_TEST_FAULTS)
    static void CounterTraceTestLimit(size_t);
    static void CounterTraceTestEmitFailure(bool);
    static void CounterTraceTestUnavailable(bool);
#endif
'''
replace(header,'    static MetalMemoryStats MemoryStats();','    static MetalMemoryStats MemoryStats();\n'+api)
replace(mm,'#include <cstring>','#include <cstring>\n#include <cstdio>\n#include <cstdlib>\n#include <unistd.h>\n#include "nlohmann/json.hpp"')
fields='''
#ifdef GARNET_METAL_GPU_PASS_COUNTER_PROFILE
    id<MTLCounterSampleBuffer> traceCounters=nil;
    std::FILE* traceStream=nullptr;
    bool traceActive=false;
    uint64_t traceContext=0,traceSequence=0,traceRequest=0,traceStep=0;
    std::string traceModelJSON;
    std::vector<std::string> traceNames;
#endif
'''
replace(mm,'    double gpuMs=0;','    double gpuMs=0;'+fields)
replace(mm,'static size_t Elements(','#include "gpu_pass_counter_trace.inc"\nstatic size_t Elements(')
replace(mm,'[impl_->command computeCommandEncoder]','PassTraceEncoder(*impl_,__func__)',14)
for begin,end,tag in [
 ('void MetalContext::EncodeBF16MatVec(','void MetalContext::Update(','std::string("bf16:")+std::to_string(a.rows)+":"+std::to_string(a.columns)+":"+std::to_string(o.columns)+":"+(primary?"primary":"raw")'),
 ('void MetalContext::EncodePointwise(','void MetalContext::EncodeRMSNorm(','std::string("pointwise:")+std::to_string(operation)+":"+std::to_string(a.rows)+":"+std::to_string(a.columns)'),
 ('void MetalContext::EncodeStateKernel(','void MetalContext::EncodeLayout(','std::string("state:")+std::to_string(operation)+":"+std::to_string(c.rows)+":"+std::to_string(c.columns)')]:
 text=(s/mm).read_text();start=text.index(begin);stop=text.index(end,start);part=text[start:stop];assert 'PassTraceEncoder(*impl_,__func__)'in part
 new=part.replace('PassTraceEncoder(*impl_,__func__)','PassTraceEncoder(*impl_,'+tag+')');replace(mm,part,new)
replace(mm,'    impl_->partialSubmissions=0;','    impl_->partialSubmissions=0;\n    PassTraceBegin(*impl_);')
replace(mm,'    for(auto& output:impl_->outputs){output->pending=false;output->ready=success;}','    if(success)PassTraceEmit(*impl_);\n    for(auto& output:impl_->outputs){output->pending=false;output->ready=success;}')
replace(mm,'void MetalContext::Abort() {','void MetalContext::Abort() {\n    PassTraceAbort(*impl_);')
replace(mm,'MetalContext::~MetalContext() { Abort(); }','MetalContext::~MetalContext() { Abort();\n#ifdef GARNET_METAL_GPU_PASS_COUNTER_PROFILE\n if(impl_->traceStream)std::fclose(impl_->traceStream);\n#endif\n}')
replace(entry,'    std::string InferJson(X::Value prompt, X::Value image, X::Value limit, X::Value model) {','    std::string InferJson(X::Value prompt, X::Value image, X::Value limit, X::Value model) {\n#ifdef GARNET_METAL_GPU_PASS_COUNTER_PROFILE\n        MetalCounterTraceRequest counterRequest(model.ToString());\n#endif')
methods='''#ifdef GARNET_METAL_GPU_PASS_COUNTER_PROFILE
    bool CounterTraceStep(int64_t step){MetalContext::CounterTraceDecodeStep(step);return true;}
    int64_t CounterTraceId(){return int64_t(MetalContext::CounterTraceRequestId());}
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
    bool CounterTraceEmitFault(int64_t enabled){if(enabled!=0&&enabled!=1)throw X::Error("Counter fault flag must be zero or one");MetalContext::CounterTraceTestEmitFailure(enabled!=0);return true;}
#endif
#endif
'''
replace(entry,'    std::string BackendInfo() const {',methods+'    std::string BackendInfo() const {')
replace(entry,'        return info;','''#ifdef GARNET_METAL_GPU_PASS_COUNTER_PROFILE
        info+="; gpu_pass_counter_profile";
#endif
        return info;''')
exports='''#ifdef GARNET_METAL_GPU_PASS_COUNTER_PROFILE
        APISET().AddFunc<1>("counter_trace_step",&GarnetMetalAPI::CounterTraceStep);
        APISET().AddFunc<0>("counter_trace_request_id",&GarnetMetalAPI::CounterTraceId);
#ifdef GARNET_METAL_PRIMARY_TEST_FAULTS
        APISET().AddFunc<1>("_test_counter_emit_failure",&GarnetMetalAPI::CounterTraceEmitFault);
#endif
#endif
'''
replace(entry,'        APISET().AddFunc<0>("backend_info",&GarnetMetalAPI::BackendInfo);','        APISET().AddFunc<0>("backend_info",&GarnetMetalAPI::BackendInfo);\n'+exports)
replace(serving,'import garnet\n','import garnet\n_gpu_pass_trace = "gpu_pass_counter_profile" in garnet.backend_info()\n')
replace(serving,'        start = time.monotonic()','        start = time.monotonic()\n        counter_request = garnet.counter_trace_request_id() if _gpu_pass_trace else 0')
call='            prediction = self.decoder.step(prediction, active=[1], positions=[[current, current, current]])'
replace(serving,call,'            if _gpu_pass_trace and counter_request:\n                garnet.counter_trace_step(index + 1)\n'+call)
replace(serving,'        return dict(input_prompt_sha256=','        if _gpu_pass_trace:\n            stages["counter_trace_request_id"] = counter_request\n        return dict(input_prompt_sha256=')
cmake='cmake/GarnetMetal.cmake';text=(s/cmake).read_text();extra='''
# Diagnostic only. Original shader/math/command boundaries remain unchanged.
option(GARNET_METAL_GPU_PASS_COUNTER_PROFILE "Collect bounded GPU-pass timestamps at explicit decode steps" OFF)
target_include_directories(garnet_metal PRIVATE third_party)
if(GARNET_METAL_GPU_PASS_COUNTER_PROFILE)
 target_compile_definitions(garnet_metal PUBLIC GARNET_METAL_GPU_PASS_COUNTER_PROFILE=1)
endif()
add_executable(garnet_gpu_pass_counter_test test2026/metal/gpu_pass_counter_test.cpp)
target_include_directories(garnet_gpu_pass_counter_test PRIVATE third_party)
target_link_libraries(garnet_gpu_pass_counter_test PRIVATE garnet_metal)
add_test(NAME metal_gpu_pass_counter_reference COMMAND garnet_gpu_pass_counter_test "${CMAKE_CURRENT_BINARY_DIR}/gpu-pass-counter-test.jsonl")
''';(s/cmake).write_text(text+extra);changes.setdefault(cmake,[]).append(dict(before='',after=extra,count=1,append=True))
for name,rel in [('gpu_pass_counter_trace.inc','src/backends/metal/gpu_pass_counter_trace.inc'),('gpu_pass_counter_test.cpp','test2026/metal/gpu_pass_counter_test.cpp')]:shutil.copy2(w/name,s/rel)
for name,transforms in changes.items():
 text=(s/name).read_text()
 for change in reversed(transforms):
  if change.get('append'):assert text.endswith(change['after']);text=text[:-len(change['after'])]
  else:
   assert text.count(change['after'])==change['count'],(name,'inverse count');text=text.replace(change['after'],change['before'])
 assert text==(old/name).read_text(),name
 if name.endswith('.py'):ast.parse((s/name).read_text())
pattern=r'R"([A-Za-z_]+)\((.*?)\)\1"';assert re.findall(pattern,(s/mm).read_text(),re.S)==re.findall(pattern,(old/mm).read_text(),re.S)
manifest={str(p.relative_to(s)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(s.rglob('*'))if p.is_file()}
assert len(manifest)==1720 and all(manifest[k]==v for k,v in seed.items()if k not in changes)
(w/'seed-source-manifest.json').write_text(json.dumps(seed,indent=2)+'\n');(w/'correctness-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(w/'source-transformations.json').write_text(json.dumps(changes,indent=2)+'\n')
(w/'source-inverse-proof.json').write_text(json.dumps(dict(seed_files=1718,current_files=1720,changed_or_added=sorted(k for k,v in manifest.items()if seed.get(k)!=v),original_shader_literals_graph_session_xModel_image_CUDA_and_tests_unchanged=True,metadata_transformations_inverse_exact=True,no_model_inference_or_performance_result=True),indent=2)+'\n')
shutil.copytree(w.parent/'worldsense-isolated-primary/build-profile',w/'build-profile')
print('WS62 source1720/seven-file delta; all metadata inverse exact, original shader/math/graph/session/xModel/image/CUDA unchanged')
