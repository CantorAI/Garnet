// Diagnostic-only metadata and pass counter attachments; no shader arithmetic.
#ifdef GARNET_METAL_GPU_PASS_COUNTER_PROFILE
namespace {
struct PassTraceScope {uint64_t request=0,step=0;std::string modelJSON;};
thread_local PassTraceScope passTraceScope;
std::atomic<uint64_t> passTraceNextRequest{1},passTraceNextContext{1};
uint64_t PassTraceId(std::atomic<uint64_t>& next) {
 auto value=next.load();
 for(;;){if(!value||value>=uint64_t(INT64_MAX))throw std::overflow_error("GPU trace identity exhausted");
  if(next.compare_exchange_weak(value,value+1))return value;}
}
#ifdef GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS
std::atomic<size_t> passTraceTestLimit{4096};
std::atomic<bool> passTraceTestEmitFailure{false};
std::atomic<bool> passTraceTestUnavailable{false};
#endif
}
MetalCounterTraceRequest::MetalCounterTraceRequest(const std::string& model) {
 auto encoded=nlohmann::json(model).dump();
 const auto id=PassTraceId(passTraceNextRequest);
 priorRequest_=passTraceScope.request;priorStep_=passTraceScope.step;
 priorModel_.swap(passTraceScope.modelJSON);
 passTraceScope.modelJSON.swap(encoded);passTraceScope.request=id;passTraceScope.step=0;
}
MetalCounterTraceRequest::~MetalCounterTraceRequest() noexcept {
 passTraceScope.modelJSON.swap(priorModel_);
 passTraceScope.request=priorRequest_;passTraceScope.step=priorStep_;
}
void MetalContext::CounterTraceDecodeStep(int64_t step) {
 if(!passTraceScope.request||step<=0)throw std::invalid_argument("GPU trace step requires an active request and positive index");
 passTraceScope.step=uint64_t(step);
}
uint64_t MetalContext::CounterTraceRequestId(){return passTraceScope.request;}
#ifdef GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS
void MetalContext::CounterTraceTestLimit(size_t value){if(value<2||value>4096||value%2)throw std::invalid_argument("GPU trace test capacity invalid");passTraceTestLimit=value;}
void MetalContext::CounterTraceTestEmitFailure(bool value){passTraceTestEmitFailure=value;}
void MetalContext::CounterTraceTestUnavailable(bool value){passTraceTestUnavailable=value;}
#endif
template<class State>static void PassTraceBegin(State& state) {
 state.traceNames.clear();state.traceActive=false;
 const char* enabled=std::getenv("GARNET_METAL_COUNTER_TRACE");
 if(!enabled||std::strcmp(enabled,"1")||!passTraceScope.request||
    (passTraceScope.step!=4&&passTraceScope.step!=16&&passTraceScope.step!=28))return;
 if(state.partialSubmissions||state.submitted.count)throw std::logic_error("GPU trace requires original single command buffer");
#ifdef GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS
 if(passTraceTestUnavailable.load())throw std::runtime_error("Injected unavailable GPU stage counters");
#endif
 state.traceRequest=passTraceScope.request;state.traceStep=passTraceScope.step;state.traceModelJSON=passTraceScope.modelJSON;
 if(!state.traceStream){
  const char* path=std::getenv("GARNET_METAL_COUNTER_TRACE_FILE");
  if(!path||!*path)throw std::runtime_error("GPU trace requires a dedicated output file");
  state.traceStream=std::fopen(path,"ab");if(!state.traceStream)throw std::runtime_error("Cannot open GPU trace file");
 }
 if(!state.traceCounters){
  if(![state.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])throw std::runtime_error("GPU stage counters unavailable");
  id<MTLCounterSet> timestamps=nil;
  for(id<MTLCounterSet> set in state.device.counterSets)if([set.name isEqualToString:MTLCommonCounterSetTimestamp])timestamps=set;
  if(!timestamps)throw std::runtime_error("GPU timestamp counter set unavailable");
  auto desc=[MTLCounterSampleBufferDescriptor new];desc.counterSet=timestamps;desc.sampleCount=4096;desc.storageMode=MTLStorageModeShared;
  NSError* error=nil;state.traceCounters=[state.device newCounterSampleBufferWithDescriptor:desc error:&error];
  if(!state.traceCounters)throw std::runtime_error(error?error.localizedDescription.UTF8String:"GPU timestamp allocation failed");
  state.traceContext=PassTraceId(passTraceNextContext);
 }
 ++state.traceSequence;state.traceActive=true;
}
template<class State>static id<MTLComputeCommandEncoder> PassTraceEncoder(State& state,const std::string& name) {
 if(!state.traceActive)return[state.command computeCommandEncoder];
 const size_t index=state.traceNames.size()*2;
 size_t limit=4096;
#ifdef GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS
 limit=passTraceTestLimit.load();
#endif
 if(index+1>=limit)throw std::runtime_error("GPU trace sample capacity exceeded");
 auto pass=[MTLComputePassDescriptor computePassDescriptor];pass.dispatchType=MTLDispatchTypeSerial;
 auto sample=pass.sampleBufferAttachments[0];sample.sampleBuffer=state.traceCounters;
 sample.startOfEncoderSampleIndex=index;sample.endOfEncoderSampleIndex=index+1;
 auto encoder=[state.command computeCommandEncoderWithDescriptor:pass];
 if(!encoder)throw std::runtime_error("GPU timestamp encoder unavailable");
 state.traceNames.push_back(name);return encoder;
}
template<class State>static void PassTraceEmit(State& state) {
 if(!state.traceActive)return;
 if(state.partialSubmissions)throw std::logic_error("GPU trace cannot combine partial commands");
#ifdef GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS
 if(passTraceTestEmitFailure.load())throw std::runtime_error("Injected GPU trace resolution failure");
#endif
 const size_t count=state.traceNames.size();
 auto raw=[state.traceCounters resolveCounterRange:NSMakeRange(0,count*2)];
 if(!count||raw.length!=count*2*sizeof(MTLCounterResultTimestamp))throw std::runtime_error("GPU trace resolved timestamp length invalid");
 auto samples=static_cast<const MTLCounterResultTimestamp*>(raw.bytes);
 for(size_t i=0;i<count;++i)if(!MetalContext::CounterTraceTimestampPairValid(samples[2*i].timestamp,samples[2*i+1].timestamp))throw std::runtime_error("GPU trace invalid timestamp result");
 static std::mutex fileMutex;std::lock_guard<std::mutex> lock(fileMutex);
 for(size_t i=0;i<count;++i){
  const auto op=nlohmann::json(state.traceNames[i]).dump();
  if(std::fprintf(state.traceStream,"metal_counter_profile {\"pid\":%d,\"context\":%llu,\"command\":%llu,\"request\":%llu,\"model\":%s,\"decode_step\":%llu,\"index\":%zu,\"op\":%s,\"start_raw\":%llu,\"end_raw\":%llu,\"diagnostic\":true,\"serialized\":false}\n",getpid(),(unsigned long long)state.traceContext,(unsigned long long)state.traceSequence,(unsigned long long)state.traceRequest,state.traceModelJSON.c_str(),(unsigned long long)state.traceStep,i,op.c_str(),(unsigned long long)samples[2*i].timestamp,(unsigned long long)samples[2*i+1].timestamp)<0)throw std::runtime_error("GPU trace write failed");
 }
 if(std::fprintf(state.traceStream,"metal_counter_command {\"pid\":%d,\"context\":%llu,\"command\":%llu,\"request\":%llu,\"model\":%s,\"decode_step\":%llu,\"samples\":%zu,\"gpu_span_ms\":%.17g,\"partial_submissions\":%zu,\"diagnostic\":true,\"serialized\":false}\n",getpid(),(unsigned long long)state.traceContext,(unsigned long long)state.traceSequence,(unsigned long long)state.traceRequest,state.traceModelJSON.c_str(),(unsigned long long)state.traceStep,count,state.gpuMs,state.partialSubmissions)<0||std::fflush(state.traceStream))throw std::runtime_error("GPU trace flush failed");
 state.traceActive=false;
}
template<class State>static void PassTraceAbort(State& state)noexcept{state.traceActive=false;state.traceNames.clear();}
#else
MetalCounterTraceRequest::MetalCounterTraceRequest(const std::string&){}
MetalCounterTraceRequest::~MetalCounterTraceRequest()noexcept{}
void MetalContext::CounterTraceDecodeStep(int64_t){}
uint64_t MetalContext::CounterTraceRequestId(){return 0;}
template<class State>static void PassTraceBegin(State&){}
#define PassTraceEncoder(state,name) [(state).command computeCommandEncoder]
template<class State>static void PassTraceEmit(State&){}
template<class State>static void PassTraceAbort(State&)noexcept{}
#endif
bool MetalContext::CounterTraceTimestampPairValid(uint64_t start,uint64_t end){return start&&start!=UINT64_MAX&&end!=UINT64_MAX&&end>=start;}
