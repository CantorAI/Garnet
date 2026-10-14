// Diagnostic-only helper, installed into a separate source tree.
template<class State> static void TraceBegin(State& state) {
    const char* enabled=std::getenv("GARNET_METAL_COUNTER_TRACE");
    state.traceActive=enabled&&std::strcmp(enabled,"1")==0;
    state.traceNames.clear();
    if(!state.traceActive)return;
    if(state.partialSubmissions)throw std::logic_error("Counter trace requires one command buffer");
    if(!state.traceCounters) {
        if(![state.device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])throw std::runtime_error("Compute stage counters unavailable");
        id<MTLCounterSet> timestamps=nil;
        for(id<MTLCounterSet> set in state.device.counterSets)if([set.name isEqualToString:MTLCommonCounterSetTimestamp])timestamps=set;
        if(!timestamps)throw std::runtime_error("Timestamp counter set unavailable");
        auto desc=[MTLCounterSampleBufferDescriptor new];desc.counterSet=timestamps;desc.sampleCount=2048;desc.storageMode=MTLStorageModeShared;
        NSError* error=nil;state.traceCounters=[state.device newCounterSampleBufferWithDescriptor:desc error:&error];
        if(!state.traceCounters)throw std::runtime_error(error.localizedDescription.UTF8String);
        static std::atomic<uint64_t> next{1};state.traceContext=next.fetch_add(1);
    }
    ++state.traceSequence;
}
template<class State> static id<MTLComputeCommandEncoder> TraceEncoder(State& state,const std::string& name) {
    if(!state.traceActive)return [state.command computeCommandEncoder];
    const size_t index=state.traceNames.size()*2;
    if(index+1>=2048)throw std::runtime_error("Counter trace sample capacity exceeded");
    auto pass=[MTLComputePassDescriptor computePassDescriptor];auto sample=pass.sampleBufferAttachments[0];
    sample.sampleBuffer=state.traceCounters;sample.startOfEncoderSampleIndex=index;sample.endOfEncoderSampleIndex=index+1;
    auto encoder=[state.command computeCommandEncoderWithDescriptor:pass];
    if(!encoder)throw std::runtime_error("Counter trace encoder unavailable");
    state.traceNames.push_back(name);return encoder;
}
template<class State> static void TraceEmit(State& state) {
    if(!state.traceActive)return;
    if(state.partialSubmissions)throw std::logic_error("Counter trace cannot combine partial command buffers");
    const size_t count=state.traceNames.size();
    if(count) {
        auto raw=[state.traceCounters resolveCounterRange:NSMakeRange(0,count*2)];
        if(raw.length!=count*2*sizeof(MTLCounterResultTimestamp))throw std::runtime_error("Counter trace timestamp length invalid");
        auto samples=static_cast<const MTLCounterResultTimestamp*>(raw.bytes);
        for(size_t i=0;i<count;++i) {
            const auto start=samples[i*2].timestamp,end=samples[i*2+1].timestamp;
            if(!start||end<start)throw std::runtime_error("Counter trace timestamps invalid");
            std::fprintf(stderr,"metal_counter_profile {\"context\":%llu,\"command\":%llu,\"index\":%zu,\"op\":\"%s\",\"start_raw\":%llu,\"end_raw\":%llu,\"diagnostic\":true,\"serialized\":false}\n",
                (unsigned long long)state.traceContext,(unsigned long long)state.traceSequence,i,state.traceNames[i].c_str(),(unsigned long long)start,(unsigned long long)end);
        }
    }
    std::fprintf(stderr,"metal_counter_command {\"context\":%llu,\"command\":%llu,\"samples\":%zu,\"gpu_span_ms\":%.9g,\"partial_submissions\":%zu,\"diagnostic\":true,\"serialized\":false}\n",
        (unsigned long long)state.traceContext,(unsigned long long)state.traceSequence,count,state.gpuMs,state.partialSubmissions);
}
