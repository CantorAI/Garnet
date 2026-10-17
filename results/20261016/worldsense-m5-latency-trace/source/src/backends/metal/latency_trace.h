#pragma once
#include "metal_context.h"
#include <chrono>
#include <exception>
#include <sys/resource.h>
namespace Garnet {
struct NativeLatencyTrace {
    bool valid=false,scoped=false,failed=false,bodyEntered=false;
    uint64_t sequence=0;
    double nativeSeconds=0,bodySeconds=0,preBodySeconds=0,postBodySeconds=0,samplingSeconds=0,userSeconds=0,systemSeconds=0;
    int beforeStatus=0,afterStatus=0;
    long minorFaults=0,majorFaults=0;
};
inline thread_local NativeLatencyTrace lastNativeLatencyTrace;
inline thread_local uint64_t nativeLatencySequence=0;
template<class F> void TraceNativeSessionRun(bool scoped,F&& fn) {
    using Clock=std::chrono::steady_clock;
    auto seconds=[](auto a,auto b){return std::chrono::duration<double>(b-a).count();};
    NativeLatencyTrace trace;trace.valid=true;trace.scoped=scoped;
    rusage before{},after{};auto sampleStart=Clock::now();trace.beforeStatus=getrusage(RUSAGE_SELF,&before);auto start=Clock::now();
    trace.samplingSeconds=seconds(sampleStart,start);auto bodyStart=start,bodyEnd=start;
    std::exception_ptr error;
    auto body=[&]{trace.bodyEntered=true;bodyStart=Clock::now();try{fn();}catch(...){bodyEnd=Clock::now();throw;}bodyEnd=Clock::now();};
    try{if(scoped)MetalContext::StorageScope(body);else body();}catch(...){error=std::current_exception();trace.failed=true;}
    auto stop=Clock::now();trace.nativeSeconds=seconds(start,stop);
    if(trace.bodyEntered){trace.bodySeconds=seconds(bodyStart,bodyEnd);trace.preBodySeconds=seconds(start,bodyStart);trace.postBodySeconds=seconds(bodyEnd,stop);}
    sampleStart=Clock::now();trace.afterStatus=getrusage(RUSAGE_SELF,&after);trace.samplingSeconds+=seconds(sampleStart,Clock::now());
    if(trace.beforeStatus==0&&trace.afterStatus==0){
        auto cpu=[](const timeval& t){return double(t.tv_sec)+double(t.tv_usec)*1e-6;};
        trace.userSeconds=cpu(after.ru_utime)-cpu(before.ru_utime);trace.systemSeconds=cpu(after.ru_stime)-cpu(before.ru_stime);
        trace.minorFaults=after.ru_minflt-before.ru_minflt;trace.majorFaults=after.ru_majflt-before.ru_majflt;
    }
    trace.sequence=++nativeLatencySequence;lastNativeLatencyTrace=trace;
    if(error)std::rethrow_exception(error);
}
}
