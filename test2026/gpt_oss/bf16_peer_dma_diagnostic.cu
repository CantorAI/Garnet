// SPDX-License-Identifier: Apache-2.0
// Private eager synchronization diagnosis only; no performance or serving claim.
#include "bf16_peer_dma_diagnostic_native.cuh"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {
using Peer=GarnetDmaDiagnostic::Bf16PeerNative;
struct Event {unsigned long long ns=0;const char* stage=nullptr;bool entering=false;int code=0;};
struct Trace {
    const std::chrono::steady_clock::time_point origin=std::chrono::steady_clock::now();
    std::array<std::array<Event,64>,2> events{};
    std::array<size_t,2> counts{};std::array<bool,2> overflow{};
    static void record(void* context,int rank,const char* stage,bool entering,int code){
        auto& self=*static_cast<Trace*>(context);
        if(rank<0 || rank>1)return;
        auto& index=self.counts[rank];if(index==64){self.overflow[rank]=true;return;}
        const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-self.origin).count();
        self.events[rank][index++]={static_cast<unsigned long long>(ns),stage,entering,code};
    }
    template<class F> cudaError_t call(int rank,const char* stage,F body){
        record(this,rank,stage,true,0);auto e=body();record(this,rank,stage,false,int(e));return e;
    }
    bool save(const std::filesystem::path& path,const std::array<cudaError_t,2>& errors){
        if(std::filesystem::exists(path))return false;FILE* f=std::fopen(path.string().c_str(),"wb");if(!f)return false;
        std::fprintf(f,"{\"protocol\":\"opt251-private-eager-host-trace-v1\",\"clock\":\"common-steady-relative-nanoseconds\",\"host_only\":true,\"overflow\":[%s,%s],\"rank_operation_errors\":[%d,%d],\"events\":[",overflow[0]?"true":"false",overflow[1]?"true":"false",int(errors[0]),int(errors[1]));
        bool first=true;for(int r=0;r<2;++r)for(size_t i=0;i<counts[r];++i){const auto& e=events[r][i];
            std::fprintf(f,"%s{\"rank\":%d,\"index\":%zu,\"ns\":%llu,\"stage\":\"%s\",\"entering\":%s,\"code\":%d}",first?"":",",r,i,e.ns,e.stage,e.entering?"true":"false",e.code);first=false;}
        std::fprintf(f,"],\"scope\":\"Instrumented single eager host submission diagnosis. No GPU timing, safety qualification or serving speed.\"}\n");
        const bool failed=std::ferror(f)!=0;const int closed=std::fclose(f);return !failed && closed==0;
    }
};
uint32_t bits(float x){uint32_t v;std::memcpy(&v,&x,4);return v;}
float value(uint32_t v){float x;std::memcpy(&x,&v,4);return x;}
bool saveRaw(const std::filesystem::path& path,const std::vector<float>& values){
    if(std::filesystem::exists(path))return false;FILE* f=std::fopen(path.string().c_str(),"wb");if(!f)return false;
    bool failed=std::fwrite(values.data(),4,values.size(),f)!=values.size();const int closed=std::fclose(f);return !failed && closed==0;
}
struct Buffers {
    int original=0;std::array<float*,2> input{},output{};
    ~Buffers(){for(int r=0;r<2;++r){auto e=cudaSetDevice(r);if(e!=cudaSuccess)std::terminate();
        for(auto* p:{input[r],output[r]})if(p && cudaFree(p)!=cudaSuccess)std::terminate();}
        if(cudaSetDevice(original)!=cudaSuccess)std::terminate();}
};
int run(const std::filesystem::path& folder){
    constexpr size_t count=512*2880;constexpr int blocks=128;
    // Buffers die after owner quiescence; the owner and both streams stay live
    // throughout every diagnostic copy/snapshot and full raw output capture.
    Buffers buffers;if(cudaGetDevice(&buffers.original)!=cudaSuccess)return 1;
    Peer peer;auto e=peer.initialize(count,blocks);if(e!=cudaSuccess)return 1;
    std::array<std::vector<float>,2> input{std::vector<float>(count),std::vector<float>(count)};
    std::array<std::vector<float>,2> actual{std::vector<float>(count),std::vector<float>(count)};
    std::vector<float> expected(count);
    for(size_t i=0;i<count;++i){uint16_t a=uint16_t(i);if((a&0x7f80U)==0x7f80U)a^=0x80U;
        input[0][i]=input[1][i]=value(uint32_t(a)<<16);volatile float sum=input[0][i]+input[1][i];
        const auto b=bits(sum);expected[i]=value((b+0x7fffU+((b>>16)&1U))&0xffff0000U);}
    for(int r=0;r<2;++r){if(cudaSetDevice(r)!=cudaSuccess)return 1;
        if(cudaMalloc(&buffers.input[r],count*4)!=cudaSuccess || cudaMalloc(&buffers.output[r],count*4)!=cudaSuccess)return 1;
        if(cudaMemcpyAsync(buffers.input[r],input[r].data(),count*4,cudaMemcpyHostToDevice,peer.streams[r])!=cudaSuccess ||
           cudaMemsetAsync(buffers.output[r],0,count*4,peer.streams[r])!=cudaSuccess || cudaStreamSynchronize(peer.streams[r])!=cudaSuccess)return 1;}
    std::printf("OPT251_PRIVATE_EAGER_START rows=512 blocks=128 threads=256 output_zero_initialized=1 HOST_TRACE_NOT_TIMING\n");std::fflush(stdout);
    Trace trace;peer.setTrace(&Trace::record,&trace);std::array<cudaError_t,2> operationErrors{};
    std::array<std::thread,2> threads;
    for(int r=0;r<2;++r)threads[r]=std::thread([&,r]{
        auto code=trace.call(r,"thread-set-device",[&]{return cudaSetDevice(r);});
        if(code==cudaSuccess)code=trace.call(r,"enqueue-call",[&]{return peer.enqueue(buffers.input[r],buffers.output[r],count,r,peer.streams[r]);});
        // Always attempt same-rank stream quiescence after submission failure.
        const auto synchronized=trace.call(r,"stream-synchronize",[&]{return cudaStreamSynchronize(peer.streams[r]);});
        operationErrors[r]=code==cudaSuccess?synchronized:code;
    });
    for(auto& thread:threads)thread.join();peer.setTrace(nullptr,nullptr);
    if(!trace.save(folder/"host-trace.json",operationErrors))return 5;
    Peer::Snapshot snapshot;e=peer.snapshotAfterQuiescence(&snapshot);
    if(e!=cudaSuccess){std::fprintf(stderr,"SNAPSHOT_API_FAILURE code=%d\n",int(e));return 1;}
    auto snapshotPath=folder/"quiescent-snapshot.json";if(std::filesystem::exists(snapshotPath))return 5;
    FILE* f=std::fopen(snapshotPath.string().c_str(),"wb");if(!f)return 5;
    std::fprintf(f,"{\"protocol\":\"opt251-quiescent-epochs-v1\",\"both_streams_quiescent\":true,\"start\":[%u,%u],\"end\":[%u,%u],\"sequence\":[%u,%u],\"fault\":[%u,%u],\"ready_timeout_code\":101,\"finish_timeout_code\":201,\"watchdog_unchanged_seconds\":30}\n",snapshot.start[0],snapshot.start[1],snapshot.end[0],snapshot.end[1],snapshot.sequence[0],snapshot.sequence[1],snapshot.fault[0],snapshot.fault[1]);
    const bool failed=std::ferror(f)!=0;const int closed=std::fclose(f);if(failed || closed)return 5;
    std::array<cudaError_t,2> copyErrors{};
    for(int r=0;r<2;++r){copyErrors[r]=cudaSetDevice(r);if(copyErrors[r]==cudaSuccess)copyErrors[r]=cudaMemcpy(actual[r].data(),buffers.output[r],count*4,cudaMemcpyDeviceToHost);
        if(copyErrors[r]==cudaSuccess && !saveRaw(folder/("family0-eager-rank"+std::to_string(r)+".bin"),actual[r]))return 5;}
    std::printf("OPT251_QUIESCENT start=%u/%u end=%u/%u sequence=%u/%u faults=%u/%u\n",snapshot.start[0],snapshot.start[1],snapshot.end[0],snapshot.end[1],snapshot.sequence[0],snapshot.sequence[1],snapshot.fault[0],snapshot.fault[1]);std::fflush(stdout);
    for(int r=0;r<2;++r)if(operationErrors[r]!=cudaSuccess || copyErrors[r]!=cudaSuccess){std::fprintf(stderr,"ACTUAL_API_FAILURE rank=%d operation=%d copy=%d\n",r,int(operationErrors[r]),int(copyErrors[r]));return 1;}
    if(trace.overflow[0] || trace.overflow[1])return 4;
    bool clean[2]{};for(int r=0;r<2;++r){e=peer.checkFaults(r,&clean[r]);if(e!=cudaSuccess)return 1;}
    if(!clean[0] || !clean[1])return 3;
    if(snapshot.start!=std::array<unsigned,2>{1,1} || snapshot.end!=std::array<unsigned,2>{1,1} || snapshot.sequence!=std::array<unsigned,2>{1,1})return 4;
    for(int r=0;r<2;++r)for(size_t i=0;i<count;++i)if(bits(actual[r][i])!=bits(expected[i])){std::fprintf(stderr,"FULL_MATRIX_FAIL rank=%d index=%zu\n",r,i);return 3;}
    std::puts("OPT251_SINGLE_EAGER_MATRIX_PAIR_EXACT NO_PRIVATE_FULL_GATES_NO_SERVING_ACCEPTANCE");return 0;
}
}
int main(int argc,char** argv){
    if(argc!=2)return 2;const std::filesystem::path folder=argv[1];
    if(!std::filesystem::is_directory(folder) || !std::filesystem::is_empty(folder))return 2;
    return run(folder);
}
