// SPDX-License-Identifier: Apache-2.0
// Private exact FP32 transport experiment. No inference selection or compression.
#include "fp32_peer_native.cuh"
#include "tp_direct.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>
#define CUDA_OK(call) do {auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));std::exit(1);}}while(0)
namespace {
uint32_t bits(float f){uint32_t n;std::memcpy(&n,&f,4);return n;}
float value(uint32_t n){float f;std::memcpy(&f,&n,4);return f;}
constexpr uint32_t sentinel=0xc2f68000U;
std::array<float,2> inputs(size_t i,int family,int offset){
    const uint32_t j=uint32_t((i+offset)%4096),h=j*2654435761U+uint32_t(family)*2246822519U;
    const uint32_t a=(h&0x80000000U)|((87U+(j/32+uint32_t(family)*19)%80)<<23)|(h&0x7fffffU);
    uint32_t b=a;
    switch(family){
    case 0:break;
    case 1:b=a^0x80000000U;break;
    case 2:b=a+1;break;
    case 3:b=(a&0x80000000U)|((((a>>23)&255U)-24)<<23)|((h>>7)&0x7fffffU);break;
    case 4:b=(h&0x80000000U)^0x80000000U;b|=(87U+(j/16+23)%80)<<23;b|=(h>>3)&0x7fffffU;break;
    default:return {value((j&1U)<<31),value((j&2U)<<30)};
    }
    return {value(a),value(b)};
}
void write(const std::filesystem::path& p,const std::vector<float>& v){
    if(std::filesystem::exists(p))std::exit(5);auto* f=std::fopen(p.string().c_str(),"wb");if(!f)std::exit(5);
    bool failed=std::fwrite(v.data(),4,v.size(),f)!=v.size();int closed=std::fclose(f);if(failed||closed)std::exit(5);
}
void env(const char* name,const char* v){
#ifdef _WIN32
    if(_putenv_s(name,v))std::exit(5);
#else
    if(setenv(name,v,1))std::exit(5);
#endif
}
struct Control {
    size_t count,capacity;bool owned;int initial=0;
    GarnetFp32Prototype::Fp32PeerNative peer;
    std::array<cudaStream_t,2> streams{};std::array<float*,2> input{},output{};
    std::array<std::vector<float>,2> uploaded;
    std::array<cudaGraph_t,2> graphs{};std::array<cudaGraphExec_t,2> execution{};
    Control(size_t n,int blocks,bool candidate,bool external=false):count(n),capacity(n<=256*2880?256*2880:512*2880),owned(candidate){
        CUDA_OK(cudaGetDevice(&initial));
        if(owned){CUDA_OK(peer.initialize(capacity,blocks,-1,external));streams=peer.streams;}
        else {
            env("GARNET_GPT_OSS_DIRECT_MAX_BATCH",capacity==256*2880?"256":"512");
            env("GARNET_GPT_OSS_DIRECT_BATCH_ALLREDUCE","1");env("GARNET_GPT_OSS_DIRECT_LARGE_BATCH_ALLREDUCE","1");
            env("GARNET_GPT_OSS_DIRECT_BATCH_CTAS","32");CUDA_OK(Garnet::GptOssTpDirectAcquire());
            for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaStreamCreateWithFlags(&streams[r],cudaStreamNonBlocking));}
        }
        for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaMalloc(&input[r],capacity*4));CUDA_OK(cudaMalloc(&output[r],capacity*4));}
        CUDA_OK(cudaSetDevice(initial));
        std::printf("FP32_CONTROL count=%zu capacity=%zu owned=%d candidate_ctas=%d original_ctas=32 owned_bytes_per_rank=%zu mapped_bytes=%zu\n",
            count,capacity,owned,blocks,owned?peer.ownedBytesPerRank():capacity*4,owned?peer.mappedBytes():size_t(0));
    }
    template<class F>void paired(F f){std::array<std::thread,2> t;for(int r=0;r<2;++r)t[r]=std::thread([&,r]{CUDA_OK(cudaSetDevice(r));f(r);});for(auto& thread:t)thread.join();}
    void reduce(int r){if(owned)CUDA_OK(peer.enqueue(input[r],output[r],count,r,streams[r]));else CUDA_OK(Garnet::GptOssTpDirectAllReduce(input[r],output[r],count,r,streams[r]));}
    void upload(const std::array<std::vector<float>,2>& host){
        uploaded=host;
        std::vector<float> tail(capacity,value(sentinel));
        paired([&](int r){CUDA_OK(cudaMemcpyAsync(input[r],host[r].data(),capacity*4,cudaMemcpyHostToDevice,streams[r]));
            CUDA_OK(cudaMemcpyAsync(output[r],tail.data(),capacity*4,cudaMemcpyHostToDevice,streams[r]));CUDA_OK(cudaStreamSynchronize(streams[r]));});
    }
    void destroyGraph(){for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));if(execution[r]){CUDA_OK(cudaGraphExecDestroy(execution[r]));execution[r]=nullptr;}
        if(graphs[r]){CUDA_OK(cudaGraphDestroy(graphs[r]));graphs[r]=nullptr;}}}
    void capture(int operations){destroyGraph();paired([&](int r){CUDA_OK(cudaStreamBeginCapture(streams[r],cudaStreamCaptureModeThreadLocal));
        for(int i=0;i<operations;++i)reduce(r);CUDA_OK(cudaStreamEndCapture(streams[r],&graphs[r]));CUDA_OK(cudaGraphInstantiate(&execution[r],graphs[r],0));});}
    void replay(int repeats){paired([&](int r){for(int i=0;i<repeats;++i)CUDA_OK(cudaGraphLaunch(execution[r],streams[r]));CUDA_OK(cudaStreamSynchronize(streams[r]));});}
    void check(const std::filesystem::path& folder,const std::string& stem,const std::vector<float>& expected){
        std::vector<float> result(capacity);
        for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaMemcpy(result.data(),output[r],capacity*4,cudaMemcpyDeviceToHost));
            write(folder/(stem+"-input"+std::to_string(r)+".f32"),uploaded[r]);
            write(folder/(stem+"-rank"+std::to_string(r)+".f32"),result);
            if(owned){bool clean=false;CUDA_OK(peer.checkFaults(r,&clean));if(!clean)std::exit(3);}
            for(size_t i=0;i<capacity;++i)if(bits(result[i])!=bits(expected[i])){
                std::fprintf(stderr,"FULL_MATRIX_FAIL stem=%s rank=%d index=%zu actual=%08x expected=%08x\n",stem.c_str(),r,i,bits(result[i]),bits(expected[i]));std::exit(3);}
            std::printf("FULL_MATRIX_PASS stem=%s rank=%d values=%zu active=%zu\n",stem.c_str(),r,capacity,count);std::fflush(stdout);
        }
    }
    ~Control(){destroyGraph();for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaFree(input[r]));CUDA_OK(cudaFree(output[r]));}
        if(owned)CUDA_OK(peer.close());else {
            Garnet::GptOssTpDirectRelease();
            // Isolated fresh-process control owns these links; original release leaves them enabled.
            for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaStreamDestroy(streams[r]));CUDA_OK(cudaDeviceDisablePeerAccess(1-r));}
        }
        CUDA_OK(cudaSetDevice(initial));}
};
}
int main(int argc,char** argv){
    if(argc!=6||std::fegetround()!=FE_TONEAREST)return 2;
    char* end=nullptr;const long rows=std::strtol(argv[1],&end,10);if(!end||*end||(rows!=224&&rows!=512))return 2;
    const long blocks=std::strtol(argv[2],&end,10);if(!end||*end||(blocks!=32&&blocks!=64))return 2;
    const std::string arm=argv[3],mode=argv[4];if((arm!="original"&&arm!="peer")||(mode!="parity"&&mode!="cost"))return 2;
    const std::filesystem::path folder=argv[5];if(!std::filesystem::is_directory(folder)||!std::filesystem::is_empty(folder))return 2;
    const bool owned=arm=="peer";const size_t count=size_t(rows)*2880,capacity=rows<=256?256*2880:512*2880;
    std::printf("EXACT_FP32_TRANSPORT rows=%ld blocks=%ld arm=%s mode=%s NO_INFERENCE_SELECTION\n",rows,blocks,arm.c_str(),mode.c_str());
    int initial=0;CUDA_OK(cudaGetDevice(&initial));
    if(owned){
        for(int simulated:{0,1}){GarnetFp32Prototype::Fp32PeerNative fail;
            if(fail.initialize(capacity,blocks,simulated)!=cudaErrorUnknown||fail.ready())return 4;
            int device=-1;CUDA_OK(cudaGetDevice(&device));if(device!=initial)return 4;CUDA_OK(fail.close());}
        GarnetFp32Prototype::Fp32PeerNative invalid;
        for(size_t n:{size_t(0),size_t(3),GarnetFp32Prototype::kMaximumPairElements+4,SIZE_MAX})if(invalid.initialize(n,blocks)!=cudaErrorInvalidValue)return 4;
        for(int n:{0,16,129})if(invalid.initialize(capacity,n)!=cudaErrorInvalidValue)return 4;
    }
    std::array<std::vector<float>,2> host{std::vector<float>(capacity),std::vector<float>(capacity)};std::vector<float> expected(capacity,value(sentinel));
    auto fill=[&](int family,int offset){
        std::fill(expected.begin(),expected.end(),value(sentinel));
        for(size_t i=0;i<capacity;++i){const auto pair=inputs(i,family,offset);host[0][i]=pair[0];host[1][i]=pair[1];
            if(i<count){double sum=double(pair[0])+double(pair[1]);expected[i]=float(sum);if(!std::isfinite(expected[i]))std::exit(3);}}
    };
    {
        Control c(count,blocks,owned);
        if(owned){
            GarnetFp32Prototype::Fp32PeerNative competing;if(competing.initialize(capacity,blocks)!=cudaErrorNotReady)return 4;
            c.paired([&](int r){
                auto reject=[&](const float* x,float* y,size_t n,int rank,cudaStream_t stream){if(c.peer.enqueue(x,y,n,rank,stream)!=cudaErrorInvalidValue)std::exit(4);};
                reject(nullptr,c.output[r],count,r,c.streams[r]);reject(c.input[r],nullptr,count,r,c.streams[r]);
                reject(c.input[r]+1,c.output[r],count,r,c.streams[r]);reject(c.input[r],c.output[r]+1,count,r,c.streams[r]);
                for(size_t bad:{size_t(0),size_t(3),capacity+4,SIZE_MAX})reject(c.input[r],c.output[r],bad,r,c.streams[r]);
                for(int bad:{-1,2})reject(c.input[r],c.output[r],count,bad,c.streams[r]);reject(c.input[r],c.output[r],count,r,nullptr);
                if(c.peer.enqueue(c.input[1-r],c.output[r],count,r,c.streams[r])!=cudaErrorInvalidDevicePointer)std::exit(4);
                cudaStream_t foreign=nullptr;CUDA_OK(cudaStreamCreateWithFlags(&foreign,cudaStreamNonBlocking));
                reject(c.input[r],c.output[r],count,r,foreign);CUDA_OK(cudaStreamDestroy(foreign));
            });
            CUDA_OK(cudaSetDevice(0));CUDA_OK(cudaStreamBeginCapture(c.streams[0],cudaStreamCaptureModeThreadLocal));
            if(c.peer.reset()!=cudaErrorStreamCaptureUnsupported||c.peer.close()!=cudaErrorStreamCaptureUnsupported||!c.peer.ready())return 4;
            cudaGraph_t empty=nullptr;CUDA_OK(cudaStreamEndCapture(c.streams[0],&empty));CUDA_OK(cudaGraphDestroy(empty));
        }
        for(int family=0;family<6;++family)for(int graph=0;graph<2;++graph){
            fill(family,graph?131:0);const std::string stem="family"+std::to_string(family)+(graph?"-graph":"-eager");
            c.upload(host);if(graph&&family==0)c.capture(1);
            if(graph)c.replay(3);else c.paired([&](int r){c.reduce(r);CUDA_OK(cudaStreamSynchronize(c.streams[r]));});
            c.check(folder,stem,expected);
        }
        if(owned){CUDA_OK(c.peer.reset());fill(4,257);c.upload(host);c.replay(3);c.check(folder,"life-reset",expected);}
        if(mode=="cost"){
            fill(4,521);c.upload(host);constexpr int operations=72,replays=3;c.capture(operations);
            for(int warm=0;warm<3;++warm)c.replay(replays);std::array<double,9> times{};
            for(int trial=0;trial<9;++trial){const auto begin=std::chrono::steady_clock::now();c.replay(replays);
                times[trial]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/(operations*replays);
                c.check(folder,"cost-trial"+std::to_string(trial),expected);}
            auto* f=std::fopen((folder/"cost.json").string().c_str(),"wb");if(!f)return 5;
            std::fprintf(f,"{\"protocol\":\"exact-fp32-copy-sum-native-v1\",\"arm\":\"%s\",\"rows\":%ld,\"candidate_blocks\":%ld,\"original_blocks\":32,\"capacity\":%zu,\"graph_operations\":72,\"replays\":3,\"warmups\":3,\"microseconds\":[",arm.c_str(),rows,blocks,capacity);
            for(int i=0;i<9;++i)std::fprintf(f,"%s%.17g",i?",":"",times[i]);std::fprintf(f,"],\"scope\":\"isolated native diagnostic; not serving qualification\"}\n");if(std::fclose(f))return 5;
        }
    }
    for(int graph=0;graph<2;++graph){Control reacquired(count,blocks,owned);fill(2,513+graph);reacquired.upload(host);
        if(graph){reacquired.capture(1);reacquired.replay(3);}else reacquired.paired([&](int r){reacquired.reduce(r);CUDA_OK(cudaStreamSynchronize(reacquired.streams[r]));});
        reacquired.check(folder,std::string("life-reacquire")+(graph?"-graph":"-eager"),expected);}
    if(owned){
        for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaDeviceEnablePeerAccess(1-r,0));}
        {Control external(count,blocks,true,true);fill(3,1021);external.upload(host);external.paired([&](int r){external.reduce(r);CUDA_OK(cudaStreamSynchronize(external.streams[r]));});external.check(folder,"life-external",expected);}
        for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaDeviceDisablePeerAccess(1-r));}
    }
    CUDA_OK(cudaSetDevice(initial));std::printf("FP32_PRIVATE_COMPLETE finite_matrices=%d cost_matrices=%d BOTH_RANKS_FULL_CAPACITY_TAILS NO_INFERENCE_SELECTION\n",owned?32:28,mode=="cost"?18:0);return 0;
}
