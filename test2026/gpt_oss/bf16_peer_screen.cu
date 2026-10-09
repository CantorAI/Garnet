// SPDX-License-Identifier: Apache-2.0
// Private owned BF16 pack/peer/SUM/unpack. No production selection or NCCL substitute.
#include "bf16_peer_native.cuh"
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#define CUDA_OK(call) do {const auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));std::exit(1);}}while(0)
namespace {
uint32_t bits(float f){uint32_t n;std::memcpy(&n,&f,4);return n;}
float value(uint32_t n){float f;std::memcpy(&f,&n,4);return f;}
uint16_t finite(uint16_t n){return (n&0x7f80U)==0x7f80U?uint16_t(n^0x80U):n;}
float rounded(float f){const uint32_t n=bits(f);return value((n+0x7fffU+((n>>16)&1U))&0xffff0000U);}
std::array<float,2> inputs(size_t i,int family,int offset){
    const uint16_t a=finite(uint16_t(i+offset));uint16_t b=0;
    switch(family){
        case 0:b=a;break;case 1:b=uint16_t(a^0x8000U);break;
        case 2:b=finite(uint16_t(a+1));break;case 3:b=uint16_t(a&0x8000U);break;
        case 4:b=uint16_t((a&0x8000U)|1U);break;default:b=uint16_t((a&0x8000U)|0x3f80U);break;
    }
    return {value(uint32_t(a)<<16),value(uint32_t(b)<<16)};
}
void save(const std::filesystem::path& path,const std::vector<float>& v){
    if(std::filesystem::exists(path))std::exit(5);FILE* f=std::fopen(path.string().c_str(),"wb");if(!f)std::exit(5);
    const bool failed=std::fwrite(v.data(),4,v.size(),f)!=v.size();const int closed=std::fclose(f);if(failed||closed)std::exit(5);
}
struct Control {
    size_t count;int original=0;GarnetPrototype::Bf16PeerNative peer;
    std::array<cudaStream_t,2> streams{};std::array<float*,2> input{},output{};
    std::array<cudaGraph_t,2> graphs{};std::array<cudaGraphExec_t,2> execution{};
    explicit Control(size_t n,int blocks,bool external=false):count(n){
        CUDA_OK(cudaGetDevice(&original));CUDA_OK(peer.initialize(count,blocks,-1,external));streams=peer.streams;
        std::printf("OWNED_BUFFERS bytes_per_rank=%zu mapped_bytes=%zu blocks=%d EXCLUSIVE_FIXED_STREAMS\n",peer.ownedBytesPerRank(),peer.mappedBytes(),blocks);
        for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));
            CUDA_OK(cudaMalloc(&input[r],count*4));CUDA_OK(cudaMalloc(&output[r],count*4));}
    }
    template<class F> void paired(F work){
        std::array<std::thread,2> threads;
        for(int r=0;r<2;++r)threads[r]=std::thread([&,r]{CUDA_OK(cudaSetDevice(r));work(r);});
        for(auto& t:threads)t.join();
    }
    void reduce(int r){CUDA_OK(peer.enqueue(input[r],output[r],count,r,streams[r]));}
    void upload(const std::array<std::vector<float>,2>& host){
        paired([&](int r){CUDA_OK(cudaMemcpyAsync(input[r],host[r].data(),count*4,cudaMemcpyHostToDevice,streams[r]));CUDA_OK(cudaStreamSynchronize(streams[r]));});
    }
    void destroyGraph(){for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));if(execution[r]){CUDA_OK(cudaGraphExecDestroy(execution[r]));execution[r]=nullptr;}
        if(graphs[r]){CUDA_OK(cudaGraphDestroy(graphs[r]));graphs[r]=nullptr;}}}
    void capture(int operations){destroyGraph();paired([&](int r){
        CUDA_OK(cudaStreamBeginCapture(streams[r],cudaStreamCaptureModeThreadLocal));
        for(int i=0;i<operations;++i)reduce(r);
        CUDA_OK(cudaStreamEndCapture(streams[r],&graphs[r]));CUDA_OK(cudaGraphInstantiate(&execution[r],graphs[r],0));});}
    void replay(int repeats){paired([&](int r){for(int i=0;i<repeats;++i)CUDA_OK(cudaGraphLaunch(execution[r],streams[r]));CUDA_OK(cudaStreamSynchronize(streams[r]));});}
    void check(const std::filesystem::path& folder,const std::string& stem,const std::vector<float>& expected){
        std::vector<float> actual(count);
        for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaMemcpy(actual.data(),output[r],count*4,cudaMemcpyDeviceToHost));
            // Whole actual matrix is preserved BEFORE evaluating numerical success.
            save(folder/(stem+"-rank"+std::to_string(r)+".bin"),actual);
            bool clean=false;CUDA_OK(peer.checkFaults(r,&clean));if(!clean)std::exit(3);
            for(size_t i=0;i<count;++i)if(bits(actual[i])!=bits(expected[i])){
                std::fprintf(stderr,"FULL_MATRIX_FAIL stem=%s rank=%d index=%zu actual=%08x expected=%08x\n",stem.c_str(),r,i,bits(actual[i]),bits(expected[i]));std::exit(3);}
            std::printf("FULL_MATRIX_PASS stem=%s rank=%d values=%zu\n",stem.c_str(),r,count);std::fflush(stdout);
        }
    }
    ~Control(){destroyGraph();for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));
        if(input[r])CUDA_OK(cudaFree(input[r]));if(output[r])CUDA_OK(cudaFree(output[r]));}
        CUDA_OK(peer.close());CUDA_OK(cudaSetDevice(original));}
};
}
int main(int argc,char** argv){
    if(argc!=5)return 2;char* end=nullptr;const long rows=std::strtol(argv[1],&end,10);
    if(!end||*end||(rows!=224&&rows!=512&&rows!=3584&&rows!=4096))return 2;
    end=nullptr;const long blocks=std::strtol(argv[2],&end,10);if(!end||*end||(blocks!=32&&blocks!=64&&blocks!=128&&blocks!=188))return 2;
    const std::string mode=argv[3];if(mode!="parity"&&mode!="cost")return 2;
    const std::filesystem::path folder=argv[4];if(!std::filesystem::is_directory(folder)||!std::filesystem::is_empty(folder))return 2;
    const size_t count=size_t(rows)*2880;
    std::printf("OWNED_BF16_PEER_CONTROL rows=%ld blocks=%ld mode=%s protocol=BF16_PEER_MAPPED_COOPERATIVE NO_INFERENCE_SELECTION\n",rows,blocks,mode.c_str());
    int initial=0;CUDA_OK(cudaGetDevice(&initial));
    for(int simulated:{0,1}){
        GarnetPrototype::Bf16PeerNative failed;
        if(failed.initialize(count,blocks,simulated)!=cudaErrorUnknown || failed.ready())return 4;
        int current=-1;CUDA_OK(cudaGetDevice(&current));if(current!=initial)return 4;
        CUDA_OK(failed.close());
    }
    GarnetPrototype::Bf16PeerNative invalid;
    for(size_t bad:{size_t(0),size_t(7),GarnetPrototype::kMaximumPairElements+8,SIZE_MAX})
        if(invalid.initialize(bad,blocks)!=cudaErrorInvalidValue)return 4;
    for(int bad:{0,16,129})if(invalid.initialize(count,bad)!=cudaErrorInvalidValue)return 4;
    std::array<std::vector<float>,2> host{std::vector<float>(count),std::vector<float>(count)};std::vector<float> expected(count);
    {
    Control control(count,blocks);
    GarnetPrototype::Bf16PeerNative competing;
    if(competing.initialize(count,blocks)!=cudaErrorNotReady)return 4;
    control.paired([&](int r){
        auto reject=[&](const float* x,float* y,size_t n,int rank,cudaStream_t stream){if(control.peer.enqueue(x,y,n,rank,stream)!=cudaErrorInvalidValue)std::exit(4);};
        reject(nullptr,control.output[r],count,r,control.streams[r]);reject(control.input[r],nullptr,count,r,control.streams[r]);
        reject(control.input[r]+1,control.output[r],count,r,control.streams[r]);reject(control.input[r],control.output[r]+1,count,r,control.streams[r]);
        for(size_t bad:{size_t(0),size_t(7),count+8,SIZE_MAX})reject(control.input[r],control.output[r],bad,r,control.streams[r]);
        for(int bad:{-1,2})reject(control.input[r],control.output[r],count,bad,control.streams[r]);
        reject(control.input[r],control.output[r],count,r,nullptr);
    });
    for(int family=0;family<6;++family)for(int graph=0;graph<2;++graph){
        const int offset=graph?131:0;
        for(size_t i=0;i<count;++i){const auto pair=inputs(i,family,offset);host[0][i]=pair[0];host[1][i]=pair[1];volatile float sum=pair[0]+pair[1];expected[i]=rounded(sum);}
        control.upload(host);
        // First complete eager pair proves data/signals before capture.
        if(graph && family==0)control.capture(1);
        if(graph)control.replay(3);else control.paired([&](int r){control.reduce(r);CUDA_OK(cudaStreamSynchronize(control.streams[r]));});
        control.check(folder,"family"+std::to_string(family)+(graph?"-graph":"-eager"),expected);
    }
    // Same captured executable is retained across a fully quiescent epoch reset.
    for(size_t i=0;i<count;++i){const auto pair=inputs(i,5,257);host[0][i]=pair[0];host[1][i]=pair[1];volatile float sum=pair[0]+pair[1];expected[i]=rounded(sum);}
    CUDA_OK(control.peer.reset());control.upload(host);control.replay(3);control.check(folder,"life-reset",expected);
    std::printf("PEER_FINITE_PARITY_PASS matrices=26 values_per_matrix=%zu API_GUARDS_PARTIAL_LEASE_RESET\n",count);
    if(mode=="cost"){
        for(size_t i=0;i<count;++i){host[0][i]=float(int(i%127)-63)/64;host[1][i]=float(int(i%91)-45)/32;volatile float sum=host[0][i]+host[1][i];expected[i]=rounded(sum);}
        control.upload(host);constexpr int operations=72,replays=3;control.capture(operations);
        for(int warm=0;warm<3;++warm)control.replay(replays);
        std::array<double,9> times{};
        for(int trial=0;trial<9;++trial){
            const auto begin=std::chrono::steady_clock::now();control.replay(replays);
            times[trial]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/(operations*replays);
            control.check(folder,"cost-trial"+std::to_string(trial),expected);
        }
        const auto path=folder/"cost.json";FILE* f=std::fopen(path.string().c_str(),"wb");if(!f)return 5;
        std::fprintf(f,"{\"rows\":%ld,\"blocks\":%ld,\"elements\":%zu,\"graph_operations\":72,\"replays_per_trial\":3,\"warmups\":3,\"protocol\":\"BF16_PEER_MAPPED_COOPERATIVE\",\"microseconds_per_owned_pack_peer_unpack\":[",rows,blocks,count);
        for(int i=0;i<9;++i)std::fprintf(f,"%s%.17g",i?",":"",times[i]);
        std::fprintf(f,"],\"scope\":\"Native isolated diagnostic only; requires external idle/source/version/dispatch controls; no serving speed claim\"}\n");if(std::fclose(f))return 5;
        std::puts("PEER_CONTROL_COST_COMPLETE 9_TRIALS FULL_OUTPUTS NO_SERVING_SPEED_CLAIM");
    }
    }
    auto lifetime=[&](const char* stem,int family,int first,bool external){
        Control again(count,blocks,external);
        for(int graph=0;graph<2;++graph){
            for(size_t i=0;i<count;++i){auto pair=inputs(i,family,first+graph);host[0][i]=pair[0];host[1][i]=pair[1];volatile float sum=pair[0]+pair[1];expected[i]=rounded(sum);}
            again.upload(host);
            if(graph){again.capture(1);again.replay(3);}else again.paired([&](int r){again.reduce(r);CUDA_OK(cudaStreamSynchronize(again.streams[r]));});
            again.check(folder,std::string(stem)+(graph?"-graph":"-eager"),expected);
        }
    };
    lifetime("life-reacquire",4,513,false);
    for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaDeviceEnablePeerAccess(1-r,0));}
    lifetime("life-external",2,1021,true);
    for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaDeviceDisablePeerAccess(1-r));}
    CUDA_OK(cudaSetDevice(initial));
    std::printf("OWNED_PEER_ALL_GATES_COMPLETE finite_lifetime_matrices=34 cost_matrices=%d NO_INFERENCE_SELECTION\n",mode=="cost"?18:0);
    return 0;
}
