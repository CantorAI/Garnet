// SPDX-License-Identifier: Apache-2.0
// Private ORIGINAL NCCL pack/SUM/unpack control. No new transport or inference selector.
#include "gpt_oss_kernels.h"
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
    size_t count;int original=0;bool acquired=false;
    std::array<cudaStream_t,2> streams{};std::array<float*,2> input{},output{};
    std::array<void*,2> scratch{};std::array<cudaGraph_t,2> graphs{};std::array<cudaGraphExec_t,2> execution{};
    explicit Control(size_t n):count(n){
        CUDA_OK(cudaGetDevice(&original));CUDA_OK(Garnet::GptOssTpAcquire());acquired=true;
        for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaStreamCreateWithFlags(&streams[r],cudaStreamNonBlocking));
            CUDA_OK(cudaMalloc(&input[r],count*4));CUDA_OK(cudaMalloc(&output[r],count*4));CUDA_OK(cudaMalloc(&scratch[r],count*4));}
    }
    template<class F> void paired(F work){
        std::array<std::thread,2> threads;
        for(int r=0;r<2;++r)threads[r]=std::thread([&,r]{CUDA_OK(cudaSetDevice(r));work(r);});
        for(auto& t:threads)t.join();
    }
    void reduce(int r){CUDA_OK(Garnet::GptOssTpAllReduceBf16(input[r],output[r],scratch[r],count,r,streams[r]));}
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
            for(size_t i=0;i<count;++i)if(bits(actual[i])!=bits(expected[i])){
                std::fprintf(stderr,"FULL_MATRIX_FAIL stem=%s rank=%d index=%zu actual=%08x expected=%08x\n",stem.c_str(),r,i,bits(actual[i]),bits(expected[i]));std::exit(3);}
            std::printf("FULL_MATRIX_PASS stem=%s rank=%d values=%zu\n",stem.c_str(),r,count);std::fflush(stdout);
        }
    }
    ~Control(){destroyGraph();for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));
        if(input[r])CUDA_OK(cudaFree(input[r]));if(output[r])CUDA_OK(cudaFree(output[r]));if(scratch[r])CUDA_OK(cudaFree(scratch[r]));if(streams[r])CUDA_OK(cudaStreamDestroy(streams[r]));}
        if(acquired)Garnet::GptOssTpRelease();CUDA_OK(cudaSetDevice(original));}
};
}
int main(int argc,char** argv){
    if(argc!=4)return 2;char* end=nullptr;const long rows=std::strtol(argv[1],&end,10);
    if(!end||*end||(rows!=224&&rows!=512&&rows!=3584&&rows!=4096))return 2;
    const std::string mode=argv[2];if(mode!="parity"&&mode!="cost")return 2;
    const std::filesystem::path folder=argv[3];if(!std::filesystem::is_directory(folder)||!std::filesystem::is_empty(folder))return 2;
    // No silent custom peer or forced algorithm substitute for the actual NCCL control.
    if(const char* v=std::getenv("GARNET_GPT_OSS_DIRECT_ALLREDUCE"))if(std::strcmp(v,"0"))return 2;
    if(const char* v=std::getenv("NCCL_ALGO"))if(*v)return 2;
    const char* proto=std::getenv("NCCL_PROTO");
    if(proto&&std::strcmp(proto,"LL")&&std::strcmp(proto,"Simple"))return 2;
    const size_t count=size_t(rows)*2880;
    std::printf("ORIGINAL_NCCL_BF16_CONTROL rows=%ld mode=%s protocol=%s PACK_SUM_UNPACK NO_PEER_SELECTION\n",rows,mode.c_str(),proto?proto:"AUTO_UNSET");
    Control control(count);std::array<std::vector<float>,2> host{std::vector<float>(count),std::vector<float>(count)};std::vector<float> expected(count);
    for(int family=0;family<6;++family)for(int graph=0;graph<2;++graph){
        const int offset=graph?131:0;
        for(size_t i=0;i<count;++i){const auto pair=inputs(i,family,offset);host[0][i]=pair[0];host[1][i]=pair[1];volatile float sum=pair[0]+pair[1];expected[i]=rounded(sum);}
        control.upload(host);
        // NCCL establishes its lazy connections on the preceding eager pair.
        // Capture only after that pair has completed and passed the full check.
        if(graph && family==0)control.capture(1);
        if(graph)control.replay(3);else control.paired([&](int r){control.reduce(r);CUDA_OK(cudaStreamSynchronize(control.streams[r]));});
        control.check(folder,"family"+std::to_string(family)+(graph?"-graph":"-eager"),expected);
    }
    std::printf("NCCL_CONTROL_FINITE_PARITY_PASS matrices=24 values_per_matrix=%zu\n",count);
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
        std::fprintf(f,"{\"rows\":%ld,\"elements\":%zu,\"graph_operations\":72,\"replays_per_trial\":3,\"warmups\":3,\"protocol\":\"%s\",\"microseconds_per_original_pack_nccl_unpack\":[",rows,count,proto?proto:"AUTO_UNSET");
        for(int i=0;i<9;++i)std::fprintf(f,"%s%.17g",i?",":"",times[i]);
        std::fprintf(f,"],\"scope\":\"Native isolated diagnostic only; requires external idle/source/version/dispatch controls; no serving speed claim\"}\n");if(std::fclose(f))return 5;
        std::puts("NCCL_CONTROL_COST_COMPLETE 9_TRIALS FULL_OUTPUTS NO_SERVING_SPEED_CLAIM");
    }
    return 0;
}
