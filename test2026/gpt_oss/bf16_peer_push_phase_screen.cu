// SPDX-License-Identifier: Apache-2.0
// Production-owner grid handshake CPU-bit, retained-graph and phase lifetime gate.
#include "bf16_peer_push_native.cuh"
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#define CUDA_OK(call) do {const auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));std::exit(1);}}while(0)
namespace {
constexpr size_t capacity=size_t(4096)*2880;
int ctas=128;bool gridMode=false;
constexpr uint32_t sentinel=0xa5a5a5a5U;
uint32_t bits(float f){uint32_t b;std::memcpy(&b,&f,4);return b;}
float value(uint32_t b){float f;std::memcpy(&f,&b,4);return f;}
uint16_t finite(uint16_t b){return (b&0x7f80U)==0x7f80U?uint16_t(b^0x80U):b;}
float rounded(float f){auto b=bits(f);return value((b+0x7fffU+((b>>16)&1U))&0xffff0000U);}
std::array<float,2> input(size_t i,int family,int offset){
    const auto a=finite(uint16_t(i+offset));uint16_t b=0;
    switch(family){case 0:b=a;break;case 1:b=uint16_t(a^0x8000U);break;
        case 2:b=finite(uint16_t(a+1));break;case 3:b=uint16_t(a&0x8000U);break;
        case 4:b=uint16_t((a&0x8000U)|1U);break;default:b=uint16_t((a&0x8000U)|0x3f80U);break;}
    return {value(uint32_t(a)<<16),value(uint32_t(b)<<16)};
}
template<class F> void paired(F work){
    std::array<std::thread,2> threads;
    for(int r=0;r<2;++r)threads[r]=std::thread([&,r]{CUDA_OK(cudaSetDevice(r));work(r);});
    for(auto& t:threads)t.join();
}
struct PhaseControl {
    int original=0;GarnetPushPrototype::Bf16PeerNative peer;
    GarnetPushPrototype::Bf16PeerNative::PhaseStreams borrowed{};
    std::array<float*,2> x{},y{};
    std::array<std::array<cudaGraph_t,2>,2> graphs{};
    std::array<std::array<cudaGraphExec_t,2>,2> execution{};
    std::array<size_t,2> counts{};
    PhaseControl(size_t first,size_t second,bool external):counts{first,second}{
        CUDA_OK(cudaGetDevice(&original));
        for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));
            for(int phase=0;phase<2;++phase)CUDA_OK(cudaStreamCreateWithFlags(&borrowed[phase][r],cudaStreamNonBlocking));
            CUDA_OK(cudaMalloc(&x[r],capacity*4));CUDA_OK(cudaMalloc(&y[r],capacity*4));}
        CUDA_OK(cudaSetDevice(original));
        auto invalid=borrowed;invalid[1][0]=invalid[0][0];
        if(peer.initialize(capacity,ctas,-1,external,&invalid,gridMode)!=cudaErrorInvalidValue)std::exit(4);
#if CUDART_VERSION >= 13000
        invalid=borrowed;invalid[0][0]=borrowed[0][1];
        if(peer.initialize(capacity,ctas,-1,external,&invalid,gridMode)!=cudaErrorInvalidDevice||peer.ready())std::exit(4);
#endif
        for(int simulated:{0,1}){
            if(peer.initialize(capacity,ctas,simulated,external,&borrowed,gridMode)!=cudaErrorUnknown||peer.ready())std::exit(4);
            CUDA_OK(peer.close());verifyBorrowed();}
        CUDA_OK(peer.initialize(capacity,ctas,-1,external,&borrowed,gridMode));
        if(peer.bindBorrowedPhase(-1)!=cudaErrorInvalidValue||peer.bindBorrowedPhase(2)!=cudaErrorInvalidValue)std::exit(4);
        std::printf("VARIABLE_OWNER capacity=%zu bytes_per_rank=%zu mapped=%zu borrowed_phases=2 first=%zu second=%zu\n",
            capacity,peer.ownedBytesPerRank(),peer.mappedBytes(),first,second);
    }
    void verifyBorrowed(){
        for(int phase=0;phase<2;++phase)for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));
            unsigned flags=0;CUDA_OK(cudaStreamGetFlags(borrowed[phase][r],&flags));
            if(!(flags&cudaStreamNonBlocking))std::exit(4);CUDA_OK(cudaStreamSynchronize(borrowed[phase][r]));}
        CUDA_OK(cudaSetDevice(original));
    }
    void capture(int phase){
        CUDA_OK(peer.bindBorrowedPhase(phase));
        paired([&](int r){
            const auto stream=borrowed[phase][r];
            // Inactive aliases must be rejected before any CUDA work is queued.
            if(peer.enqueue(x[r],y[r],counts[phase],r,borrowed[1-phase][r])!=cudaErrorInvalidValue)std::exit(4);
            CUDA_OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            if(r==0){
                if(peer.bindBorrowedPhase(1-phase)!=cudaErrorStreamCaptureUnsupported ||
                    peer.reset()!=cudaErrorStreamCaptureUnsupported ||
                    peer.close()!=cudaErrorStreamCaptureUnsupported ||!peer.ready())std::exit(4);
            }
            for(int i=0;i<phase+2;++i)CUDA_OK(peer.enqueue(x[r],y[r],counts[phase],r,stream));
            CUDA_OK(cudaStreamEndCapture(stream,&graphs[phase][r]));
            CUDA_OK(cudaGraphInstantiate(&execution[phase][r],graphs[phase][r],0));
        });
    }
    void run(const std::filesystem::path& folder,const std::string& stem,int phase,int family,int offset,bool graph){
        CUDA_OK(peer.bindBorrowedPhase(phase));const size_t count=counts[phase];
        std::array<std::vector<float>,2> host{std::vector<float>(count),std::vector<float>(count)};
        std::vector<float> expected(capacity,value(sentinel)),actual(capacity);
        for(size_t i=0;i<count;++i){const auto p=input(i,family,offset);host[0][i]=p[0];host[1][i]=p[1];volatile float sum=p[0]+p[1];expected[i]=rounded(sum);}
        paired([&](int r){const auto stream=borrowed[phase][r];
            CUDA_OK(cudaMemcpyAsync(x[r],host[r].data(),count*4,cudaMemcpyHostToDevice,stream));
            CUDA_OK(cudaMemsetAsync(y[r],0xa5,capacity*4,stream));
            if(graph){for(int i=0;i<3;++i)CUDA_OK(cudaGraphLaunch(execution[phase][r],stream));}
            else CUDA_OK(peer.enqueue(x[r],y[r],count,r,stream));
            CUDA_OK(cudaStreamSynchronize(stream));});
        for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaMemcpy(actual.data(),y[r],capacity*4,cudaMemcpyDeviceToHost));
            const auto path=folder/(stem+"-rank"+std::to_string(r)+".bin");if(std::filesystem::exists(path))std::exit(5);
            FILE* f=std::fopen(path.string().c_str(),"wb");if(!f)std::exit(5);
            const bool shortWrite=std::fwrite(actual.data(),4,capacity,f)!=capacity;const auto closed=std::fclose(f);if(shortWrite||closed)std::exit(5);
            bool clean=false;CUDA_OK(peer.checkFaults(r,&clean));if(!clean)std::exit(3);
            for(size_t i=0;i<capacity;++i)if(bits(actual[i])!=bits(expected[i])){
                std::fprintf(stderr,"VARIABLE_FULL_MATRIX_FAIL stem=%s rank=%d index=%zu active=%zu actual=%08x expected=%08x\n",
                    stem.c_str(),r,i,count,bits(actual[i]),bits(expected[i]));std::exit(3);}
            std::printf("VARIABLE_FULL_MATRIX_PASS stem=%s rank=%d active=%zu saved=%zu BOTH_PHASE_GRAPHS_RETAINED\n",stem.c_str(),r,count,capacity);std::fflush(stdout);
        }
    }
    ~PhaseControl(){
        CUDA_OK(peer.bindBorrowedPhase(0));
        for(int phase=0;phase<2;++phase)for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));
            if(execution[phase][r])CUDA_OK(cudaGraphExecDestroy(execution[phase][r]));
            if(graphs[phase][r])CUDA_OK(cudaGraphDestroy(graphs[phase][r]));}
        CUDA_OK(peer.close());verifyBorrowed();
        for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaFree(x[r]));CUDA_OK(cudaFree(y[r]));
            for(int phase=0;phase<2;++phase)CUDA_OK(cudaStreamDestroy(borrowed[phase][r]));}
        CUDA_OK(cudaSetDevice(original));
    }
};
}
int main(int argc,char** argv){
    if(argc!=5)return 2;ctas=std::stoi(argv[3]);const std::string mode=argv[4];
    if((ctas!=64&&ctas!=128&&ctas!=188)||mode!="1")return 2;gridMode=mode=="1";
    const std::string scope=argv[1];if(scope!="short"&&scope!="long")return 2;
    const std::filesystem::path folder=argv[2];if(!std::filesystem::is_directory(folder)||!std::filesystem::is_empty(folder))return 2;
    const size_t first=(scope=="short"?4096:3584)*size_t(2880),second=(scope=="short"?512:224)*size_t(2880);
    int original=0;CUDA_OK(cudaGetDevice(&original));
    std::printf("VARIABLE_BORROWED_PEER scope=%s protocol=PRIVATE_PUSH_GRID_VARIABLE_BORROWED capacity=%zu ctas=%d grid_signals=%d NO_INFERENCE_SELECTION\n",scope.c_str(),capacity,ctas,int(gridMode));
    {
        PhaseControl control(first,second,false);
        GarnetPushPrototype::Bf16PeerNative competing;if(competing.initialize(capacity,ctas)!=cudaErrorNotReady)return 4;
        // Eager warm controls precede capture. Both graph pairs survive all data/phase updates.
        control.run(folder,"warm-prefill",0,0,0,false);control.run(folder,"warm-decode",1,1,17,false);
        control.capture(0);control.capture(1);
        for(int family=0;family<6;++family)for(int phase=0;phase<2;++phase)
            control.run(folder,"family"+std::to_string(family)+"-phase"+std::to_string(phase),phase,family,131+family*31+phase*17,true);
        CUDA_OK(control.peer.reset());
        control.run(folder,"reset-prefill",0,5,257,true);control.run(folder,"reset-decode",1,2,273,true);
    }
    // Distinct owner, borrowed streams and simultaneous executable pairs.
    {
        PhaseControl again(first,second,false);again.capture(0);again.capture(1);
        again.run(folder,"reacquire-decode",1,4,513,true);again.run(folder,"reacquire-prefill",0,4,514,true);
    }
    for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaDeviceEnablePeerAccess(1-r,0));}
    {
        PhaseControl external(first,second,true);external.capture(0);external.capture(1);
        external.run(folder,"external-prefill",0,2,1021,true);external.run(folder,"external-decode",1,2,1022,true);
    }
    for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaDeviceDisablePeerAccess(1-r));}
    CUDA_OK(cudaSetDevice(original));
    std::puts("PRIVATE_PUSH_GRID_OWNER_ALL_GATES_COMPLETE matrices=40 FULL_CAPACITY_TAILS CALLER_STREAMS_PRESERVED NO_SERVING_ACCEPT");return 0;
}
