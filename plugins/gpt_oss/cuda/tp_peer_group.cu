// SPDX-License-Identifier: Apache-2.0
#include "gpt_oss_peer_group.h"
#include "tp_peer_owner.cuh"
#include "gpt_oss_peer_group_options.h"
#include <atomic>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <cstdlib>
namespace Garnet {
namespace {
struct Group {
    std::atomic<unsigned> references{1};
    std::mutex mutex;
    unsigned activeRanks=0;
    int phase=0,original=-1;
    bool tp=false;
    GptOssPeer::Bf16PeerNative owner;
    GptOssPeer::Bf16PeerNative::PhaseStreams streams{};
    std::array<size_t,2> elements{};
    std::string description;
    ~Group(){
        if(activeRanks)std::terminate();
        if(owner.close()!=cudaSuccess)std::terminate();
        for(int r=0;r<2;++r){
            if(!streams[0][r] && !streams[1][r])continue;
            if(cudaSetDevice(r)!=cudaSuccess)std::terminate();
            for(int p=0;p<2;++p)if(streams[p][r] && cudaStreamDestroy(streams[p][r])!=cudaSuccess)std::terminate();
        }
        if(tp)GptOssTpRelease();
        if(original>=0 && cudaSetDevice(original)!=cudaSuccess)std::terminate();
    }
};
thread_local Group* bound=nullptr;
thread_local int boundRank=-1,boundPhase=-1;
void Retain(void* p){static_cast<Group*>(p)->references.fetch_add(1,std::memory_order_relaxed);}
void Release(void* p){auto* g=static_cast<Group*>(p);if(g->references.fetch_sub(1,std::memory_order_acq_rel)==1)delete g;}
int Bind(void* p,uint32_t phase){
    auto* g=static_cast<Group*>(p);if(phase>1)return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(g->mutex);
    if(g->activeRanks)return cudaErrorNotReady;
    if(g->phase==int(phase))return cudaSuccess;
    auto e=g->owner.bindBorrowedPhase(int(phase));if(e==cudaSuccess)g->phase=int(phase);return e;
}
int Enter(void* p,uint32_t phase,uint32_t rank,void** stream){
    auto* g=static_cast<Group*>(p);if(!stream || rank>1 || phase>1 || bound)return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(g->mutex);
    int device=-1;auto e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;
    if(device!=int(rank))return cudaErrorInvalidDevice;
    if(g->phase!=int(phase) || (g->activeRanks&(1U<<rank)))return cudaErrorNotReady;
    g->activeRanks|=1U<<rank;bound=g;boundRank=int(rank);boundPhase=int(phase);
    *stream=g->streams[phase][rank];return cudaSuccess;
}
int Leave(void* p,uint32_t rank){
    auto* g=static_cast<Group*>(p);if(rank>1 || bound!=g || boundRank!=int(rank))return cudaErrorInvalidValue;
    auto e=cudaStreamSynchronize(g->streams[boundPhase][rank]);bool clean=false;
    if(e==cudaSuccess)e=g->owner.checkFaults(int(rank),&clean);
    if(e==cudaSuccess && !clean)e=cudaErrorUnknown;
    {std::lock_guard<std::mutex> lock(g->mutex);g->activeRanks&=~(1U<<rank);}
    bound=nullptr;boundRank=-1;boundPhase=-1;return e;
}
const char* Status(void* p){return static_cast<Group*>(p)->description.c_str();}
const GarnetOperatorExecutionServices services{1,sizeof(GarnetOperatorExecutionServices),"gpt_oss","tensorrt",2,2,
    &Retain,&Release,&Bind,&Enter,&Leave,&Status};
}
const GarnetOperatorExecutionServices* GptOssPeerExecutionServices(){return &services;}
cudaError_t GptOssCreatePeerGroup(const char* options,size_t bytes,void** result){
    if(!options || !bytes || !result)return cudaErrorInvalidValue;*result=nullptr;
    const char* flag=std::getenv("GARNET_GPT_OSS_BF16_PEER_GROUP");
    if(!flag || std::strcmp(flag,"1"))return cudaErrorNotSupported;
    const char* gridFlag=std::getenv("GARNET_GPT_OSS_BF16_PEER_GRID_SIGNALS");
    if(gridFlag && std::strcmp(gridFlag,"0") && std::strcmp(gridFlag,"1"))return cudaErrorInvalidValue;
    const bool gridSignals=gridFlag && !std::strcmp(gridFlag,"1");
    std::array<size_t,2> counts{};size_t capacity=0;int ctas=0;
    if(!ParseGptOssPeerGroupOptions(options,bytes,GptOssPeer::kMaximumPairElements,counts,capacity,ctas))
        return cudaErrorInvalidValue;
    auto g=std::make_unique<Group>();auto e=cudaGetDevice(&g->original);if(e!=cudaSuccess)return e;
    // Retain the verified legacy peer-link owner, never probe/mask AlreadyEnabled.
    e=GptOssTpAcquire();if(e!=cudaSuccess)return e;g->tp=true;
    if(!GptOssTpHasDirectPeerOwner())return cudaErrorNotSupported;
    for(int r=0;r<2;++r){e=cudaSetDevice(r);if(e!=cudaSuccess)return e;
        for(int phase=0;phase<2;++phase){e=cudaStreamCreateWithFlags(&g->streams[phase][r],cudaStreamNonBlocking);if(e!=cudaSuccess)return e;}}
    e=g->owner.initialize(capacity,ctas,-1,true,&g->streams,gridSignals);if(e!=cudaSuccess)return e;
    g->elements=counts;
    g->description=GptOssPeerGroupDescription(counts,capacity,ctas,g->owner.ownedBytesPerRank(),g->owner.mappedBytes(),gridSignals);
    e=cudaSetDevice(g->original);if(e!=cudaSuccess)return e;*result=g.release();return cudaSuccess;
}
cudaError_t GptOssPeerGroupAllReduce(const float* x,float* y,size_t n,int rank,int prefill,cudaStream_t stream){
    if(!bound || rank!=boundRank || (prefill?0:1)!=boundPhase || n!=bound->elements[boundPhase])return cudaErrorInvalidValue;
    return bound->owner.enqueue(x,y,n,rank,stream);
}
}
