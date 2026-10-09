// SPDX-License-Identifier: Apache-2.0
// Opt-in invocation-owned BF16 peer transport; typed native group required.
#pragma once
#include "tp_peer_math.cuh"
#include "gpt_oss_kernels.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>

namespace Garnet { namespace GptOssPeer {
namespace Bf16PeerPrivate {
constexpr int kMaximumBlocks=188;
struct alignas(128) Word { unsigned value; char padding[124]; };
struct Slot { Word start[2],end[2]; };
static_assert(sizeof(Word)==128 && alignof(Word)==128);
__device__ __forceinline__ void publish(unsigned* p,unsigned v){
    asm volatile("st.release.sys.global.u32 [%1], %0;"::"r"(v),"l"(p):"memory");
}
__device__ __forceinline__ unsigned observe(const unsigned* p){
    unsigned v;asm volatile("ld.acquire.sys.global.u32 %0, [%1];":"=r"(v):"l"(p):"memory");return v;
}
__device__ __forceinline__ uint4 readPacked(const uint4* p){
    uint4 v;asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];":
        "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");return v;
}
__global__ void reduce(const uint4* first,const uint4* second,float4* output,
    Slot* signals,Word* sequence,unsigned* faults,int rank,size_t vectors,
    unsigned long long maximumWaitCycles){
    const int block=blockIdx.x;
    __shared__ unsigned epoch,failed;
    if(threadIdx.x==0){
        epoch=sequence[block].value+1;failed=epoch==0?2:0;
        if(!failed){
            // The original pack kernel precedes this launch in the owner stream.
            __threadfence_system();publish(&signals[block].start[rank].value,epoch);
            const auto begin=clock64();
            while(observe(&signals[block].start[1-rank].value)!=epoch){
                if(clock64()-begin>maximumWaitCycles){failed=1;break;}__nanosleep(32);
            }
        }
    }
    __syncthreads();if(failed){if(threadIdx.x==0)faults[block]=failed;return;}
    for(size_t i=size_t(block)*blockDim.x+threadIdx.x;i<vectors;i+=size_t(gridDim.x)*blockDim.x){
        const uint4 a=readPacked(first+i),b=readPacked(second+i);
        output[2*i]=make_float4(roundedPair(a.x,b.x),roundedPair(a.x>>16,b.x>>16),
                              roundedPair(a.y,b.y),roundedPair(a.y>>16,b.y>>16));
        output[2*i+1]=make_float4(roundedPair(a.z,b.z),roundedPair(a.z>>16,b.z>>16),
                                roundedPair(a.w,b.w),roundedPair(a.w>>16,b.w>>16));
    }
    // End signals prevent the next stream's pack from overwriting peer reads.
    __syncthreads();
    if(threadIdx.x==0){
        publish(&signals[block].end[rank].value,epoch);const auto begin=clock64();
        while(observe(&signals[block].end[1-rank].value)!=epoch){
            if(clock64()-begin>maximumWaitCycles){failed=1;break;}__nanosleep(32);
        }
        sequence[block].value=epoch;if(failed)faults[block]=failed;
    }
}
}

class Bf16PeerNative {
    // Only the process lease is shared. All GPU/host arenas belong to this owner.
    inline static std::atomic_flag leaseFlag_=ATOMIC_FLAG_INIT;
    bool leased_=false;
    int original_=0,blocks_=0;size_t capacity_=0;bool ready_=false;
    std::array<bool,2> enabled_{};
    std::array<void*,2> packed_{};
    std::array<Bf16PeerPrivate::Word*,2> sequence_{};
    std::array<unsigned*,2> faults_{};
    Bf16PeerPrivate::Slot* hostSignals_=nullptr;
    std::array<Bf16PeerPrivate::Slot*,2> signals_{};
    std::array<unsigned long long,2> waitCycles_{};
    bool borrowed_=false;
    int activePhase_=0;
    std::array<std::array<cudaStream_t,2>,2> phaseStreams_{};
    cudaError_t quiesce(){
        // Phase changes are caller-serialized; never synchronize a capture.
        for(int phase=0;phase<(borrowed_?2:1);++phase)for(int r=0;r<2;++r){
            auto stream=borrowed_?phaseStreams_[phase][r]:streams[r];if(!stream)continue;
            auto e=cudaSetDevice(r);if(e!=cudaSuccess)return e;
            cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
            e=cudaStreamIsCapturing(stream,&status);if(e!=cudaSuccess)return e;
            if(status!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
        }
        for(int phase=0;phase<(borrowed_?2:1);++phase)for(int r=0;r<2;++r){
            auto stream=borrowed_?phaseStreams_[phase][r]:streams[r];if(!stream)continue;
            auto e=cudaSetDevice(r);if(e!=cudaSuccess)return e;
            e=cudaStreamSynchronize(stream);if(e!=cudaSuccess)return e;
        }
        return cudaSuccess;
    }
    cudaError_t fail(cudaError_t e){const auto cleanup=close();return e==cudaSuccess?cleanup:e;}
public:
    std::array<cudaStream_t,2> streams{};
    std::array<cudaDeviceProp,2> properties{};
    Bf16PeerNative()=default;
    Bf16PeerNative(const Bf16PeerNative&)=delete;
    Bf16PeerNative& operator=(const Bf16PeerNative&)=delete;
    bool ready()const{return ready_;}
    int blocks()const{return blocks_;}
    size_t ownedBytesPerRank()const{return capacity_*2+sizeof(Bf16PeerPrivate::Word)*blocks_+sizeof(unsigned)*blocks_;}
    size_t mappedBytes()const{return sizeof(Bf16PeerPrivate::Slot)*blocks_;}
    // preEnabled is an explicit caller contract: BOTH peer links already enabled.
    // The owner never disables caller-owned links. The private fixture proves it.
    using PhaseStreams=std::array<std::array<cudaStream_t,2>,2>;
    // Optional two BORROWED phase/rank stream pairs. Caller keeps them alive,
    // serializes phase transitions, and destroys executable graphs before close.
    cudaError_t initialize(size_t capacity,int blocks,int simulateFailureAfterRank=-1,bool preEnabled=false,
                           const PhaseStreams* borrowedStreams=nullptr){
        if(ready_ || leased_)return cudaErrorNotReady;
        if(!capacity || capacity%8 || capacity>kMaximumPairElements ||
            (blocks!=32 && blocks!=64 && blocks!=128 && blocks!=188) || simulateFailureAfterRank<-1 || simulateFailureAfterRank>1)
            return cudaErrorInvalidValue;
        if(borrowedStreams)for(int r=0;r<2;++r)
            if(!(*borrowedStreams)[0][r] || !(*borrowedStreams)[1][r] ||
                (*borrowedStreams)[0][r]==(*borrowedStreams)[1][r])return cudaErrorInvalidValue;
#if CUDART_VERSION < 13000
        // This private mode requires the explicit stream-device runtime query.
        // Older-toolkit builds retain the original owned-stream path only.
        if(borrowedStreams)return cudaErrorNotSupported;
#endif
        if(leaseFlag_.test_and_set(std::memory_order_acquire))return cudaErrorNotReady;leased_=true;
        auto e=cudaGetDevice(&original_);if(e!=cudaSuccess){leased_=false;leaseFlag_.clear(std::memory_order_release);return e;}
        capacity_=capacity;blocks_=blocks;int devices=0;
        e=cudaGetDeviceCount(&devices);if(e!=cudaSuccess)return fail(e);
        if(devices!=2)return fail(cudaErrorNotSupported);
        if(borrowedStreams){
            for(int phase=0;phase<2;++phase)for(int r=0;r<2;++r){
                e=cudaSetDevice(r);if(e!=cudaSuccess)return fail(e);
#if CUDART_VERSION >= 13000
                int streamDevice=-1;e=cudaStreamGetDevice((*borrowedStreams)[phase][r],&streamDevice);
                if(e!=cudaSuccess)return fail(e);
                if(streamDevice!=r)return fail(cudaErrorInvalidDevice);
#endif
                unsigned flags=0;e=cudaStreamGetFlags((*borrowedStreams)[phase][r],&flags);
                if(e!=cudaSuccess)return fail(e);
                if(!(flags&cudaStreamNonBlocking))return fail(cudaErrorNotSupported);
                cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
                e=cudaStreamIsCapturing((*borrowedStreams)[phase][r],&status);if(e!=cudaSuccess)return fail(e);
                if(status!=cudaStreamCaptureStatusNone)return fail(cudaErrorStreamCaptureUnsupported);
            }
            borrowed_=true;phaseStreams_=*borrowedStreams;streams=phaseStreams_[0];activePhase_=0;
        }
        for(int r=0;r<2;++r){
            e=cudaGetDeviceProperties(&properties[r],r);if(e!=cudaSuccess)return fail(e);
            const auto& p=properties[r];
            if(!p.canMapHostMemory || !p.unifiedAddressing || !p.cooperativeLaunch || p.major<8 || blocks>p.multiProcessorCount)
                return fail(cudaErrorNotSupported);
            int peer=0;e=cudaDeviceCanAccessPeer(&peer,r,1-r);if(e!=cudaSuccess)return fail(e);
            if(!peer)return fail(cudaErrorNotSupported);
            int atomics=0;e=cudaDeviceGetP2PAttribute(&atomics,cudaDevP2PAttrNativeAtomicSupported,r,1-r);
            if(e!=cudaSuccess)return fail(e);
            std::printf("PEER_CAPABILITY rank=%d SM=%d%d multiprocessors=%d mapped=%d UVA=%d cooperative=%d peer=%d native_atomic=%d MAPPED_U32_SIGNALS\n",
                r,p.major,p.minor,p.multiProcessorCount,p.canMapHostMemory,p.unifiedAddressing,p.cooperativeLaunch,peer,atomics);
            int clockKHz=0;e=cudaDeviceGetAttribute(&clockKHz,cudaDevAttrClockRate,r);
            if(e!=cudaSuccess)return fail(e);
            waitCycles_[r]=static_cast<unsigned long long>(clockKHz)*1000*30;
            if(!waitCycles_[r])return fail(cudaErrorNotSupported);
        }
        e=cudaHostAlloc(&hostSignals_,mappedBytes(),cudaHostAllocPortable|cudaHostAllocMapped);
        if(e!=cudaSuccess)return fail(e);std::memset(hostSignals_,0,mappedBytes());
        if(reinterpret_cast<uintptr_t>(hostSignals_)%128)return fail(cudaErrorNotSupported);
        for(int r=0;r<2;++r){
            e=cudaSetDevice(r);if(e!=cudaSuccess)return fail(e);
            int resident=0;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,Bf16PeerPrivate::reduce,256,0);
            if(e!=cudaSuccess)return fail(e);if(resident*properties[r].multiProcessorCount<blocks)return fail(cudaErrorNotSupported);
            if(!preEnabled){
                e=cudaDeviceEnablePeerAccess(1-r,0);
                if(e!=cudaSuccess)return fail(e);enabled_[r]=true;
            }
            e=cudaHostGetDevicePointer(&signals_[r],hostSignals_,0);if(e!=cudaSuccess)return fail(e);
            if(!borrowed_){e=cudaStreamCreateWithFlags(&streams[r],cudaStreamNonBlocking);if(e!=cudaSuccess)return fail(e);}
            e=cudaMalloc(&packed_[r],capacity_*2);if(e!=cudaSuccess)return fail(e);
            e=cudaMalloc(&sequence_[r],sizeof(Bf16PeerPrivate::Word)*blocks);if(e!=cudaSuccess)return fail(e);
            e=cudaMalloc(&faults_[r],sizeof(unsigned)*blocks);if(e!=cudaSuccess)return fail(e);
            e=cudaMemset(sequence_[r],0,sizeof(Bf16PeerPrivate::Word)*blocks);if(e!=cudaSuccess)return fail(e);
            e=cudaMemset(faults_[r],0,sizeof(unsigned)*blocks);if(e!=cudaSuccess)return fail(e);
            if(simulateFailureAfterRank==r)return fail(cudaErrorUnknown);
        }
        e=cudaSetDevice(original_);if(e!=cudaSuccess)return fail(e);ready_=true;return cudaSuccess;
    }
    cudaError_t bindBorrowedPhase(int phase){
        if(!ready_)return cudaErrorNotReady;
        if(!borrowed_ || phase<0 || phase>1)return cudaErrorInvalidValue;
        auto e=quiesce();if(e!=cudaSuccess)return e;
        streams=phaseStreams_[phase];activePhase_=phase;
        return cudaSetDevice(original_);
    }
    int activePhase()const{return activePhase_;}
    cudaError_t enqueue(const float* input,float* output,size_t count,int rank,cudaStream_t stream){
        if(!ready_)return cudaErrorNotReady;
        if(!input || !output || !count || count%8 || count>capacity_ || rank<0 || rank>1 ||
            reinterpret_cast<uintptr_t>(input)%16 || reinterpret_cast<uintptr_t>(output)%16 || stream!=streams[rank])
            return cudaErrorInvalidValue;
        int device=0;auto e=cudaGetDevice(&device);if(e!=cudaSuccess)return e;
        if(device!=rank)return cudaErrorInvalidDevice;
        for(const void* pointer:{static_cast<const void*>(input),static_cast<const void*>(output)}){
            cudaPointerAttributes p{};e=cudaPointerGetAttributes(&p,pointer);if(e!=cudaSuccess)return e;
            if(p.type!=cudaMemoryTypeDevice || p.device!=rank)return cudaErrorInvalidDevicePointer;
        }
        e=Garnet::GptOssTpPackBf16(input,packed_[rank],count,stream);if(e!=cudaSuccess)return e;
        const uint4* first=static_cast<const uint4*>(packed_[0]);const uint4* second=static_cast<const uint4*>(packed_[1]);
        auto* y=reinterpret_cast<float4*>(output);auto* s=signals_[rank];auto* q=sequence_[rank];auto* f=faults_[rank];
        size_t vectors=count/8;auto limit=waitCycles_[rank];
        void* args[]={&first,&second,&y,&s,&q,&f,&rank,&vectors,&limit};
        return cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(Bf16PeerPrivate::reduce),dim3(blocks_),dim3(256),args,0,stream);
    }
    cudaError_t checkFaults(int rank,bool* clean){
        if(!ready_)return cudaErrorNotReady;if(!clean || rank<0 || rank>1)return cudaErrorInvalidValue;
        std::array<unsigned,Bf16PeerPrivate::kMaximumBlocks> values{};
        auto e=cudaSetDevice(rank);if(e!=cudaSuccess)return e;
        e=cudaMemcpy(values.data(),faults_[rank],blocks_*sizeof(unsigned),cudaMemcpyDeviceToHost);if(e!=cudaSuccess)return e;
        *clean=true;for(int b=0;b<blocks_;++b)if(values[b]){*clean=false;std::fprintf(stderr,"PEER_FAULT rank=%d block=%d code=%u\n",rank,b,values[b]);}
        return cudaSuccess;
    }
    // Caller finishes BOTH streams and destroys executable graphs before close.
    cudaError_t reset(){
        if(!ready_)return cudaErrorNotReady;
        auto sync=quiesce();if(sync!=cudaSuccess)return sync;
        std::memset(hostSignals_,0,mappedBytes());
        for(int r=0;r<2;++r){auto e=cudaSetDevice(r);if(e!=cudaSuccess)return e;
            e=cudaMemset(sequence_[r],0,blocks_*sizeof(Bf16PeerPrivate::Word));if(e!=cudaSuccess)return e;
            e=cudaMemset(faults_[r],0,blocks_*sizeof(unsigned));if(e!=cudaSuccess)return e;}
        return cudaSetDevice(original_);
    }
    cudaError_t close(){
        if(!leased_)return cudaSuccess;
        cudaError_t first=cudaSuccess;
        auto keep=[&](cudaError_t e){if(first==cudaSuccess && e!=cudaSuccess)first=e;};
        const auto sync=quiesce();
        // Fail closed with resources/lease intact during an active capture.
        if(sync!=cudaSuccess)return sync;
        for(int r=0;r<2;++r)if(packed_[r] || sequence_[r] || faults_[r] || streams[r]){keep(cudaSetDevice(r));
            if(packed_[r])keep(cudaFree(packed_[r]));if(sequence_[r])keep(cudaFree(sequence_[r]));if(faults_[r])keep(cudaFree(faults_[r]));
            if(streams[r] && !borrowed_)keep(cudaStreamDestroy(streams[r]));packed_[r]=nullptr;sequence_[r]=nullptr;faults_[r]=nullptr;streams[r]=nullptr;signals_[r]=nullptr;
        }
        if(hostSignals_)keep(cudaFreeHost(hostSignals_));hostSignals_=nullptr;
        for(int r=0;r<2;++r)if(enabled_[r]){keep(cudaSetDevice(r));keep(cudaDeviceDisablePeerAccess(1-r));enabled_[r]=false;}
        keep(cudaSetDevice(original_));ready_=false;capacity_=0;blocks_=0;borrowed_=false;phaseStreams_={};activePhase_=0;
        leased_=false;leaseFlag_.clear(std::memory_order_release);return first;
    }
    ~Bf16PeerNative(){const auto e=close();if(e!=cudaSuccess){std::fprintf(stderr,"PEER_OWNER_CLEANUP_FAILED: %s\n",cudaGetErrorString(e));std::terminate();}}
};
} }
