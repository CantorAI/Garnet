// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include "exact_half_codec.h"
#ifdef GARNET_METAL_SERIAL_VISION_PREFETCH
#include "serial_vision_prefetch_shader.h"
#endif
#ifdef GARNET_METAL_SCREENED_HEAD
#include "screened_head_index.h"
#include "screened_head_shader.h"
#endif
#ifdef GARNET_METAL_PARALLEL_VISION_CONTEXT
#include "parallel_vision_context_shader.h"
#endif
#ifdef GARNET_METAL_BODY_SHAPE
#include "body_shape_shader.h"
#include <shared_mutex>
#endif
#ifdef GARNET_METAL_FUSED_HEAD_ARGMAX
#include "fused_head_argmax_shader.h"
#endif
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <array>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
namespace Garnet {
struct MetalMatrix::Impl {
    MPSMatrix* matrix;
    bool ready=false;
    bool pending=false;
};
bool MetalMatrix::SharesStorage(const MetalMatrix& other) const {
    return impl_&&other.impl_&&impl_->matrix.data==other.impl_->matrix.data;
}
bool MetalMatrix::CompatibleLayout(const MetalMatrix& other) const {
    return impl_&&other.impl_&&impl_->matrix.device==other.impl_->matrix.device&&
        impl_->matrix.rows==other.impl_->matrix.rows&&impl_->matrix.columns==other.impl_->matrix.columns&&
        impl_->matrix.rowBytes==other.impl_->matrix.rowBytes;
}
struct MetalBF16Weight::Impl {
#ifdef GARNET_METAL_SCREENED_HEAD
    std::mutex screenMutex;
    bool screenAttempted=false;
    id<MTLBuffer> screenIndex,screenMetadata;
#endif
    id<MTLBuffer> buffer;
    MPSMatrix* exactHalf;
    id<MTLBuffer> halfScales;
    size_t halfScalesOffset=0;
    bool globalHalf=false;
    bool visionPrefill=false;
    int32_t globalHalfScale=0;
    size_t rows,columns;
    bool floatPrefill=false;
};
bool MetalBF16Weight::ExactHalfReady() const {return impl_&&impl_->exactHalf&&impl_->halfScales;}
bool MetalBF16Weight::GlobalHalfReady() const {return ExactHalfReady()&&impl_->globalHalf;}
struct MetalPipelines {
    id<MTLLibrary> library;
#ifdef GARNET_METAL_SERIAL_VISION_PREFETCH
    id<MTLLibrary> visionPrefetchLibrary;
    id<MTLComputePipelineState> visionPrefetch;
#endif
#ifdef GARNET_METAL_SCREENED_HEAD
    id<MTLLibrary> screenLibrary;
    id<MTLComputePipelineState> screenPrepare,screenProjection;
#endif
#ifdef GARNET_METAL_BODY_SHAPE
    std::shared_mutex bodyShapeMutex;
    std::map<std::array<size_t,5>,id<MTLComputePipelineState>> bodyShape;
#endif
    id<MTLComputePipelineState> argmax;
#ifdef GARNET_METAL_FUSED_HEAD_ARGMAX
    id<MTLComputePipelineState> fusedHead,fusedHeadReduce;
#endif
    id<MTLComputePipelineState> bf16Decode,bf16DecodeSmall,bf16Gemm,bf16Gemm16Wide,bf16Gemm32Wide;
    id<MTLComputePipelineState> bf16Split,bf16SplitReduce;
#ifdef GARNET_METAL_NATIVE_BFLOAT_BODY
    id<MTLComputePipelineState> nativeBFSerial,nativeBFDot;
#endif
    id<MTLComputePipelineState> halfRescale;
#ifdef GARNET_METAL_SINGLE_CODER_HEAD
    id<MTLComputePipelineState> singleCoderHead;
#endif
#ifdef GARNET_METAL_CONSTANT_BF16_HEAD
    id<MTLComputePipelineState> constantHead;
#endif
#ifdef GARNET_METAL_JOINED_PREFILL
    id<MTLComputePipelineState> joinedSplitRescale;
#endif
    id<MTLComputePipelineState> rms;
    id<MTLComputePipelineState> residualNorm;
    id<MTLComputePipelineState> packedPrefillNorm;
    id<MTLComputePipelineState> silu;
#ifdef GARNET_METAL_JOINED_GLOBAL_MLP
    id<MTLComputePipelineState> joinedGlobalSilu;
#endif
    id<MTLComputePipelineState> kvInplace;
    id<MTLComputePipelineState> add;
    id<MTLComputePipelineState> mask;
    id<MTLComputePipelineState> conv,convChunk;
    id<MTLComputePipelineState> delta,deltaSplit;
    id<MTLComputePipelineState> deltaSplitParallel,deltaChunk;
    id<MTLComputePipelineState> slice,sliceVector;
    id<MTLComputePipelineState> prepare;
    id<MTLComputePipelineState> gatedNorm;
    id<MTLComputePipelineState> concat;
    id<MTLComputePipelineState> normRope,kvUpdate,kvUpdateVector,scores,scoresSerial,attentionContext;
    id<MTLComputePipelineState> vision,visionNorm,visionScores,layout,kvChunk,kvChunkVector,attentionChunk;
    id<MTLComputePipelineState> visionPackQK,visionSoftmax,visionFusedContext;
    id<MTLComputePipelineState> visionParallelContext;
};
static std::mutex pipelineMutex;
static std::map<uint64_t,std::weak_ptr<MetalPipelines>> pipelineCache;
struct MetalContext::Impl {
    size_t screenedHeadDispatches=0;
    size_t screenedHeadSeedRows=0;
    size_t interleavedScreenDispatches=0;
    size_t visionPrefetchDispatches=0;
#ifdef GARNET_METAL_SCREENED_HEAD
    struct ScreenScratch {MetalMatrix seed,scores;id<MTLBuffer> stats,visits;};
    std::map<std::array<size_t,3>,ScreenScratch> screenScratch;
    std::array<size_t,3> screenDiagnosticKey{};
    bool screenDiagnosticReady=false;
#endif
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> command;
    NSMutableArray<id<MTLCommandBuffer>>* submitted=nil;
    size_t partialSubmissions=0;
    size_t deferredZeroBytes=0,deferredZeroStates=0;
#ifdef GARNET_METAL_DEFERRED_STATE_ZERO
    std::unordered_map<const MetalMatrix::Impl*,std::shared_ptr<MetalMatrix::Impl>> zeroQueued;
#endif
    size_t fusedHeadArgmaxDispatches=0;
    size_t bodyShapeDispatches=0,bodyShapeMisses=0;
#ifdef GARNET_METAL_FUSED_HEAD_ARGMAX
    std::map<std::array<size_t,2>,MetalMatrix> fusedHeadScratch;
#endif
    double gpuMs=0;
    std::vector<std::shared_ptr<MetalMatrix::Impl>> outputs;
    std::unordered_set<const MetalMatrix::Impl*> pendingOutputs;
    std::shared_ptr<MetalPipelines> kernels;
    std::map<std::array<size_t,4>,MPSMatrixMultiplication*> matmul;
    std::map<std::array<size_t,3>,MPSMatrixMultiplication*> halfMatmul;
    std::map<std::array<size_t,4>,MPSMatrixMultiplication*> globalHalfMatmul;
    std::map<std::array<size_t,2>,MetalMatrix> halfScratch;
    std::map<std::array<size_t,3>,MetalMatrix> splitScratch;
    size_t joinedPrefillPairs=0;
    size_t constantHeadDispatches=0;
    size_t singleCoderHeadDispatches=0;
    size_t residualNormDispatches=0;
    size_t packedPrefillNormDispatches=0;
    size_t globalHalfAlphaDispatches=0;
    size_t wideBatchPrefillDispatches=0;
    size_t visionGlobalAlphaDispatches=0;
    size_t visionParallelDispatches=0;
    size_t joinedGlobalMlpBlocks=0;
#ifdef GARNET_METAL_JOINED_GLOBAL_MLP
    struct JoinedGlobalWeight {
        MetalBF16Weight firstOwner,secondOwner;
        MPSMatrix* view;
    };
    std::map<std::pair<const void*,const void*>,JoinedGlobalWeight> joinedGlobalWeights;
#endif
#ifdef GARNET_METAL_JOINED_PREFILL
    struct JoinedWeight {
        MetalBF16Weight firstOwner,secondOwner;
        id<MTLBuffer> data,scales;
        MPSMatrix* view;
    };
    std::map<std::pair<const void*,const void*>,JoinedWeight> joinedWeights;
#endif
#ifdef GARNET_METAL_MPS_VISION_SCORES
    struct VisionScoreScratch {
        MetalMatrix query,key,scores;
        MPSMatrix* queryView;
        MPSMatrix* keyView;
        MPSMatrix* scoreView;
        MPSMatrixMultiplication* product;
    };
    std::map<std::array<size_t,4>,VisionScoreScratch> visionScores;
#endif
};
static size_t Elements(size_t rows, size_t columns) {
    if (!rows || !columns || rows > std::numeric_limits<size_t>::max()/columns ||
        rows*columns > std::numeric_limits<size_t>::max()/sizeof(float))
        throw std::invalid_argument("Invalid Metal matrix dimensions");
    return rows*columns;
}
MetalContext::MetalContext(): impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        impl_->device=MTLCreateSystemDefaultDevice();
        if (!impl_->device || !MPSSupportsMTLDevice(impl_->device))
            throw std::runtime_error("No MPS-capable Metal GPU available");
        impl_->queue=[impl_->device newCommandQueue];
        if (!impl_->queue) throw std::runtime_error("Cannot create Metal command queue");
        // Device kernels are immutable. Contexts retain a shared pipeline set;
        // the process cache is weak, so closing the last context releases it.
        std::lock_guard<std::mutex> kernelLock(pipelineMutex);
        auto& cached=pipelineCache[impl_->device.registryID];
        impl_->kernels=cached.lock();
        if(impl_->kernels)return;
        impl_->kernels=std::make_shared<MetalPipelines>();
        NSString* source=[NSString stringWithUTF8String:R"METAL(
#include <metal_stdlib>
using namespace metal;
struct BF16Params {ulong rows,inner,columns,aStride,oStride;};
struct HalfRescaleParams {ulong rows,columns,iStride,oStride;};
kernel void half_rescale(device const float* input [[buffer(0)]],device const int* scales [[buffer(1)]],
    device float* output [[buffer(2)]],constant HalfRescaleParams& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,column=id%p.columns;
    output[row*p.oStride+column]=ldexp(input[row*p.iStride+column],-scales[column]);
}
kernel void bf16_decode(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            ushort4 b0=*(device const ushort4*)(weight+column*p.inner+k),b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
            if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
            if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
            if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
                sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<p.inner;k+=32) {
        float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
        float4 w=0.0f;
        w.x=as_type<float>(uint(weight[column*p.inner+k])<<16);
        if(column+1<p.columns)w.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
        if(column+2<p.columns)w.z=as_type<float>(uint(weight[(column+2)*p.inner+k])<<16);
        if(column+3<p.columns)w.w=as_type<float>(uint(weight[(column+3)*p.inner+k])<<16);
        sum0+=x*w;if(p.rows==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
kernel void bf16_decode_small(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*4+simd*2;
    if(column>=p.columns)return;
    float2 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            ushort4 b0=*(device const ushort4*)(weight+column*p.inner+k),b1=ushort4(0);
            if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
            for(uint j=0;j<4;++j) {
                float2 w=as_type<float2>(uint2(b0[j],b1[j])<<16);
                sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<p.inner;k+=32) {
        float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
        float2 w=0.0f;
        w.x=as_type<float>(uint(weight[column*p.inner+k])<<16);
        if(column+1<p.columns)w.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
        sum0+=x*w;if(p.rows==2)sum1+=y*w;
    }
    }
    sum0=float2(simd_sum(sum0.x),simd_sum(sum0.y));
    sum1=float2(simd_sum(sum1.x),simd_sum(sum1.y));
    if(lane==0)for(uint j=0;j<2&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
// BF16 weights are expanded only inside a small shared-memory tile. FP32
// SIMD-group matrix multiply accumulates two 8x8 output tiles per group.
kernel void bf16_gemm(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float left[16*32],right[32*32],result[16*32];
    ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*32;
    simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
    simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(ulong base=0;base<p.inner;base+=32) {
        for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,k=base+i%32;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
        for(uint i=tid;i<32*32;i+=128){ulong k=base+i/32,column=colBase+i%32;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint offset=0;offset<32;offset+=8) {
            simdgroup_float8x8 l0,l1,r;
            simdgroup_load(l0,left+offset,32);simdgroup_load(l1,left+8*32+offset,32);simdgroup_load(r,right+offset*32+simd*8,32);
            simdgroup_multiply_accumulate(accum0,l0,r,accum0);simdgroup_multiply_accumulate(accum1,l1,r,accum1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(accum0,result+simd*8,32);simdgroup_store(accum1,result+8*32+simd*8,32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,column=colBase+i%32;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=result[i];}
}
struct SplitParams {ulong rows,inner,columns,aStride,oStride,parts;};
kernel void bf16_split_key(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant SplitParams& p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float left[16*32],right[32*32],result[16*32];
    ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*32,begin=ulong(group.z)*64,end=min(begin+64,p.inner);
    simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
    simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(ulong base=begin;base<end;base+=32) {
        for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,k=base+i%32;left[i]=(row<p.rows&&k<end)?a[row*p.aStride+k]:0.0f;}
        for(uint i=tid;i<32*32;i+=128){ulong k=base+i/32,column=colBase+i%32;right[i]=(k<end&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint offset=0;offset<32;offset+=8) {
            simdgroup_float8x8 l0,l1,r;
            simdgroup_load(l0,left+offset,32);simdgroup_load(l1,left+8*32+offset,32);simdgroup_load(r,right+offset*32+simd*8,32);
            simdgroup_multiply_accumulate(accum0,l0,r,accum0);simdgroup_multiply_accumulate(accum1,l1,r,accum1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(accum0,result+simd*8,32);simdgroup_store(accum1,result+8*32+simd*8,32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,column=colBase+i%32;if(row<p.rows&&column<p.columns)out[(ulong(group.z)*p.rows+row)*p.oStride+column]=result[i];}
}
struct SplitReduceParams {ulong rows,columns,iStride,oStride,parts;};
kernel void bf16_split_reduce(device const float* partial [[buffer(0)]],device float* out [[buffer(1)]],
    constant SplitReduceParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=ulong(id)/p.columns,col=ulong(id)%p.columns;
    float terms[16];for(uint i=0;i<16;++i)terms[i]=i<p.parts?partial[(ulong(i)*p.rows+row)*p.iStride+col]:0.0f;
    for(uint width=8;width>0;width/=2)for(uint i=0;i<width;++i)terms[i]=terms[2*i]+terms[2*i+1];
    out[row*p.oStride+col]=terms[0];
}
kernel void gemm_16_64_64(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float left[1024],right[4096],result[1024];
ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*64;
simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=64) {
for(uint i=tid;i<1024;i+=256){ulong row=rowBase+i/64,k=base+i%64;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
for(uint i=tid;i<4096;i+=256){ulong k=base+i/64,column=colBase+i%64;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint offset=0;offset<64;offset+=8) {
simdgroup_float8x8 r;
simdgroup_load(r,right+offset*64+simd*8,64);
simdgroup_float8x8 l0;
simdgroup_load(l0,left+0+offset,64);
simdgroup_multiply_accumulate(accum0,l0,r,accum0);
simdgroup_float8x8 l1;
simdgroup_load(l1,left+512+offset,64);
simdgroup_multiply_accumulate(accum1,l1,r,accum1);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(accum0,result+0+simd*8,64);
simdgroup_store(accum1,result+512+simd*8,64);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<1024;i+=256){ulong row=rowBase+i/64,column=colBase+i%64;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=result[i];}
}
kernel void gemm_32_64_32(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float left[1024],right[2048],result[2048];
ulong rowBase=ulong(group.y)*32,colBase=ulong(group.x)*64;
simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum2=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum3=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=32) {
for(uint i=tid;i<1024;i+=256){ulong row=rowBase+i/32,k=base+i%32;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
for(uint i=tid;i<2048;i+=256){ulong k=base+i/64,column=colBase+i%64;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint offset=0;offset<32;offset+=8) {
simdgroup_float8x8 r;
simdgroup_load(r,right+offset*64+simd*8,64);
simdgroup_float8x8 l0;
simdgroup_load(l0,left+0+offset,32);
simdgroup_multiply_accumulate(accum0,l0,r,accum0);
simdgroup_float8x8 l1;
simdgroup_load(l1,left+256+offset,32);
simdgroup_multiply_accumulate(accum1,l1,r,accum1);
simdgroup_float8x8 l2;
simdgroup_load(l2,left+512+offset,32);
simdgroup_multiply_accumulate(accum2,l2,r,accum2);
simdgroup_float8x8 l3;
simdgroup_load(l3,left+768+offset,32);
simdgroup_multiply_accumulate(accum3,l3,r,accum3);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(accum0,result+0+simd*8,64);
simdgroup_store(accum1,result+512+simd*8,64);
simdgroup_store(accum2,result+1024+simd*8,64);
simdgroup_store(accum3,result+1536+simd*8,64);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<2048;i+=256){ulong row=rowBase+i/64,column=colBase+i%64;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=result[i];}
}
struct ArgmaxParams {ulong rows,columns,iStride,oStride;};
kernel void argmax_rows(device const float* input [[buffer(0)]],device float* output [[buffer(1)]],
    constant ArgmaxParams& p [[buffer(2)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    threadgroup float maximum[4];threadgroup uint indices[4];threadgroup uint invalid[4];
    float best=-INFINITY;uint index=0xffffffff,bad=0;
    for(ulong j=tid;j<p.columns;j+=128) {
        float value=input[row*p.iStride+j];bad|=!isfinite(value);
        if(value>best||(value==best&&j<index)){best=value;index=uint(j);}
    }
    float groupBest=simd_max(best);uint groupIndex=simd_min(best==groupBest?index:0xffffffff);
    uint groupBad=simd_max(bad);
    if(lane==0){maximum[simd]=groupBest;indices[simd]=groupIndex;invalid[simd]=groupBad;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    best=lane<4?maximum[lane]:-INFINITY;index=lane<4?indices[lane]:0xffffffff;bad=lane<4?invalid[lane]:0;
    groupBest=simd_max(best);groupIndex=simd_min(best==groupBest?index:0xffffffff);groupBad=simd_max(bad);
    if(tid==0){output[row*p.oStride]=float(groupIndex);output[row*p.oStride+1]=float(groupBad!=0);}
}
struct Params { ulong rows,columns,aStride,bStride,cStride;float epsilon,offset; };
kernel void rms_norm(device const float* a [[buffer(0)]],device const float* weight [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    threadgroup float partial[4];
    float sum=0;
    for(ulong j=tid;j<p.columns;j+=128){float x=a[row*p.aStride+j];sum+=x*x;}
    sum=simd_sum(sum);
    if(lane==0)partial[simd]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(lane<4?partial[lane]:0.0f);
    float inverse=rsqrt(sum/float(p.columns)+p.epsilon);
    for(ulong j=tid;j<p.columns;j+=128)c[row*p.cStride+j]=a[row*p.aStride+j]*inverse*(weight[j]+p.offset);
}
kernel void residual_norm(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device const float* weight [[buffer(2)]],device float* residual [[buffer(3)]],
    device float* c [[buffer(4)]],constant Params& p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    threadgroup float partial[4];
    float sum=0;
    for(ulong j=tid;j<p.columns;j+=128){float x=a[row*p.aStride+j]+b[row*p.cStride+j];residual[row*p.cStride+j]=x;sum+=x*x;}
    sum=simd_sum(sum);
    if(lane==0)partial[simd]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    sum=simd_sum(lane<4?partial[lane]:0.0f);
    float inverse=rsqrt(sum/float(p.columns)+p.epsilon);
    for(ulong j=tid;j<p.columns;j+=128)c[row*p.cStride+j]=residual[row*p.cStride+j]*inverse*(weight[j]+p.offset);
}
struct PackedParams {ulong rows,columns,packedStride,normStride,chunk;float epsilon,offset;};
kernel void packed_residual_norm(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
 device const float* weight [[buffer(2)]],device float* packed [[buffer(3)]],device float* layout [[buffer(4)]],
 device float* out [[buffer(5)]],constant PackedParams& p [[buffer(6)]],uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 if(row>=p.rows)return;
 ulong base=(row/p.chunk)*p.packedStride+(row%p.chunk)*p.columns;
 threadgroup float partial[4];float sum=0;
 for(ulong j=tid;j<p.columns;j+=128){float x=a[base+j]+b[base+j];packed[base+j]=x;layout[row*p.normStride+j]=x;sum+=x*x;}
 sum=simd_sum(sum);if(lane==0)partial[simd]=sum;
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
 sum=simd_sum(lane<4?partial[lane]:0.0f);
 float inverse=rsqrt(sum/float(p.columns)+p.epsilon);
 for(ulong j=tid;j<p.columns;j+=128)out[row*p.normStride+j]=layout[row*p.normStride+j]*inverse*(weight[j]+p.offset);
}
kernel void silu_mul(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    float gate=a[row*p.aStride+col];
    c[row*p.cStride+col]=(gate/(1.0f+exp(-gate)))*b[row*p.bStride+col];
}
struct JoinedGlobalSiluParams {ulong rows,columns,iStride,gStride,uStride,oStride;};
kernel void joined_global_silu(device const float* joined [[buffer(0)]],device float* gates [[buffer(1)]],
    device float* ups [[buffer(2)]],device float* activated [[buffer(3)]],
    constant JoinedGlobalSiluParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    float gate=joined[row*p.iStride+col],up=joined[row*p.iStride+p.columns+col];
    gates[row*p.gStride+col]=gate;ups[row*p.uStride+col]=up;
    activated[row*p.oStride+col]=(gate/(1.0f+exp(-gate)))*up;
}
kernel void add_matrix(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    c[row*p.cStride+col]=a[row*p.aStride+col]+b[(p.offset!=0?0:row)*p.bStride+col];
}
kernel void mask_rows(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    c[row*p.cStride+col]=b[row*p.bStride]!=0.0f?a[row*p.aStride+col]:0.0f;
}
struct RopeParams {ulong batch,heads,dim,rotary,iStride,pStride,fStride,oStride;float epsilon,weightOffset;ulong normalize;};
kernel void norm_rope(device const float* input [[buffer(0)]],device const float* weight [[buffer(1)]],
    device const float* positions [[buffer(2)]],device const float* frequencies [[buffer(3)]],
    device float* output [[buffer(4)]],constant RopeParams& p [[buffer(5)]],uint id [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    if(id>=p.batch*p.heads)return;
    ulong row=id/p.heads,head=id%p.heads,base=row*p.iStride+head*p.dim;
    float sum=0.0f;
    for(ulong j=lane;j<p.dim;j+=32){float x=input[base+j];sum+=x*x;}
    sum=simd_sum(sum);
    float inverse=p.normalize?rsqrt(sum/float(p.dim)+p.epsilon):1.0f;
    for(ulong j=lane;j<p.dim;j+=32) {
        float x=input[base+j]*inverse*(p.normalize?(p.weightOffset+weight[j]):1.0f);
        if(j<p.rotary) {
            ulong halfWidth=p.rotary/2,frequency=j%halfWidth,other=j<halfWidth?j+halfWidth:j-halfWidth;
            ulong axis=ulong(frequencies[p.fStride+frequency]);
            float angle=positions[row*p.pStride+axis]*frequencies[frequency];
            float y=input[base+other]*inverse*(p.normalize?(p.weightOffset+weight[other]):1.0f);
            x=x*precise::cos(angle)+(j<halfWidth?-y:y)*precise::sin(angle);
        }
        output[row*p.oStride+head*p.dim+j]=x;
    }
}
struct KVParams {ulong batch,heads,dim,capacity,tStride,cStride,sStride,aStride,oStride;};
kernel void kv_update(device const float* token [[buffer(0)]],device const float* cache [[buffer(1)]],
    device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],
    device float* output [[buffer(4)]],constant KVParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    ulong width=p.heads*p.capacity*p.dim;
    if(id>=p.batch*width)return;
    ulong row=id/width,col=id%width,dim=col%p.dim,slot=(col/p.dim)%p.capacity,head=col/(p.capacity*p.dim);
    bool replace=active[row*p.aStride]!=0.0f&&float(slot)==slots[row*p.sStride];
    output[row*p.oStride+col]=replace?token[row*p.tStride+head*p.dim+dim]:cache[row*p.cStride+col];
}
// A session-owned cache already contains every untouched slot.
kernel void kv_update_inplace(device const float* token [[buffer(0)]],device const float* cache [[buffer(1)]],
 device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
 constant KVParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
 ulong width=p.heads*p.dim;if(id>=p.batch*width)return;
 ulong row=id/width,head=(id%width)/p.dim,d=id%p.dim;
 if(active[row*p.aStride]==0.0f)return;
 float slot=slots[row*p.sStride];
 if(!isfinite(slot)||slot<0.0f||slot>=float(p.capacity))return;
 output[row*p.oStride+(head*p.capacity+ulong(slot))*p.dim+d]=token[row*p.tStride+head*p.dim+d];
}
struct KVChunkParams {ulong batch,heads,dim,capacity,chunk,tStride,cStride,sStride,aStride,oStride,contiguous;};
// Bit-preserving cache copies for aligned dimensions fitting uint indices.
kernel void kv_update_vector4(device const float* token [[buffer(0)]],device const float* cache [[buffer(1)]],
    device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],
    device float* output [[buffer(4)]],constant KVParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    const uint dim=uint(p.dim/4),capacity=uint(p.capacity),width=uint(p.heads)*capacity*dim;
    if(id>=uint(p.batch)*width)return;
    uint row=id/width,col=id%width,d=col%dim,slot=(col/dim)%capacity,head=col/(capacity*dim);
    bool replace=active[row*uint(p.aStride)]!=0.0f&&float(slot)==slots[row*uint(p.sStride)];
    device const float4* source=(device const float4*)token;
    device const float4* previous=(device const float4*)cache;
    device float4* next=(device float4*)output;
    next[row*uint(p.oStride/4)+col]=replace?source[row*uint(p.tStride/4)+head*dim+d]:previous[row*uint(p.cStride/4)+col];
}
kernel void kv_chunk(device const float* token [[buffer(0)]],device const float* cache [[buffer(1)]],
    device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
    constant KVChunkParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    ulong width=p.heads*p.capacity*p.dim;if(id>=p.batch*width)return;
    ulong row=id/width,col=id%width,d=col%p.dim,s=(col/p.dim)%p.capacity,h=col/(p.capacity*p.dim);
    int latest=-1;
    if(p.contiguous) {
        if(active[row*p.aStride]!=0.0f) {
            int candidate=int(s)-int(slots[row*p.sStride]);
            if(candidate>=0&&ulong(candidate)<p.chunk&&active[row*p.aStride+ulong(candidate)]!=0.0f)latest=candidate;
        }
    } else {
        for(uint t=0;t<p.chunk;++t)if(active[row*p.aStride+t]!=0.0f&&slots[row*p.sStride+t]==float(s))latest=int(t);
    }
    output[row*p.oStride+col]=latest<0?cache[row*p.cStride+col]:token[(row*p.chunk+ulong(latest))*p.tStride+h*p.dim+d];
}
kernel void kv_chunk_vector4(device const float* token [[buffer(0)]],device const float* cache [[buffer(1)]],
    device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
    constant KVChunkParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    const uint dim=uint(p.dim/4),capacity=uint(p.capacity),chunk=uint(p.chunk),width=uint(p.heads)*capacity*dim;
    if(id>=uint(p.batch)*width)return;
    uint row=id/width,col=id%width,d=col%dim,s=(col/dim)%capacity,h=col/(capacity*dim);
    int latest=-1;
    if(p.contiguous) {
        if(active[row*uint(p.aStride)]!=0.0f) {
            int candidate=int(s)-int(slots[row*uint(p.sStride)]);
            if(candidate>=0&&uint(candidate)<chunk&&active[row*uint(p.aStride)+uint(candidate)]!=0.0f)latest=candidate;
        }
    } else {
        for(uint t=0;t<chunk;++t)if(active[row*uint(p.aStride)+t]!=0.0f&&slots[row*uint(p.sStride)+t]==float(s))latest=int(t);
    }
    device const float4* source=(device const float4*)token;
    device const float4* previous=(device const float4*)cache;
    device float4* next=(device float4*)output;
    next[row*uint(p.oStride/4)+col]=latest<0?previous[row*uint(p.cStride/4)+col]:source[(row*chunk+uint(latest))*uint(p.tStride/4)+h*dim+d];
}
struct ChunkAttentionParams {ulong batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,kcStride,vcStride,sStride,aStride,gStride,oStride,gated;};
kernel void attention_chunk(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],device const float* keyCache [[buffer(3)]],device const float* valueCache [[buffer(4)]],
    device const float* slots [[buffer(5)]],device const float* active [[buffer(6)]],device const float* gate [[buffer(7)]],
    device float* output [[buffer(8)]],constant ChunkAttentionParams& p [[buffer(9)]],
    threadgroup float* scratch [[threadgroup(0)]],threadgroup int* selected [[threadgroup(1)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong flatRow=id/p.heads,head=id%p.heads,row=flatRow/p.chunk,token=flatRow%p.chunk,kv=head/(p.heads/p.kvHeads);
    if(flatRow>=p.batch*p.chunk)return;
    if(active[row*p.aStride+token]==0.0f){for(ulong d=tid;d<p.dim;d+=128)output[flatRow*p.oStride+head*p.dim+d]=0.0f;return;}
    const ulong visible=min(p.capacity,ulong(slots[row*p.sStride+token])+1);
    threadgroup float partial[4];
#ifdef GARNET_ATTENTION_PREFIX
    threadgroup int contiguousPrefix;
    if(tid==0) {
        float first=slots[row*p.sStride];
        int valid=isfinite(first)&&first>=0.0f&&first==floor(first);
        for(ulong t=0;t<=token;++t)if(active[row*p.aStride+t]==0.0f||slots[row*p.sStride+t]!=first+float(t))valid=0;
        contiguousPrefix=valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    for(ulong s=simd;s<visible;s+=4) {
#ifdef GARNET_ATTENTION_PREFIX
        int latest=-1;
        if(contiguousPrefix) {
            float first=slots[row*p.sStride];
            if(float(s)>=first)latest=int(float(s)-first);
        } else for(uint t=0;t<=token;++t)if(active[row*p.aStride+t]!=0.0f&&slots[row*p.sStride+t]==float(s))latest=int(t);
#else
        int latest=-1;for(uint t=0;t<=token;++t)if(active[row*p.aStride+t]!=0.0f&&slots[row*p.sStride+t]==float(s))latest=int(t);
#endif
        float score=-1e30f;
        if(float(s)<=slots[row*p.sStride+token]) {
            score=0.0f;
            for(ulong d=lane;d<p.dim;d+=32) {
                float k=latest<0?keyCache[row*p.kcStride+(kv*p.capacity+s)*p.dim+d]:key[(row*p.chunk+ulong(latest))*p.kStride+kv*p.dim+d];
                score+=query[flatRow*p.qStride+head*p.dim+d]*k;
            }
            score=simd_sum(score)*rsqrt(float(p.dim));
        }
        if(lane==0){scratch[s]=score;selected[s]=latest;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=0.0f;
    for(ulong s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
    denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denominator=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong d=tid;d<p.dim;d+=128) {
        float sum=0.0f;
        for(ulong s=0;s<visible;++s){int latest=selected[s];float v=latest<0?valueCache[row*p.vcStride+(kv*p.capacity+s)*p.dim+d]:value[(row*p.chunk+ulong(latest))*p.vStride+kv*p.dim+d];sum+=scratch[s]/denominator*v;}
        output[flatRow*p.oStride+head*p.dim+d]=p.gated?sum/(1.0f+exp(-gate[flatRow*p.gStride+head*p.dim+d])):sum;
    }
}
struct ScoresParams {ulong batch,heads,kvHeads,dim,capacity,qStride,kStride,sStride,aStride,oStride;};
kernel void attention_scores_serial(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
    constant ScoresParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.heads)return;
    ulong row=id/p.heads,head=id%p.heads,kv=head/(p.heads/p.kvHeads),base=row*p.oStride+head*p.capacity;
    if(active[row*p.aStride]==0.0f){for(ulong s=0;s<p.capacity;++s)output[base+s]=0.0f;return;}
    float maximum=-3.402823466e38f;
    for(ulong s=0;s<p.capacity;++s) {
        float score=-1e30f;
        if(float(s)<=slots[row*p.sStride]) {
            score=0.0f;
            for(ulong j=0;j<p.dim;++j)score+=query[row*p.qStride+head*p.dim+j]*key[row*p.kStride+(kv*p.capacity+s)*p.dim+j];
            score*=rsqrt(float(p.dim));
        }
        output[base+s]=score;maximum=max(maximum,score);
    }
    float denominator=0.0f;
    for(ulong s=0;s<p.capacity;++s){float probability=exp(output[base+s]-maximum);output[base+s]=probability;denominator+=probability;}
    for(ulong s=0;s<p.capacity;++s)output[base+s]/=denominator;
}
kernel void attention_scores(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
    constant ScoresParams& p [[buffer(5)]],threadgroup float* scratch [[threadgroup(0)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(id>=p.batch*p.heads)return;
    ulong row=id/p.heads,head=id%p.heads,kv=head/(p.heads/p.kvHeads),base=row*p.oStride+head*p.capacity;
    if(active[row*p.aStride]==0.0f){for(ulong s=tid;s<p.capacity;s+=128)output[base+s]=0.0f;return;}
    const ulong visible=min(p.capacity,ulong(slots[row*p.sStride])+1);
    threadgroup float partial[4];
    for(ulong s=simd;s<visible;s+=4) {
        float score=-1e30f;
        if(float(s)<=slots[row*p.sStride]) {
            score=0.0f;
            for(ulong j=lane;j<p.dim;j+=32)score+=query[row*p.qStride+head*p.dim+j]*key[row*p.kStride+(kv*p.capacity+s)*p.dim+j];
            score=simd_sum(score)*rsqrt(float(p.dim));
        }
        if(lane==0)scratch[s]=score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    // All threads consume the max before partial storage is reused for sums.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=0.0f;
    for(ulong s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
    denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denominator=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong s=tid;s<p.capacity;s+=128)output[base+s]=s<visible?scratch[s]/denominator:0.0f;
}
struct AttentionParams {ulong batch,heads,kvHeads,dim,capacity,pStride,vStride,gStride,sStride,oStride,gated;};
kernel void attention_context(device const float* probabilities [[buffer(0)]],device const float* value [[buffer(1)]],
    device const float* gate [[buffer(2)]],device const float* slots [[buffer(3)]],device float* output [[buffer(4)]],constant AttentionParams& p [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.heads*p.dim)return;
    ulong row=id/(p.heads*p.dim),head=(id/p.dim)%p.heads,col=id%p.dim,kv=head/(p.heads/p.kvHeads);
    const float slot=slots[row*p.sStride];
    const ulong visible=isfinite(slot)&&slot>=0.0f&&slot<float(p.capacity)?ulong(slot)+1:p.capacity;
    float result=0.0f;
    for(ulong s=0;s<visible;++s)result+=probabilities[row*p.pStride+head*p.capacity+s]*value[row*p.vStride+(kv*p.capacity+s)*p.dim+col];
    output[row*p.oStride+head*p.dim+col]=p.gated?result/(1.0f+exp(-gate[row*p.gStride+head*p.dim+col])):result;
}
struct PrepareParams {ulong batch,keyHeads,valueHeads,keyDim,valueDim,kSize,cStride,aStride,bStride,oStride,width;};
kernel void linear_prepare(device const float* conv [[buffer(0)]],device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],device const float* aLog [[buffer(3)]],device const float* dtBias [[buffer(4)]],
    device float* output [[buffer(5)]],constant PrepareParams& p [[buffer(6)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.width)return;
    ulong row=id/p.width,col=id%p.width,keyWidth=p.valueHeads*p.keyDim,valueWidth=p.valueHeads*p.valueDim;
    float result;
    if(col<2*keyWidth) {
        ulong part=col/keyWidth,within=col%keyWidth,head=within/p.keyDim,dim=within%p.keyDim;
        ulong channel=part*p.keyHeads*p.keyDim+(head/(p.valueHeads/p.keyHeads))*p.keyDim+dim;
        result=conv[row*p.cStride+channel*(p.kSize+1)];
    }else if(col<2*keyWidth+valueWidth) {
        ulong channel=2*p.keyHeads*p.keyDim+col-2*keyWidth;
        result=conv[row*p.cStride+channel*(p.kSize+1)];
    }else if(col<2*keyWidth+valueWidth+p.valueHeads) {
        ulong head=col-2*keyWidth-valueWidth;
        float x=a[row*p.aStride+head]+dtBias[head];
        float softplus=max(x,0.0f)+log(1.0f+exp(-abs(x)));
        result=-exp(aLog[head])*softplus;
    }else {
        ulong head=col-2*keyWidth-valueWidth-p.valueHeads;
        result=1.0f/(1.0f+exp(-b[row*p.bStride+head]));
    }
    output[row*p.oStride+col]=result;
}
struct GateParams {ulong batch,heads,valueDim,cStride,zStride,oStride,columnOffset;float epsilon;};
kernel void gated_norm(device const float* core [[buffer(0)]],device const float* z [[buffer(1)]],
    device const float* norm [[buffer(2)]],device float* output [[buffer(3)]],
    constant GateParams& p [[buffer(4)]],uint id [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    if(id>=p.batch*p.heads)return;
    ulong row=id/p.heads,head=id%p.heads;
    float sum=0.0f;
    for(ulong j=lane;j<p.valueDim;j+=32){float x=core[row*p.cStride+head*p.valueDim+j];sum+=x*x;}
    sum=simd_sum(sum);
    float inverse=rsqrt(sum/float(p.valueDim)+p.epsilon);
    for(ulong j=lane;j<p.valueDim;j+=32) {
        float gate=z[row*p.zStride+head*p.valueDim+j];
        output[row*p.oStride+p.columnOffset+head*p.valueDim+j]=core[row*p.cStride+head*p.valueDim+j]*inverse*norm[j]*gate/(1.0f+exp(-gate));
    }
}
struct ConcatParams {ulong batch,aWidth,bWidth,cWidth,aStride,bStride,cStride,oStride;};
kernel void concat_rows(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],device float* output [[buffer(3)]],
    constant ConcatParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    ulong width=p.aWidth+p.bWidth+p.cWidth;
    if(id>=p.batch*width)return;
    ulong row=id/width,col=id%width;
    output[row*p.oStride+col]=col<p.aWidth?a[row*p.aStride+col]:
        (col<p.aWidth+p.bWidth?b[row*p.bStride+col-p.aWidth]:c[row*p.cStride+col-p.aWidth-p.bWidth]);
}
struct SliceParams {ulong batch,groups,sourceWidth,start,count,iStride,oStride;};
kernel void slice_groups(device const float* input [[buffer(0)]],device float* output [[buffer(1)]],
    constant SliceParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.groups*p.count)return;
    ulong col=id%p.count,group=(id/p.count)%p.groups,row=id/(p.count*p.groups);
    output[row*p.oStride+group*p.count+col]=input[row*p.iStride+group*p.sourceWidth+p.start+col];
}
kernel void slice_groups_vector4(device const float4* input [[buffer(0)]],device float4* output [[buffer(1)]],
    constant SliceParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]) {
    uint count=uint(p.count/4),groups=uint(p.groups);
    if(id>=uint(p.batch)*groups*count)return;
    uint col=id%count,group=(id/count)%groups,row=id/(count*groups);
    output[row*uint(p.oStride/4)+group*count+col]=input[row*uint(p.iStride/4)+group*uint(p.sourceWidth/4)+uint(p.start/4)+col];
}
struct ConvParams {ulong batch,channels,kSize,tStride,wStride,sStride,aStride,oStride;};
kernel void causal_conv(device const float* token [[buffer(0)]],device const float* weight [[buffer(1)]],
    device const float* state [[buffer(2)]],device const float* active [[buffer(3)]],
    device float* packed [[buffer(4)]],constant ConvParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.channels)return;
    ulong row=id/p.channels,col=id%p.channels;
    bool enabled=active[row*p.aStride]!=0.0f;
    float sum=0.0f;
    for(ulong j=0;j<p.kSize;++j) {
        float next=j+1==p.kSize?token[row*p.tStride+col]:state[row*p.sStride+col*p.kSize+j+1];
        packed[row*p.oStride+col*(p.kSize+1)+1+j]=enabled?next:state[row*p.sStride+col*p.kSize+j];
        sum+=next*weight[col*p.wStride+j];
    }
    packed[row*p.oStride+col*(p.kSize+1)]=enabled?sum/(1.0f+exp(-sum)):0.0f;
}
struct ConvChunkParams {ulong batch,chunk,channels,kSize,tStride,wStride,sStride,aStride,oStride,nStride;};
kernel void causal_conv_chunk(device const float* tokens [[buffer(0)]],device const float* weight [[buffer(1)]],
    device const float* state [[buffer(2)]],device const float* active [[buffer(3)]],
    device float* output [[buffer(4)]],device float* nextState [[buffer(5)]],
    constant ConvChunkParams& p [[buffer(6)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.channels)return;
    ulong row=id/p.channels,col=id%p.channels;
    float history[8];
    for(ulong j=0;j<p.kSize;++j)history[j]=state[row*p.sStride+col*p.kSize+j];
    for(ulong token=0;token<p.chunk;++token) {
        bool enabled=active[row*p.aStride+token]!=0.0f;
        float sum=0.0f;
        for(ulong j=0;j<p.kSize;++j) {
            float next=j+1==p.kSize?tokens[(row*p.chunk+token)*p.tStride+col]:history[j+1];
            if(enabled)history[j]=next;
            sum+=next*weight[col*p.wStride+j];
        }
        output[(row*p.chunk+token)*p.oStride+col]=enabled?sum/(1.0f+exp(-sum)):0.0f;
    }
    for(ulong j=0;j<p.kSize;++j)nextState[row*p.nStride+col*p.kSize+j]=history[j];
}
struct DeltaParams {ulong batch,heads,keyDim,valueDim,qStride,kStride,vStride,gStride,bStride,sStride,aStride,oStride;};
kernel void gated_delta(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],device const float* decay [[buffer(3)]],
    device const float* beta [[buffer(4)]],device const float* state [[buffer(5)]],
    device const float* active [[buffer(6)]],device float* packed [[buffer(7)]],
    constant DeltaParams& p [[buffer(8)]],uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(id>=p.batch*p.heads)return;
    ulong head=id%p.heads,row=id/p.heads;
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim;
    ulong outputBase=row*p.oStride+head*(p.keyDim+1)*p.valueDim;
    ulong outBase=outputBase+p.valueDim;
    if(active[row*p.aStride]==0.0f) {
        for(ulong column=tid;column<p.valueDim;column+=128) {
            packed[outputBase+column]=0.0f;
            for(ulong j=0;j<p.keyDim;++j)packed[outBase+j*p.valueDim+column]=state[stateBase+j*p.valueDim+column];
        }
        return;
    }
    threadgroup float qPartial[4],kPartial[4];
    float qs=0.0f,ks=0.0f;
    for(ulong j=tid;j<p.keyDim;j+=128) {
        float q=query[row*p.qStride+head*p.keyDim+j],k=key[row*p.kStride+head*p.keyDim+j];qs+=q*q;ks+=k*k;
    }
    qs=simd_sum(qs);ks=simd_sum(ks);if(lane==0){qPartial[simd]=qs;kPartial[simd]=ks;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qs=simd_sum(lane<4?qPartial[lane]:0.0f)+1e-6f;ks=simd_sum(lane<4?kPartial[lane]:0.0f)+1e-6f;
    float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks);
    for(ulong column=tid;column<p.valueDim;column+=128) {
    float factor=exp(decay[row*p.gStride+head]),memory=0.0f;
    for(ulong j=0;j<p.keyDim;++j)
        memory+=state[stateBase+j*p.valueDim+column]*factor*key[row*p.kStride+head*p.keyDim+j]*ki;
    float change=(value[row*p.vStride+head*p.valueDim+column]-memory)*beta[row*p.bStride+head];
    float result=0.0f;
    for(ulong j=0;j<p.keyDim;++j) {
        float next=state[stateBase+j*p.valueDim+column]*factor+key[row*p.kStride+head*p.keyDim+j]*ki*change;
        packed[outBase+j*p.valueDim+column]=next;
        result+=next*query[row*p.qStride+head*p.keyDim+j]*qi;
    }
    packed[outputBase+column]=result;
    }
}
struct DeltaSplitParams {ulong batch,heads,keyDim,valueDim,qStride,kStride,vStride,gStride,bStride,sStride,aStride,oStride,nextStride;};
kernel void gated_delta_split(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],device const float* decay [[buffer(3)]],
    device const float* beta [[buffer(4)]],device const float* state [[buffer(5)]],
    device const float* active [[buffer(6)]],device float* resultOutput [[buffer(7)]],device float* nextState [[buffer(8)]],
    constant DeltaSplitParams& p [[buffer(9)]],uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(id>=p.batch*p.heads)return;
    ulong head=id%p.heads,row=id/p.heads;
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim;
    ulong outputBase=row*p.oStride+head*p.valueDim;
    ulong outBase=row*p.nextStride+head*p.keyDim*p.valueDim;
    if(active[row*p.aStride]==0.0f) {
        for(ulong column=tid;column<p.valueDim;column+=128) {
            resultOutput[outputBase+column]=0.0f;
            for(ulong j=0;j<p.keyDim;++j)nextState[outBase+j*p.valueDim+column]=state[stateBase+j*p.valueDim+column];
        }
        return;
    }
    threadgroup float qPartial[4],kPartial[4];
    float qs=0.0f,ks=0.0f;
    for(ulong j=tid;j<p.keyDim;j+=128) {
        float q=query[row*p.qStride+head*p.keyDim+j],k=key[row*p.kStride+head*p.keyDim+j];qs+=q*q;ks+=k*k;
    }
    qs=simd_sum(qs);ks=simd_sum(ks);if(lane==0){qPartial[simd]=qs;kPartial[simd]=ks;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qs=simd_sum(lane<4?qPartial[lane]:0.0f)+1e-6f;ks=simd_sum(lane<4?kPartial[lane]:0.0f)+1e-6f;
    float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks);
    for(ulong column=tid;column<p.valueDim;column+=128) {
    float factor=exp(decay[row*p.gStride+head]),memory=0.0f;
    for(ulong j=0;j<p.keyDim;++j)
        memory+=state[stateBase+j*p.valueDim+column]*factor*key[row*p.kStride+head*p.keyDim+j]*ki;
    float change=(value[row*p.vStride+head*p.valueDim+column]-memory)*beta[row*p.bStride+head];
    float result=0.0f;
    for(ulong j=0;j<p.keyDim;++j) {
        float next=state[stateBase+j*p.valueDim+column]*factor+key[row*p.kStride+head*p.keyDim+j]*ki*change;
        nextState[outBase+j*p.valueDim+column]=next;
        result+=next*query[row*p.qStride+head*p.keyDim+j]*qi;
    }
    resultOutput[outputBase+column]=result;
    }
}
kernel void gated_delta_split_parallel(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],device const float* decay [[buffer(3)]],
    device const float* beta [[buffer(4)]],device const float* state [[buffer(5)]],
    device const float* active [[buffer(6)]],device float* resultOutput [[buffer(7)]],device float* nextState [[buffer(8)]],
    constant DeltaSplitParams& p [[buffer(9)]],uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong tiles=p.valueDim/16,headId=ulong(id)/tiles;
    if(headId>=p.batch*p.heads)return;
    ulong head=headId%p.heads,row=headId/p.heads,column=(ulong(id)%tiles)*16+simd*4;
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim;
    ulong outputBase=row*p.oStride+head*p.valueDim;
    ulong outBase=row*p.nextStride+head*p.keyDim*p.valueDim;
    if(active[row*p.aStride]==0.0f) {
        for(ulong j=lane;j<p.keyDim;j+=32)
            *(device float4*)(nextState+outBase+j*p.valueDim+column)=*(device const float4*)(state+stateBase+j*p.valueDim+column);
        if(lane==0)*(device float4*)(resultOutput+outputBase+column)=float4(0.0f);
        return;
    }
    threadgroup float qPartial[4],kPartial[4];
    float qs=0.0f,ks=0.0f;
    for(ulong j=tid;j<p.keyDim;j+=128) {
        float q=query[row*p.qStride+head*p.keyDim+j],k=key[row*p.kStride+head*p.keyDim+j];qs+=q*q;ks+=k*k;
    }
    qs=simd_sum(qs);ks=simd_sum(ks);if(lane==0){qPartial[simd]=qs;kPartial[simd]=ks;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qs=simd_sum(lane<4?qPartial[lane]:0.0f)+1e-6f;ks=simd_sum(lane<4?kPartial[lane]:0.0f)+1e-6f;
    float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks);
    float factor=exp(decay[row*p.gStride+head]);float4 memory=0.0f;
    for(ulong j=lane;j<p.keyDim;j+=32)
        memory+=*(device const float4*)(state+stateBase+j*p.valueDim+column)*factor*key[row*p.kStride+head*p.keyDim+j]*ki;
    memory=float4(simd_sum(memory.x),simd_sum(memory.y),simd_sum(memory.z),simd_sum(memory.w));
    float4 change=(*(device const float4*)(value+row*p.vStride+head*p.valueDim+column)-memory)*beta[row*p.bStride+head];
    float4 result=0.0f;
    for(ulong j=lane;j<p.keyDim;j+=32) {
        float4 next=*(device const float4*)(state+stateBase+j*p.valueDim+column)*factor+key[row*p.kStride+head*p.keyDim+j]*ki*change;
        *(device float4*)(nextState+outBase+j*p.valueDim+column)=next;
        result+=next*query[row*p.qStride+head*p.keyDim+j]*qi;
    }
    result=float4(simd_sum(result.x),simd_sum(result.y),simd_sum(result.z),simd_sum(result.w));
    if(lane==0)*(device float4*)(resultOutput+outputBase+column)=result;
}
struct DeltaChunkParams {ulong batch,chunk,heads,keyDim,valueDim,pStride,sStride,aStride,oStride,nextStride;};
kernel void gated_delta_chunk(device const float* prepared [[buffer(0)]],device const float* state [[buffer(1)]],
    device const float* active [[buffer(2)]],device float* output [[buffer(3)]],device float* nextState [[buffer(4)]],
    constant DeltaChunkParams& p [[buffer(5)]],uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong tiles=p.valueDim/16,headId=ulong(id)/tiles;
    if(headId>=p.batch*p.heads)return;
    ulong head=headId%p.heads,row=headId/p.heads,column=(ulong(id)%tiles)*16+simd*4;
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim,nextBase=row*p.nextStride+head*p.keyDim*p.valueDim;
    float4 memories[4];
    for(uint j=0;j<4;++j)memories[j]=*(device const float4*)(state+stateBase+(ulong(lane)+32*j)*p.valueDim+column);
    threadgroup float qPartial[4],kPartial[4];
    ulong qw=p.heads*p.keyDim,vw=p.heads*p.valueDim;
    for(ulong token=0;token<p.chunk;++token) {
        ulong base=(row*p.chunk+token)*p.pStride,outputBase=(row*p.chunk+token)*p.oStride+head*p.valueDim;
        if(active[row*p.aStride+token]!=0.0f) {
            float q=prepared[base+head*p.keyDim+tid],k=prepared[base+qw+head*p.keyDim+tid];
            float qs=simd_sum(q*q),ks=simd_sum(k*k);
            if(lane==0){qPartial[simd]=qs;kPartial[simd]=ks;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            qs=simd_sum(lane<4?qPartial[lane]:0.0f)+1e-6f;ks=simd_sum(lane<4?kPartial[lane]:0.0f)+1e-6f;
            float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks),factor=exp(prepared[base+2*qw+vw+head]);
            float4 memory=0.0f;
            for(uint j=0;j<4;++j)memory+=memories[j]*factor*prepared[base+qw+head*p.keyDim+ulong(lane)+32*j]*ki;
            memory=float4(simd_sum(memory.x),simd_sum(memory.y),simd_sum(memory.z),simd_sum(memory.w));
            float4 change=(*(device const float4*)(prepared+base+2*qw+head*p.valueDim+column)-memory)*prepared[base+2*qw+vw+p.heads+head];
            float4 result=0.0f;
            for(uint j=0;j<4;++j) {
                ulong keyIndex=ulong(lane)+32*j;
                memories[j]=memories[j]*factor+prepared[base+qw+head*p.keyDim+keyIndex]*ki*change;
                result+=memories[j]*prepared[base+head*p.keyDim+keyIndex]*qi;
            }
            result=float4(simd_sum(result.x),simd_sum(result.y),simd_sum(result.z),simd_sum(result.w));
            if(lane==0)*(device float4*)(output+outputBase+column)=result;
        }else if(lane==0)*(device float4*)(output+outputBase+column)=float4(0.0f);
        // Every SIMD group must finish reading the partial norms before any
        // group can replace them for the next token.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<4;++j)*(device float4*)(nextState+nextBase+(ulong(lane)+32*j)*p.valueDim+column)=memories[j];
}
struct LayoutParams {ulong aRows,aColumns,aStride,bStride,oRows,oColumns,oStride,tokenWidth;};
kernel void chunk_layout(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
 device float* out [[buffer(2)]],constant LayoutParams& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.oRows*p.oColumns)return;
 ulong sourceRow=id/p.aColumns,sourceColumn=id%p.aColumns,row=id/p.oColumns,col=id%p.oColumns;
 float value=a[sourceRow*p.aStride+sourceColumn];
 if(p.tokenWidth&&b[sourceRow*p.bStride+sourceColumn/p.tokenWidth]==0.0f)value=0;
 out[row*p.oStride+col]=value;
}
struct VisionParams {ulong rows,columns,outRows,outColumns,aStride,bStride,cStride,oStride,tokens,hidden,heads,unit,mode;};
kernel void vision_norm(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.outRows)return;threadgroup float partial[4];
    float mean=0.0f;for(ulong j=tid;j<p.columns;j+=128)mean+=a[row*p.aStride+j];
    mean=simd_sum(mean);if(lane==0)partial[simd]=mean;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean=simd_sum(lane<4?partial[lane]:0.0f)/float(p.columns);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float variance=0.0f;for(ulong j=tid;j<p.columns;j+=128){float x=a[row*p.aStride+j]-mean;variance+=x*x;}
    variance=simd_sum(variance);if(lane==0)partial[simd]=variance;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    variance=simd_sum(lane<4?partial[lane]:0.0f);
    float inverse=rsqrt(variance/float(p.columns)+1e-6f);
    for(ulong j=tid;j<p.columns;j+=128)out[row*p.oStride+j]=(a[row*p.aStride+j]-mean)*inverse*b[j]+c[j];
}
kernel void vision_scores(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],threadgroup float* scratch [[threadgroup(0)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong row=id/p.heads,head=id%p.heads,dim=p.hidden/p.heads,batchBase=(row/p.tokens)*p.tokens;
    if(row>=p.outRows)return;threadgroup float partial[4];
    for(ulong key=simd;key<p.tokens;key+=4) {
        float score=0;for(ulong d=lane;d<dim;d+=32)score+=a[row*p.aStride+head*dim+d]*a[(batchBase+key)*p.aStride+p.hidden+head*dim+d];
        score=simd_sum(score)*rsqrt(float(dim));if(lane==0)scratch[key]=score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong key=tid;key<p.tokens;key+=128)maximum=max(maximum,scratch[key]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;for(ulong key=tid;key<p.tokens;key+=128){float v=exp(scratch[key]-maximum);scratch[key]=v;total+=v;}
    total=simd_sum(total);if(lane==0)partial[simd]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong key=tid;key<p.tokens;key+=128)out[row*p.oStride+head*p.tokens+key]=scratch[key]/total;
}
kernel void vision_pack_qk(device const float* input [[buffer(0)]],device float* query [[buffer(1)]],
    device float* key [[buffer(2)]],constant VisionParams& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    ulong dim=p.hidden/p.heads,packedRow=ulong(id)/dim,d=ulong(id)%dim;
    if(packedRow>=p.rows*p.heads)return;
    ulong row=(packedRow/(p.heads*p.tokens))*p.tokens+packedRow%p.tokens,head=(packedRow/p.tokens)%p.heads;
    query[packedRow*p.bStride+d]=input[row*p.aStride+head*dim+d];
    key[packedRow*p.cStride+d]=input[row*p.aStride+p.hidden+head*dim+d];
}
kernel void vision_softmax(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],threadgroup float* scratch [[threadgroup(0)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong row=id/p.heads,head=id%p.heads,dim=p.hidden/p.heads,batchBase=(row/p.tokens)*p.tokens;
    if(row>=p.outRows)return;threadgroup float partial[4];
    ulong packedRow=((row/p.tokens)*p.heads+head)*p.tokens+row%p.tokens;
    for(ulong key=tid;key<p.tokens;key+=128)scratch[key]=b[packedRow*p.bStride+key]*rsqrt(float(dim));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong key=tid;key<p.tokens;key+=128)maximum=max(maximum,scratch[key]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;for(ulong key=tid;key<p.tokens;key+=128){float v=exp(scratch[key]-maximum);scratch[key]=v;total+=v;}
    total=simd_sum(total);if(lane==0)partial[simd]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong key=tid;key<p.tokens;key+=128)out[row*p.oStride+head*p.tokens+key]=scratch[key]/total;
}
// Input/output matrices contain flattened batch/token rows. The merger copies
// across padded MPS row boundaries explicitly rather than aliasing row strides.
kernel void vision_op(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],device float* out [[buffer(3)]],
    constant VisionParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.outRows*p.outColumns)return;
    ulong row=id/p.outColumns,col=id%p.outColumns;
    if(p.mode==0){out[row*p.oStride+col]=a[row*p.aStride+col]+b[col];return;}
    if(p.mode==1){
        // Compute row statistics once rather than repeating both sums per column.
        if(col)return;
        float mean=0;for(ulong j=0;j<p.columns;++j)mean+=a[row*p.aStride+j];mean/=float(p.columns);
        float variance=0;for(ulong j=0;j<p.columns;++j){float x=a[row*p.aStride+j]-mean;variance+=x*x;}
        float inverse=rsqrt(variance/float(p.columns)+1e-6f);
        for(ulong j=0;j<p.columns;++j)out[row*p.oStride+j]=(a[row*p.aStride+j]-mean)*inverse*b[j]+c[j];return;
    }
    if(p.mode==2||p.mode==3){
        float x=a[row*p.aStride+col],activation;
        // Avoid native tanh overflow; at |argument| >= 10 its FP32 value saturates.
        if(p.mode==2)activation=tanh(clamp(0.7978845608028654f*(x+0.044715f*x*x*x),-10.0f,10.0f));
        else {
            // Abramowitz-Stegun erf approximation (absolute error < 1.5e-7).
            float z=abs(x)*0.7071067811865475f,t=1.0f/(1.0f+0.3275911f*z);
            float v=1.0f-(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t*exp(-z*z);
            activation=x<0?-v:v;
        }
        out[row*p.oStride+col]=0.5f*x*(1.0f+activation);return;
    }
    if(p.mode==4){out[row*p.oStride+col]=a[(row*p.unit+col/p.columns)*p.aStride+col%p.columns];return;}
    if(p.mode==9){out[row*p.oStride+col]=a[(row/p.tokens)*p.aStride+(row%p.tokens)*p.outColumns+col];return;}
    if(p.mode==5){
        ulong token=row%p.tokens;float position=0;
        for(ulong i=0;i<4;++i)position+=b[ulong(c[token*p.cStride+i])*p.bStride+col]*c[token*p.cStride+4+i];
        out[row*p.oStride+col]=a[row*p.aStride+col]+position;return;
    }
    ulong dim=p.hidden/p.heads;
    if(p.mode==6){
        if(col>=2*p.hidden){out[row*p.oStride+col]=a[row*p.aStride+col];return;}
        ulong d=col%dim,base=col-d,other=(d+dim/2)%dim;
        float angle=b[(row%p.tokens)*p.bStride+d];
        out[row*p.oStride+col]=a[row*p.aStride+col]*cos(angle)+(d<dim/2?-1.0f:1.0f)*a[row*p.aStride+base+other]*sin(angle);return;
    }
    if(p.mode==7){
        // One thread handles a whole softmax row; dispatch grid includes
        // columns, but only its first column performs the reduction.
        if(col%p.tokens)return;
        ulong head=col/p.tokens,batchBase=(row/p.tokens)*p.tokens;float maximum=-INFINITY;
        for(ulong key=0;key<p.tokens;++key){
            float score=0;for(ulong d=0;d<dim;++d)score+=a[row*p.aStride+head*dim+d]*a[(batchBase+key)*p.aStride+p.hidden+head*dim+d];
            score*=rsqrt(float(dim));out[row*p.oStride+head*p.tokens+key]=score;maximum=max(maximum,score);
        }
        float total=0;for(ulong key=0;key<p.tokens;++key){float v=exp(out[row*p.oStride+head*p.tokens+key]-maximum);out[row*p.oStride+head*p.tokens+key]=v;total+=v;}
        for(ulong key=0;key<p.tokens;++key)out[row*p.oStride+head*p.tokens+key]/=total;return;
    }
    if(p.mode==8){
        ulong head=col/dim,d=col%dim,batchBase=(row/p.tokens)*p.tokens;float result=0;
        for(ulong key=0;key<p.tokens;++key)result+=a[row*p.aStride+head*p.tokens+key]*b[(batchBase+key)*p.bStride+2*p.hidden+head*dim+d];
        out[row*p.oStride+col]=result;
    }
}
)METAL"];
        NSError* error=nil;
#ifdef GARNET_METAL_ATTENTION_PREFIX
        source=[@"#define GARNET_ATTENTION_PREFIX 1\n" stringByAppendingString:source];
#endif
#ifdef GARNET_METAL_FUSED_VISION_CONTEXT
        source=[source stringByAppendingString:[NSString stringWithUTF8String:R"METAL(
kernel void vision_softmax_context(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],threadgroup float* scratch [[threadgroup(0)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong row=id/p.heads,head=id%p.heads,dim=p.hidden/p.heads,batchBase=(row/p.tokens)*p.tokens;
    if(row>=p.outRows)return;threadgroup float partial[4];
    ulong packedRow=((row/p.tokens)*p.heads+head)*p.tokens+row%p.tokens;
    for(ulong key=tid;key<p.tokens;key+=128)scratch[key]=b[packedRow*p.bStride+key]*rsqrt(float(dim));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong key=tid;key<p.tokens;key+=128)maximum=max(maximum,scratch[key]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;for(ulong key=tid;key<p.tokens;key+=128){float v=exp(scratch[key]-maximum);scratch[key]=v;total+=v;}
    total=simd_sum(total);if(lane==0)partial[simd]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong key=tid;key<p.tokens;key+=128)scratch[key]/=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(ulong d=tid;d<dim;d+=128) {
        float result=0;
        for(ulong key=0;key<p.tokens;++key)result+=scratch[key]*a[(batchBase+key)*p.aStride+2*p.hidden+head*dim+d];
        out[row*p.oStride+head*dim+d]=result;
    }
}
)METAL"]];
#endif
#ifdef GARNET_METAL_PARALLEL_VISION_CONTEXT
        source=[source stringByAppendingString:[NSString stringWithUTF8String:kParallelVisionContextShader]];
#endif
#ifdef GARNET_METAL_JOINED_PREFILL
        source=[source stringByAppendingString:@"\nkernel void joined_split_rescale(device const float* input [[buffer(0)]],device const int* scales [[buffer(1)]],device float* first [[buffer(2)]],device float* second [[buffer(3)]],constant HalfRescaleParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {if(id>=p.rows*p.columns*2)return;ulong row=id/(p.columns*2),col=id%(p.columns*2);float v=ldexp(input[row*p.iStride+col],-scales[col]);if(col<p.columns)first[row*p.oStride+col]=v;else second[row*p.oStride+col-p.columns]=v;}\n"];
#endif
        MTLCompileOptions* compileOptions=nil;
#ifdef GARNET_METAL_NATIVE_BFLOAT_BODY
        source=[source stringByAppendingString:[NSString stringWithUTF8String:R"METAL(
kernel void bf16_native_serial_t64(device const float* a [[buffer(0)]],device const bfloat* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*8+simd*4;
    if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            bfloat4 b0=*(device const bfloat4*)(weight+column*p.inner+k),b1=bfloat4(0),b2=bfloat4(0),b3=bfloat4(0);
            if(column+1<p.columns)b1=*(device const bfloat4*)(weight+(column+1)*p.inner+k);
            if(column+2<p.columns)b2=*(device const bfloat4*)(weight+(column+2)*p.inner+k);
            if(column+3<p.columns)b3=*(device const bfloat4*)(weight+(column+3)*p.inner+k);
            for(uint j=0;j<4;++j) {
                float4 w=float4(float(b0[j]),float(b1[j]),float(b2[j]),float(b3[j]));
                sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<p.inner;k+=32) {
        float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
        float4 w=0.0f;
        w.x=float(weight[column*p.inner+k]);
        if(column+1<p.columns)w.y=float(weight[(column+1)*p.inner+k]);
        if(column+2<p.columns)w.z=float(weight[(column+2)*p.inner+k]);
        if(column+3<p.columns)w.w=float(weight[(column+3)*p.inner+k]);
        sum0+=x*w;if(p.rows==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
kernel void bf16_native_dot_t128(device const float* a [[buffer(0)]],device const bfloat* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            bfloat4 b0=*(device const bfloat4*)(weight+column*p.inner+k),b1=bfloat4(0),b2=bfloat4(0),b3=bfloat4(0);
            if(column+1<p.columns)b1=*(device const bfloat4*)(weight+(column+1)*p.inner+k);
            if(column+2<p.columns)b2=*(device const bfloat4*)(weight+(column+2)*p.inner+k);
            if(column+3<p.columns)b3=*(device const bfloat4*)(weight+(column+3)*p.inner+k);
            float4 w0=float4(b0),w1=float4(b1),w2=float4(b2),w3=float4(b3);
            sum0+=float4(dot(x,w0),dot(x,w1),dot(x,w2),dot(x,w3));
            if(p.rows==2)sum1+=float4(dot(y,w0),dot(y,w1),dot(y,w2),dot(y,w3));
        }
    } else {
    for(ulong k=lane;k<p.inner;k+=32) {
        float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
        float4 w=0.0f;
        w.x=float(weight[column*p.inner+k]);
        if(column+1<p.columns)w.y=float(weight[(column+1)*p.inner+k]);
        if(column+2<p.columns)w.z=float(weight[(column+2)*p.inner+k]);
        if(column+3<p.columns)w.w=float(weight[(column+3)*p.inner+k]);
        sum0+=x*w;if(p.rows==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
)METAL"]];
        compileOptions=[MTLCompileOptions new];compileOptions.languageVersion=MTLLanguageVersion3_1;
#endif
#ifdef GARNET_METAL_CONSTANT_BF16_HEAD
        const auto headStart=[source rangeOfString:@"kernel void bf16_decode("];
        const auto headEnd=[source rangeOfString:@"kernel void bf16_decode_small("];
        if(headStart.location==NSNotFound||headEnd.location==NSNotFound||headEnd.location<=headStart.location)
            throw std::runtime_error("Cannot locate original BF16 vocabulary shader");
        auto headSource=[source substringWithRange:NSMakeRange(headStart.location,headEnd.location-headStart.location)];
        headSource=[headSource stringByReplacingOccurrencesOfString:@"kernel void bf16_decode(" withString:@"kernel void bf16_decode_constant_head("];
        headSource=[headSource stringByReplacingOccurrencesOfString:@"device const ushort* weight" withString:@"constant ushort* weight"];
        headSource=[headSource stringByReplacingOccurrencesOfString:@"device const ushort4*" withString:@"constant ushort4*"];
        source=[source stringByAppendingString:headSource];
#endif
#ifdef GARNET_METAL_SINGLE_CODER_HEAD
        const auto singleStart=[source rangeOfString:@"kernel void bf16_decode("];
        const auto singleEnd=[source rangeOfString:@"kernel void bf16_decode_small("];
        if(singleStart.location==NSNotFound||singleEnd.location==NSNotFound||singleEnd.location<=singleStart.location)
            throw std::runtime_error("Cannot locate original coder vocabulary shader");
        auto singleSource=[source substringWithRange:NSMakeRange(singleStart.location,singleEnd.location-singleStart.location)];
        singleSource=[singleSource stringByReplacingOccurrencesOfString:@"kernel void bf16_decode(" withString:@"kernel void bf16_decode_single_coder_head("];
        singleSource=[singleSource stringByReplacingOccurrencesOfString:@"p.rows==2" withString:@"false"];
        source=[source stringByAppendingString:singleSource];
#endif
#ifdef GARNET_METAL_BODY_SHAPE
        source=[source stringByAppendingString:[NSString stringWithUTF8String:kBodyShapeShader]];
#endif
        auto library=[impl_->device newLibraryWithSource:source options:compileOptions error:&error];
        if(!library)throw std::runtime_error(std::string(error.localizedDescription.UTF8String));
        impl_->kernels->library=library;
        impl_->kernels->argmax=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"argmax_rows"] error:&error];
#ifdef GARNET_METAL_SERIAL_VISION_PREFETCH
        auto prefetchOptions=[MTLCompileOptions new];prefetchOptions.fastMathEnabled=NO;
        auto prefetchLibrary=[impl_->device newLibraryWithSource:[NSString stringWithUTF8String:kSerialVisionPrefetchShader] options:prefetchOptions error:&error];
        if(prefetchLibrary){
            impl_->kernels->visionPrefetchLibrary=prefetchLibrary;
            impl_->kernels->visionPrefetch=[impl_->device newComputePipelineStateWithFunction:[prefetchLibrary newFunctionWithName:@"vision_prefetch8"] error:&error];
        }
#endif
#ifdef GARNET_METAL_SCREENED_HEAD
        auto screenOptions=[MTLCompileOptions new];screenOptions.fastMathEnabled=NO;
        screenOptions.preprocessorMacros=@{@"GARNET_SCREENED_HEAD_SEED_ROWS":@(kScreenedHeadSeedRows),@"GARNET_SCREENED_HEAD_INTERLEAVED":@(int(kScreenedHeadInterleaved))};
        auto screenLibrary=[impl_->device newLibraryWithSource:[NSString stringWithUTF8String:kScreenedHeadShader] options:screenOptions error:&error];
        if(screenLibrary){
            impl_->kernels->screenLibrary=screenLibrary;
            impl_->kernels->screenPrepare=[impl_->device newComputePipelineStateWithFunction:[screenLibrary newFunctionWithName:@"prepare_head_bounds"] error:&error];
            impl_->kernels->screenProjection=[impl_->device newComputePipelineStateWithFunction:[screenLibrary newFunctionWithName:@"screened_head"] error:&error];
        }
#endif
#ifdef GARNET_METAL_FUSED_HEAD_ARGMAX
        auto greedyLibrary=[impl_->device newLibraryWithSource:[NSString stringWithUTF8String:kFusedHeadArgmaxShader] options:compileOptions error:&error];
        if(greedyLibrary) {
            impl_->kernels->fusedHead=[impl_->device newComputePipelineStateWithFunction:[greedyLibrary newFunctionWithName:@"fused_head"] error:&error];
            impl_->kernels->fusedHeadReduce=[impl_->device newComputePipelineStateWithFunction:[greedyLibrary newFunctionWithName:@"fused_head_reduce"] error:&error];
        }
#endif
#ifdef GARNET_METAL_RESIDENT_KV_INPLACE
        impl_->kernels->kvInplace=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kv_update_inplace"] error:&error];
        if(!impl_->kernels->kvInplace)throw std::runtime_error("Cannot create private KV update pipeline");
#endif
        impl_->kernels->bf16Decode=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_decode"] error:&error];
#ifdef GARNET_METAL_SINGLE_CODER_HEAD
        impl_->kernels->singleCoderHead=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_decode_single_coder_head"] error:&error];
        if(!impl_->kernels->singleCoderHead)throw std::runtime_error("Cannot create single-row coder vocabulary pipeline");
#endif
#ifdef GARNET_METAL_CONSTANT_BF16_HEAD
        impl_->kernels->constantHead=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_decode_constant_head"] error:&error];
        if(!impl_->kernels->constantHead)throw std::runtime_error("Cannot create constant-address vocabulary pipeline");
#endif
#ifdef GARNET_METAL_NATIVE_BFLOAT_BODY
        impl_->kernels->nativeBFSerial=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_native_serial_t64"] error:&error];
        impl_->kernels->nativeBFDot=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_native_dot_t128"] error:&error];
        if(!impl_->kernels->nativeBFSerial||!impl_->kernels->nativeBFDot)throw std::runtime_error("Cannot create native BF16 body pipelines");
#endif
#ifdef GARNET_METAL_SMALL_OUTPUT_DECODE
        impl_->kernels->bf16DecodeSmall=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_decode_small"] error:&error];
        if(!impl_->kernels->bf16DecodeSmall)throw std::runtime_error("Cannot create small-output decode pipeline");
#endif
        impl_->kernels->bf16Gemm=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_gemm"] error:&error];
#ifdef GARNET_METAL_FLOAT_DENSE_KEY_PREFILL
        impl_->kernels->bf16Split=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_split_key"] error:&error];
        impl_->kernels->bf16SplitReduce=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_split_reduce"] error:&error];
        if(!impl_->kernels->bf16Split||!impl_->kernels->bf16SplitReduce)throw std::runtime_error("Cannot create split key projection pipelines");
#endif
        impl_->kernels->halfRescale=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"half_rescale"] error:&error];
#ifdef GARNET_METAL_JOINED_PREFILL
        impl_->kernels->joinedSplitRescale=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"joined_split_rescale"] error:&error];
        if(!impl_->kernels->joinedSplitRescale)throw std::runtime_error("Cannot create joined prefill rescale pipeline");
#endif
        impl_->kernels->bf16Gemm16Wide=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gemm_16_64_64"] error:&error];
        impl_->kernels->bf16Gemm32Wide=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gemm_32_64_32"] error:&error];
        impl_->kernels->rms=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rms_norm"] error:&error];
#ifdef GARNET_METAL_PACKED_PREFILL_NORM
        impl_->kernels->packedPrefillNorm=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"packed_residual_norm"] error:&error];
        if(!impl_->kernels->packedPrefillNorm)throw std::runtime_error("Cannot create packed prefill normalization pipeline");
#endif
#ifdef GARNET_METAL_RESIDUAL_NORM
        impl_->kernels->residualNorm=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"residual_norm"] error:&error];
        if(!impl_->kernels->residualNorm)throw std::runtime_error("Cannot create residual normalization pipeline");
#endif
        impl_->kernels->silu=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"silu_mul"] error:&error];
#ifdef GARNET_METAL_JOINED_GLOBAL_MLP
        impl_->kernels->joinedGlobalSilu=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"joined_global_silu"] error:&error];
        if(!impl_->kernels->joinedGlobalSilu)throw std::runtime_error("Cannot create joined global split-SiLU pipeline");
#endif
        impl_->kernels->add=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"add_matrix"] error:&error];
        impl_->kernels->mask=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"mask_rows"] error:&error];
        impl_->kernels->conv=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"causal_conv"] error:&error];
#ifdef GARNET_METAL_LINEAR_CHUNK_CONV
        impl_->kernels->convChunk=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"causal_conv_chunk"] error:&error];
        if(!impl_->kernels->convChunk)throw std::runtime_error("Cannot create chunk convolution pipeline");
#endif
        impl_->kernels->delta=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gated_delta"] error:&error];
        impl_->kernels->deltaSplit=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gated_delta_split"] error:&error];
#ifdef GARNET_METAL_PARALLEL_DELTA_SPLIT
        impl_->kernels->deltaSplitParallel=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gated_delta_split_parallel"] error:&error];
        if(!impl_->kernels->deltaSplitParallel)throw std::runtime_error("Cannot create parallel recurrent pipeline");
#endif
#ifdef GARNET_METAL_LINEAR_CHUNK_DELTA
        impl_->kernels->deltaChunk=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gated_delta_chunk"] error:&error];
        if(!impl_->kernels->deltaChunk)throw std::runtime_error("Cannot create recurrent chunk pipeline");
#endif
        impl_->kernels->slice=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"slice_groups"] error:&error];
        impl_->kernels->sliceVector=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"slice_groups_vector4"] error:&error];
        impl_->kernels->prepare=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"linear_prepare"] error:&error];
        impl_->kernels->gatedNorm=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gated_norm"] error:&error];
        impl_->kernels->concat=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"concat_rows"] error:&error];
        impl_->kernels->normRope=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"norm_rope"] error:&error];
        impl_->kernels->kvUpdate=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kv_update"] error:&error];
        impl_->kernels->kvUpdateVector=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kv_update_vector4"] error:&error];
        impl_->kernels->scores=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"attention_scores"] error:&error];
        impl_->kernels->scoresSerial=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"attention_scores_serial"] error:&error];
        impl_->kernels->attentionContext=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"attention_context"] error:&error];
        impl_->kernels->visionNorm=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"vision_norm"] error:&error];
        impl_->kernels->visionScores=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"vision_scores"] error:&error];
#ifdef GARNET_METAL_MPS_VISION_SCORES
        impl_->kernels->visionPackQK=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"vision_pack_qk"] error:&error];
        impl_->kernels->visionSoftmax=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"vision_softmax"] error:&error];
        if(!impl_->kernels->visionPackQK||!impl_->kernels->visionSoftmax)throw std::runtime_error("Cannot create MPS vision score pipelines");
#endif
#ifdef GARNET_METAL_FUSED_VISION_CONTEXT
        impl_->kernels->visionFusedContext=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"vision_softmax_context"] error:&error];
        if(!impl_->kernels->visionFusedContext)throw std::runtime_error("Cannot create fused vision context pipeline");
#ifdef GARNET_METAL_PARALLEL_VISION_CONTEXT
        impl_->kernels->visionParallelContext=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"vision_parallel_context"] error:&error];
#endif
#endif
        impl_->kernels->vision=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"vision_op"] error:&error];
        impl_->kernels->kvChunk=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kv_chunk"] error:&error];
        impl_->kernels->kvChunkVector=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kv_chunk_vector4"] error:&error];
        impl_->kernels->attentionChunk=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"attention_chunk"] error:&error];
        impl_->kernels->layout=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"chunk_layout"] error:&error];
        if(!impl_->kernels->argmax||!impl_->kernels->bf16Decode||!impl_->kernels->bf16Gemm||!impl_->kernels->rms||!impl_->kernels->silu||!impl_->kernels->add||!impl_->kernels->mask||!impl_->kernels->conv||!impl_->kernels->delta||!impl_->kernels->slice||
            !impl_->kernels->prepare||!impl_->kernels->gatedNorm||!impl_->kernels->concat||!impl_->kernels->normRope||!impl_->kernels->kvUpdate||!impl_->kernels->kvUpdateVector||!impl_->kernels->scores||!impl_->kernels->scoresSerial||!impl_->kernels->attentionContext||!impl_->kernels->vision||!impl_->kernels->visionNorm||!impl_->kernels->visionScores||!impl_->kernels->layout||!impl_->kernels->kvChunk||!impl_->kernels->attentionChunk)
            throw std::runtime_error("Cannot create Qwen Metal pipelines");
        cached=impl_->kernels;
    }
}
MetalContext::~MetalContext() { Abort(); }
std::string MetalContext::DeviceName() const {
    return std::string([impl_->device.name UTF8String]);
}
std::string MetalContext::AvailableDeviceName() {
    @autoreleasepool {
        auto device=MTLCreateSystemDefaultDevice();
        if(!device||!MPSSupportsMTLDevice(device))throw std::runtime_error("No MPS-capable Metal GPU available");
        return std::string(device.name.UTF8String);
    }
}
MetalMatrix MetalContext::Allocate(size_t rows,size_t columns) {
    @autoreleasepool {
    Elements(rows,columns);
    const auto stride=[MPSMatrixDescriptor rowBytesFromColumns:columns dataType:MPSDataTypeFloat32];
    if(stride>std::numeric_limits<size_t>::max()/rows)
        throw std::overflow_error("Metal matrix storage overflow");
    if(rows*stride>impl_->device.maxBufferLength)throw std::runtime_error("Metal matrix exceeds device buffer limit");
    id<MTLBuffer> buffer=[impl_->device newBufferWithLength:rows*stride options:MTLResourceStorageModeShared];
    if(!buffer)throw std::runtime_error("Metal matrix allocation failed");
    auto descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns
        rowBytes:stride dataType:MPSDataTypeFloat32];
    auto storage=std::make_shared<MetalMatrix::Impl>();
    storage->matrix=[[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
    return MetalMatrix(storage);
    }
}
MetalBF16Weight MetalContext::UploadBF16(const void* data,size_t bytes,size_t rows,size_t columns,bool floatPrefill,bool visionPrefill) {
    @autoreleasepool {
#ifndef GARNET_METAL_VISION_GLOBAL_ALPHA
        if(visionPrefill)throw std::logic_error("Vision global alpha experiment is disabled");
#endif
        if(visionPrefill&&floatPrefill)throw std::invalid_argument("Vision common alpha cannot use the decoder split-key route");
        if(!data||bytes!=Elements(rows,columns)*sizeof(uint16_t))throw std::invalid_argument("BF16 weight size mismatch");
        if(bytes>impl_->device.maxBufferLength)throw std::invalid_argument("BF16 weight exceeds device buffer limit");
        // Inspect raw bits without introducing alignment requirements on mapped files.
        for(size_t i=0;i<rows*columns;++i){uint16_t bits;std::memcpy(&bits,static_cast<const char*>(data)+2*i,2);
            if((bits&0x7f80)==0x7f80)throw std::invalid_argument("Nonfinite BF16 checkpoint weight");}
        auto value=std::make_shared<MetalBF16Weight::Impl>();value->rows=rows;value->columns=columns;value->floatPrefill=floatPrefill;
        value->buffer=[impl_->device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if(!value->buffer)throw std::runtime_error("BF16 weight allocation failed");
#ifdef GARNET_METAL_EXACT_HALF_PREFILL
        // Preserve original BF16 decode storage. Pack only modest body matrices;
        // a single nonexact value disables the optional representation entirely.
        if(!floatPrefill&&bytes<=16*1024*1024&&impl_->kernels->halfRescale) {
            const auto stride=[MPSMatrixDescriptor rowBytesFromColumns:columns dataType:MPSDataTypeFloat16];
            if(stride<=impl_->device.maxBufferLength/rows) {
                id<MTLBuffer> encoded=[impl_->device newBufferWithLength:rows*stride options:MTLResourceStorageModeShared];
                if(!encoded)throw std::runtime_error("Exact half weight allocation failed");
                auto* encodedBytes=static_cast<char*>(encoded.contents);
                std::memset(encodedBytes,0,rows*stride);
                std::vector<int32_t> scales(rows);bool exact=true,common=false;
#ifdef GARNET_METAL_GLOBAL_HALF_ALPHA
                uint16_t largest=0;
                for(size_t i=0;i<rows*columns;++i){uint16_t bits;std::memcpy(&bits,static_cast<const char*>(data)+2*i,2);largest=std::max(largest,uint16_t(bits&0x7fff));}
                int32_t commonScale=0;common=ExactHalfRowScale(largest,commonScale);
                for(size_t row=0;row<rows&&common;++row)for(size_t col=0;col<columns;++col) {
                    uint16_t bits,half;std::memcpy(&bits,static_cast<const char*>(data)+2*(row*columns+col),2);
                    if(!EncodeExactScaledHalf(bits,commonScale,half)){common=false;break;}
                    std::memcpy(encodedBytes+row*stride+col*2,&half,2);
                }
                if(common){std::fill(scales.begin(),scales.end(),commonScale);value->globalHalf=true;value->globalHalfScale=commonScale;value->visionPrefill=visionPrefill;}
#endif
                for(size_t row=0;row<rows&&exact&&!common;++row) {
                    uint16_t largest=0;
                    for(size_t col=0;col<columns;++col) {
                        uint16_t bits;std::memcpy(&bits,static_cast<const char*>(data)+2*(row*columns+col),2);
                        largest=std::max(largest,uint16_t(bits&0x7fff));
                    }
                    int32_t scale=0;
                    if(!ExactHalfRowScale(largest,scale)){exact=false;break;}
                    scales[row]=scale;
                    for(size_t col=0;col<columns;++col) {
                        uint16_t bits;std::memcpy(&bits,static_cast<const char*>(data)+2*(row*columns+col),2);
                        uint16_t half;
                        if(!EncodeExactScaledHalf(bits,scale,half)){exact=false;break;}
                        std::memcpy(encodedBytes+row*stride+col*2,&half,2);
                    }
                }
                if(exact) {
                    auto descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns rowBytes:stride dataType:MPSDataTypeFloat16];
                    value->exactHalf=[[MPSMatrix alloc] initWithBuffer:encoded descriptor:descriptor];
                    value->halfScales=[impl_->device newBufferWithBytes:scales.data() length:rows*sizeof(int32_t) options:MTLResourceStorageModeShared];
                    if(!value->exactHalf||!value->halfScales)throw std::runtime_error("Exact half metadata allocation failed");
                }
            }
        }
#endif
        return MetalBF16Weight(value);
    }
}
void MetalContext::EncodeArgmax(const MetalMatrix& input,const MetalMatrix& output) {
    @autoreleasepool {
        if(!impl_->command||!input.impl_||!output.impl_)throw std::invalid_argument("Missing GPU argmax input/batch");
        auto a=input.impl_->matrix,o=output.impl_->matrix;
        if(a.device!=impl_->device||o.device!=impl_->device||o.rows!=a.rows||o.columns!=2||a.columns>16777216||a.data==o.data)
            throw std::invalid_argument("GPU argmax shape/device mismatch");
        if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_.get())!=0))
            throw std::logic_error("Uninitialized GPU argmax input");
        if(output.impl_->pending&&impl_->pendingOutputs.count(output.impl_.get())==0)
            throw std::logic_error("GPU argmax output belongs to another command batch");
        auto pipeline=impl_->kernels->argmax;
        if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128)
            throw std::runtime_error("GPU argmax requires 32-lane SIMD groups");
        struct Params{uint64_t rows,columns,iStride,oStride;};Params p{a.rows,a.columns,a.rowBytes/4,o.rowBytes/4};
        auto encoder=[impl_->command computeCommandEncoder];
        if(!encoder)throw std::runtime_error("Cannot create GPU argmax encoder");
        [encoder setComputePipelineState:pipeline];[encoder setBuffer:a.data offset:0 atIndex:0];[encoder setBuffer:o.data offset:0 atIndex:1];
        [encoder setBytes:&p length:sizeof(p) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(a.rows,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [encoder endEncoding];output.impl_->ready=false;output.impl_->pending=true;if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
    }
}
#include "screened_head_native_impl.h"
size_t MetalContext::LastFusedHeadArgmaxDispatches() const {return impl_->fusedHeadArgmaxDispatches;}
bool MetalContext::EncodeBF16Argmax(const MetalMatrix& input,const MetalBF16Weight& weight,
    const MetalMatrix& output,const MetalMatrix* debugLogits) {
#ifndef GARNET_METAL_FUSED_HEAD_ARGMAX
    return false;
#else
    @autoreleasepool {
        if(!input.impl_||!weight.impl_||!output.impl_)return false;
        auto a=input.impl_->matrix,o=output.impl_->matrix;auto w=weight.impl_;
        auto projection=impl_->kernels->fusedHead,reduction=impl_->kernels->fusedHeadReduce;
        if(a.rows>2||w->rows>16777216||!projection||!reduction||
            projection.threadExecutionWidth!=32||projection.maxTotalThreadsPerThreadgroup<128||
            projection.staticThreadgroupMemoryLength>impl_->device.maxThreadgroupMemoryLength||
            reduction.threadExecutionWidth!=32||reduction.maxTotalThreadsPerThreadgroup<128||
            reduction.staticThreadgroupMemoryLength>impl_->device.maxThreadgroupMemoryLength)return false;
        if(!impl_->command)throw std::logic_error("Begin a Metal command batch first");
        if(a.device!=impl_->device||o.device!=impl_->device||w->buffer.device!=impl_->device||
            a.columns!=w->columns||o.rows!=a.rows||o.columns!=2||a.data==o.data||w->buffer==o.data)
            throw std::invalid_argument("Fused greedy head shape/device mismatch");
        if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_.get())))
            throw std::logic_error("Uninitialized fused greedy head input");
        if(output.impl_->pending&&!impl_->pendingOutputs.count(output.impl_.get()))
            throw std::logic_error("Fused prediction belongs to another command batch");
        if(debugLogits) {
            if(!debugLogits->impl_)throw std::invalid_argument("Empty fused debug logits");
            auto debug=debugLogits->impl_->matrix;
            if(debug.device!=impl_->device||debug.rows!=a.rows||debug.columns!=w->rows||
                debug.data==a.data||debug.data==o.data||debug.data==w->buffer)
                throw std::invalid_argument("Fused debug logits shape/device mismatch");
            if(debugLogits->impl_->pending&&!impl_->pendingOutputs.count(debugLogits->impl_.get()))
                throw std::logic_error("Fused debug output belongs to another command batch");
        }
        const size_t groups=(w->rows+15)/16;
        const std::array<size_t,2> shape{a.rows,groups};auto scratch=impl_->fusedHeadScratch.find(shape);
        if(scratch==impl_->fusedHeadScratch.end())scratch=impl_->fusedHeadScratch.emplace(shape,Allocate(a.rows,groups*3)).first;
        auto parts=scratch->second.impl_->matrix;
        struct ProjectionParams{uint64_t rows,inner,columns,aStride,oStride,partialStride;};
        ProjectionParams p{a.rows,a.columns,w->rows,a.rowBytes/4,debugLogits?debugLogits->impl_->matrix.rowBytes/4:0,parts.rowBytes/4};
        const uint32_t debugEnabled=debugLogits?1:0;
        auto encoder=[impl_->command computeCommandEncoder];
        if(!encoder)throw std::runtime_error("Cannot create fused projection encoder");
        [encoder setComputePipelineState:projection];[encoder setBuffer:a.data offset:0 atIndex:0];
        [encoder setBuffer:w->buffer offset:0 atIndex:1];[encoder setBuffer:parts.data offset:0 atIndex:2];
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        [encoder setBuffer:debugLogits?debugLogits->impl_->matrix.data:o.data offset:0 atIndex:4];
        [encoder setBytes:&debugEnabled length:sizeof(debugEnabled) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
        struct ReduceParams{uint64_t rows,columns,iStride,oStride;};ReduceParams rp{a.rows,groups,parts.rowBytes/4,o.rowBytes/4};
        encoder=[impl_->command computeCommandEncoder];
        if(!encoder)throw std::runtime_error("Cannot create fused argmax reduction encoder");
        [encoder setComputePipelineState:reduction];[encoder setBuffer:parts.data offset:0 atIndex:0];
        [encoder setBuffer:o.data offset:0 atIndex:1];[encoder setBytes:&rp length:sizeof(rp) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(a.rows,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
        for(const auto& target:{scratch->second.impl_,output.impl_,debugLogits?debugLogits->impl_:std::shared_ptr<MetalMatrix::Impl>()}) {
            if(!target)continue;target->ready=false;target->pending=true;
            if(impl_->pendingOutputs.insert(target.get()).second)impl_->outputs.push_back(target);
        }
        ++impl_->fusedHeadArgmaxDispatches;return true;
    }
#endif
}
bool MetalContext::EncodeJoinedBF16Prefill(const MetalMatrix& input,const MetalBF16Weight& first,
    const MetalBF16Weight& second,const MetalMatrix& firstOutput,const MetalMatrix& secondOutput) {
#ifndef GARNET_METAL_JOINED_PREFILL
    return false;
#else
    @autoreleasepool {
        if(!input.impl_||!first.impl_||!second.impl_||!firstOutput.impl_||!secondOutput.impl_)return false;
        auto a=input.impl_->matrix,x=firstOutput.impl_->matrix,y=secondOutput.impl_->matrix;
        auto w0=first.impl_,w1=second.impl_;
        const bool selected=(a.rows==44&&a.columns==1024&&w0->rows==3072)||
            (a.rows==26&&a.columns==896&&w0->rows==4864);
        if(!selected||w0->rows!=w1->rows||w0->columns!=w1->columns||w0->floatPrefill||w1->floatPrefill||
            !w0->exactHalf||!w1->exactHalf||!w0->halfScales||!w1->halfScales)return false;
        if(!impl_->command)throw std::logic_error("Begin a Metal command batch first");
        if(a.device!=impl_->device||x.device!=impl_->device||y.device!=impl_->device||
            w0->buffer.device!=impl_->device||w1->buffer.device!=impl_->device||
            a.columns!=w0->columns||x.rows!=a.rows||y.rows!=a.rows||x.columns!=w0->rows||y.columns!=w1->rows)
            throw std::invalid_argument("Joined prefill shape/device mismatch");
        if(x.data==y.data||x.data==a.data||y.data==a.data||x.rowBytes!=y.rowBytes)return false;
        auto usable=[&](const MetalMatrix& m){return m.impl_->ready||(m.impl_->pending&&impl_->pendingOutputs.count(m.impl_.get())!=0);};
        if(!usable(input))throw std::logic_error("Uninitialized joined prefill input");
        for(const auto* output:{&firstOutput,&secondOutput})if(output->impl_->pending&&impl_->pendingOutputs.count(output->impl_.get())==0)
            throw std::logic_error("Joined prefill output belongs to another batch");
        const auto identity=std::make_pair(static_cast<const void*>(w0.get()),static_cast<const void*>(w1.get()));
        auto found=impl_->joinedWeights.find(identity);
        if(found==impl_->joinedWeights.end()) {
            const size_t rowBytes=[MPSMatrixDescriptor rowBytesFromColumns:a.columns dataType:MPSDataTypeFloat16];
            auto descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:x.columns*2 columns:a.columns rowBytes:rowBytes dataType:MPSDataTypeFloat16];
            Impl::JoinedWeight pack;pack.firstOwner=first;pack.secondOwner=second;
            pack.data=[impl_->device newBufferWithLength:x.columns*2*rowBytes options:MTLResourceStorageModeShared];
            pack.scales=[impl_->device newBufferWithLength:x.columns*8 options:MTLResourceStorageModeShared];
            if(!pack.data||!pack.scales)throw std::runtime_error("Joined prefill allocation failed");
            std::memset(pack.data.contents,0,x.columns*2*rowBytes);
            for(size_t index=0;index<2;++index) {
                auto weight=index?w1:w0;
                for(size_t row=0;row<x.columns;++row)std::memcpy(static_cast<char*>(pack.data.contents)+(index*x.columns+row)*rowBytes,
                    static_cast<const char*>(weight->exactHalf.data.contents)+weight->exactHalf.offset+row*weight->exactHalf.rowBytes,a.columns*2);
                std::memcpy(static_cast<char*>(pack.scales.contents)+index*x.columns*4,
                    static_cast<const char*>(weight->halfScales.contents)+weight->halfScalesOffset,x.columns*4);
            }
            pack.view=[[MPSMatrix alloc] initWithBuffer:pack.data descriptor:descriptor];
            if(!pack.view)throw std::runtime_error("Joined prefill matrix view failed");
#ifdef GARNET_METAL_COMPACT_JOINED_PREFILL
            // Retain the learned BF16 decode buffers. Individual half projections
            // share the joined allocation, with their original row scales.
            auto singleDescriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:x.columns columns:a.columns rowBytes:rowBytes dataType:MPSDataTypeFloat16];
            auto firstView=[[MPSMatrix alloc] initWithBuffer:pack.data offset:0 descriptor:singleDescriptor];
            auto secondView=[[MPSMatrix alloc] initWithBuffer:pack.data offset:x.columns*rowBytes descriptor:singleDescriptor];
            if(!firstView||!secondView)throw std::runtime_error("Joined prefill subview creation failed");
            w0->exactHalf=firstView;w1->exactHalf=secondView;
            w0->halfScales=pack.scales;w0->halfScalesOffset=0;
            w1->halfScales=pack.scales;w1->halfScalesOffset=x.columns*4;
#endif
            found=impl_->joinedWeights.emplace(identity,std::move(pack)).first;
        }
        const std::array<size_t,3> key{a.rows,x.columns*2,a.columns};
        auto product=impl_->halfMatmul.find(key);
        if(product==impl_->halfMatmul.end())product=impl_->halfMatmul.emplace(key,[[MPSMatrixMultiplication alloc] initWithDevice:impl_->device transposeLeft:NO transposeRight:YES resultRows:a.rows resultColumns:x.columns*2 interiorColumns:a.columns alpha:1 beta:0]).first;
        if(!product->second)throw std::runtime_error("Joined prefill product creation failed");
        const std::array<size_t,2> shape{a.rows,x.columns*2};auto scratch=impl_->halfScratch.find(shape);
        if(scratch==impl_->halfScratch.end())scratch=impl_->halfScratch.emplace(shape,Allocate(a.rows,x.columns*2)).first;
        auto temporary=scratch->second.impl_->matrix;
        [product->second encodeToCommandBuffer:impl_->command leftMatrix:a rightMatrix:found->second.view resultMatrix:temporary];
        struct Params{uint64_t rows,columns,iStride,oStride;};Params p{a.rows,x.columns,temporary.rowBytes/4,x.rowBytes/4};
        auto encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Joined prefill encoder unavailable");
        [encoder setComputePipelineState:impl_->kernels->joinedSplitRescale];[encoder setBuffer:temporary.data offset:0 atIndex:0];
        [encoder setBuffer:found->second.scales offset:0 atIndex:1];[encoder setBuffer:x.data offset:0 atIndex:2];[encoder setBuffer:y.data offset:0 atIndex:3];[encoder setBytes:&p length:sizeof(p) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(a.rows*x.columns*2,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
        for(const auto* output:{&firstOutput,&secondOutput}){output->impl_->ready=false;output->impl_->pending=true;if(impl_->pendingOutputs.insert(output->impl_.get()).second)impl_->outputs.push_back(output->impl_);}
        ++impl_->joinedPrefillPairs;return true;
    }
#endif
}
bool MetalContext::EncodeJoinedGlobalMlp(const MetalMatrix& input,const MetalBF16Weight& gate,
    const MetalBF16Weight& up,const MetalMatrix& gateOutput,const MetalMatrix& upOutput,const MetalMatrix& activated) {
#ifndef GARNET_METAL_JOINED_GLOBAL_MLP
    return false;
#else
    @autoreleasepool {
        if(!input.impl_||!gate.impl_||!up.impl_||!gateOutput.impl_||!upOutput.impl_||!activated.impl_)return false;
        auto a=input.impl_->matrix,g=gateOutput.impl_->matrix,u=upOutput.impl_->matrix,o=activated.impl_->matrix;
        auto w0=gate.impl_,w1=up.impl_;
        const bool selected=(a.columns==1024&&w0->rows==3072&&(a.rows==24||a.rows==44))||
            (a.columns==896&&w0->rows==4864&&a.rows==26)||
            (a.columns==1024&&w0->rows==3584&&a.rows==32);
        if(!selected||w0==w1||!gate.GlobalHalfReady()||!up.GlobalHalfReady()||
            w0->floatPrefill||w1->floatPrefill||w0->rows!=w1->rows||w0->columns!=w1->columns||
            w0->globalHalfScale!=w1->globalHalfScale)return false;
        if(!impl_->command)throw std::logic_error("Begin a Metal command batch first");
        if(a.device!=impl_->device||w0->buffer.device!=impl_->device||w1->buffer.device!=impl_->device||
            a.columns!=w0->columns)throw std::invalid_argument("Joined global input/weight device or shape mismatch");
        for(auto m:{g,u,o})if(m.device!=impl_->device||m.rows!=a.rows||m.columns!=w0->rows)
            throw std::invalid_argument("Joined global output device or shape mismatch");
        // Existing custom kernels use zero buffer offsets. Keep every public
        // output allocation independent and materialize all three results.
        if(a.offset||g.offset||u.offset||o.offset||g.data==u.data||g.data==o.data||u.data==o.data||
            a.data==g.data||a.data==u.data||a.data==o.data)return false;
        if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_.get())))
            throw std::logic_error("Uninitialized joined global input");
        for(const auto* out:{&gateOutput,&upOutput,&activated})
            if(out->impl_->pending&&!impl_->pendingOutputs.count(out->impl_.get()))
                throw std::logic_error("Joined global output belongs to another command batch");
        const auto identity=std::make_pair(static_cast<const void*>(w0.get()),static_cast<const void*>(w1.get()));
        auto found=impl_->joinedGlobalWeights.find(identity);
        if(found==impl_->joinedGlobalWeights.end()) {
            const size_t rowBytes=[MPSMatrixDescriptor rowBytesFromColumns:a.columns dataType:MPSDataTypeFloat16];
            auto descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:w0->rows*2 columns:a.columns rowBytes:rowBytes dataType:MPSDataTypeFloat16];
            auto buffer=[impl_->device newBufferWithLength:w0->rows*2*rowBytes options:MTLResourceStorageModeShared];
            if(!buffer)throw std::runtime_error("Joined global weight allocation failed");
            std::memset(buffer.contents,0,w0->rows*2*rowBytes);
            for(size_t index=0;index<2;++index) {
                auto weight=index?w1:w0;
                for(size_t row=0;row<w0->rows;++row)std::memcpy(static_cast<char*>(buffer.contents)+(index*w0->rows+row)*rowBytes,
                    static_cast<const char*>(weight->exactHalf.data.contents)+weight->exactHalf.offset+row*weight->exactHalf.rowBytes,a.columns*2);
            }
            auto view=[[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
            auto single=[MPSMatrixDescriptor matrixDescriptorWithRows:w0->rows columns:a.columns rowBytes:rowBytes dataType:MPSDataTypeFloat16];
            auto firstView=[[MPSMatrix alloc] initWithBuffer:buffer offset:0 descriptor:single];
            auto secondView=[[MPSMatrix alloc] initWithBuffer:buffer offset:w0->rows*rowBytes descriptor:single];
            if(!view||!firstView||!secondView)throw std::runtime_error("Joined global weight views failed");
            // MPS honors half-weight offsets. Raw BF16 decode storage and each
            // exact common scale stay unchanged; no duplicate half allocation.
            w0->exactHalf=firstView;w1->exactHalf=secondView;
            Impl::JoinedGlobalWeight pack;pack.firstOwner=gate;pack.secondOwner=up;pack.view=view;
            found=impl_->joinedGlobalWeights.emplace(identity,std::move(pack)).first;
        }
        const std::array<size_t,4> key{a.rows,w0->rows*2,a.columns,size_t(w0->globalHalfScale+64)};
        auto product=impl_->globalHalfMatmul.find(key);
        if(product==impl_->globalHalfMatmul.end())product=impl_->globalHalfMatmul.emplace(key,
            [[MPSMatrixMultiplication alloc] initWithDevice:impl_->device transposeLeft:NO transposeRight:YES
                resultRows:a.rows resultColumns:w0->rows*2 interiorColumns:a.columns alpha:std::ldexp(1.0,-w0->globalHalfScale) beta:0]).first;
        if(!product->second)throw std::runtime_error("Joined global MPS product creation failed");
        const std::array<size_t,2> shape{a.rows,w0->rows*2};
        auto scratch=impl_->halfScratch.find(shape);
        if(scratch==impl_->halfScratch.end())scratch=impl_->halfScratch.emplace(shape,Allocate(a.rows,w0->rows*2)).first;
        auto temporary=scratch->second.impl_->matrix;
        [product->second encodeToCommandBuffer:impl_->command leftMatrix:a rightMatrix:found->second.view resultMatrix:temporary];
        struct Params{uint64_t rows,columns,iStride,gStride,uStride,oStride;};
        Params p{a.rows,w0->rows,temporary.rowBytes/4,g.rowBytes/4,u.rowBytes/4,o.rowBytes/4};
        auto encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Joined global split-SiLU encoder unavailable");
        [encoder setComputePipelineState:impl_->kernels->joinedGlobalSilu];
        [encoder setBuffer:temporary.data offset:0 atIndex:0];[encoder setBuffer:g.data offset:0 atIndex:1];
        [encoder setBuffer:u.data offset:0 atIndex:2];[encoder setBuffer:o.data offset:0 atIndex:3];
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        [encoder dispatchThreads:MTLSizeMake(a.rows*w0->rows,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
        for(const auto* out:{&gateOutput,&upOutput,&activated}) {
            out->impl_->ready=false;out->impl_->pending=true;
            if(impl_->pendingOutputs.insert(out->impl_.get()).second)impl_->outputs.push_back(out->impl_);
        }
        ++impl_->joinedGlobalMlpBlocks;++impl_->globalHalfAlphaDispatches;return true;
    }
#endif
}
size_t MetalContext::LastJoinedGlobalMlpBlocks() const {return impl_->joinedGlobalMlpBlocks;}
size_t MetalContext::LastJoinedPrefillPairs() const {return impl_->joinedPrefillPairs;}
size_t MetalContext::LastConstantHeadDispatches() const {return impl_->constantHeadDispatches;}
size_t MetalContext::LastSingleCoderHeadDispatches() const {return impl_->singleCoderHeadDispatches;}
size_t MetalContext::LastResidualNormDispatches() const {return impl_->residualNormDispatches;}
size_t MetalContext::LastPackedPrefillNormDispatches() const {return impl_->packedPrefillNormDispatches;}
size_t MetalContext::LastGlobalHalfAlphaDispatches() const {return impl_->globalHalfAlphaDispatches;}
size_t MetalContext::LastWideBatchPrefillDispatches() const {return impl_->wideBatchPrefillDispatches;}
size_t MetalContext::LastVisionGlobalAlphaDispatches() const {return impl_->visionGlobalAlphaDispatches;}
size_t MetalContext::LastVisionParallelDispatches() const {return impl_->visionParallelDispatches;}
size_t MetalContext::LastVisionPrefetchDispatches() const {return impl_->visionPrefetchDispatches;}
bool MetalContext::PrepareBF16BodyShape(const MetalMatrix& input,const MetalBF16Weight& weight,const MetalMatrix& output) {
#ifdef GARNET_METAL_BODY_SHAPE
    @autoreleasepool {
        if(impl_->command)throw std::logic_error("Body shape preparation requires a closed command batch");
        if(!input.impl_||!output.impl_||!weight.impl_)return false;
        auto a=input.impl_->matrix,o=output.impl_->matrix;auto w=weight.impl_;
        if(a.device!=impl_->device||o.device!=impl_->device||w->buffer.device!=impl_->device||a.rows>2||o.rows!=a.rows||
            a.columns!=w->columns||o.columns!=w->rows||o.columns>=100000)return false;
        const std::array<size_t,5> key{a.rows,a.columns,o.columns,a.rowBytes/4,o.rowBytes/4};
        for(auto value:key)if(value>UINT32_MAX)return false;
        std::unique_lock<std::shared_mutex> lock(impl_->kernels->bodyShapeMutex);
        auto& cache=impl_->kernels->bodyShape;
        auto found=cache.find(key);if(found!=cache.end())return found->second!=nil;
        if(cache.size()>=128)return false;
        bool small=false;
#ifdef GARNET_METAL_SMALL_OUTPUT_DECODE
        small=o.columns<=256;
#endif
        uint32_t dimensions[5];for(size_t i=0;i<5;++i)dimensions[i]=uint32_t(key[i]);
        auto constants=[MTLFunctionConstantValues new];
        for(NSUInteger i=0;i<5;++i)[constants setConstantValue:&dimensions[i] type:MTLDataTypeUInt atIndex:i];
        bool noTail=o.columns%(small?4:16)==0;
        [constants setConstantValue:&noTail type:MTLDataTypeBool atIndex:5];
        NSError* error=nil;
        auto function=[impl_->kernels->library newFunctionWithName:small?@"specialized_bf16_decode_small":@"specialized_bf16_decode" constantValues:constants error:&error];
        id<MTLComputePipelineState> pipeline=nil;
        if(function)pipeline=[impl_->device newComputePipelineStateWithFunction:function error:&error];
        if(pipeline&&(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<(small?64:128)))pipeline=nil;
        cache.emplace(key,pipeline);return pipeline!=nil;
    }
#else
    (void)input;(void)weight;(void)output;return false;
#endif
}
size_t MetalContext::BodyShapePipelineCount() const {
#ifdef GARNET_METAL_BODY_SHAPE
    std::shared_lock<std::shared_mutex> lock(impl_->kernels->bodyShapeMutex);return impl_->kernels->bodyShape.size();
#else
    return 0;
#endif
}
size_t MetalContext::LastBodyShapeDispatches() const {return impl_->bodyShapeDispatches;}
size_t MetalContext::LastBodyShapeMisses() const {return impl_->bodyShapeMisses;}
void MetalContext::EncodeBF16MatVec(const MetalMatrix& input,const MetalBF16Weight& weight,const MetalMatrix& output) {
    @autoreleasepool {
        if(!impl_->command||!input.impl_||!weight.impl_||!output.impl_)throw std::invalid_argument("Missing BF16 projection batch/input");
        auto a=input.impl_->matrix,o=output.impl_->matrix;auto w=weight.impl_;
        bool supportedDecoderRows=a.rows<=128;
#ifdef GARNET_METAL_QWEN35_WIDE_BATCH_PREFILL
        supportedDecoderRows=supportedDecoderRows||a.rows==160;
#endif
        if(a.device!=impl_->device||o.device!=impl_->device||w->buffer.device!=impl_->device||
            a.columns!=w->columns||o.rows!=a.rows||o.columns!=w->rows||
            (w->visionPrefill?a.rows>32768:!supportedDecoderRows)||o.data==a.data)
            throw std::invalid_argument("BF16 projection shape/device mismatch or unsupported row count");
        auto usable=[&](const MetalMatrix& m){return m.impl_->ready||(m.impl_->pending&&
            impl_->pendingOutputs.count(m.impl_.get())!=0);};
        if(!usable(input))throw std::logic_error("Uninitialized BF16 projection input");
        if(output.impl_->pending&&impl_->pendingOutputs.count(output.impl_.get())==0)
            throw std::logic_error("BF16 output belongs to another command batch");
        if(!w->visionPrefill&&a.rows==160)++impl_->wideBatchPrefillDispatches;
#ifdef GARNET_METAL_FLOAT_DENSE_KEY_PREFILL
        if(w->floatPrefill&&a.rows>2&&a.columns<=1024) {
            const size_t parts=(a.columns+63)/64;
            const std::array<size_t,3> key{a.rows,o.columns,parts};auto scratch=impl_->splitScratch.find(key);
            if(scratch==impl_->splitScratch.end())scratch=impl_->splitScratch.emplace(key,Allocate(a.rows*parts,o.columns)).first;
            auto partial=scratch->second.impl_->matrix;
            struct Params{uint64_t rows,inner,columns,aStride,oStride,parts;};
            Params p{a.rows,a.columns,o.columns,a.rowBytes/4,partial.rowBytes/4,parts};
            auto encoder=[impl_->command computeCommandEncoder];
            if(!encoder)throw std::runtime_error("Cannot create split key projection encoder");
            [encoder setComputePipelineState:impl_->kernels->bf16Split];[encoder setBuffer:a.data offset:0 atIndex:0];
            [encoder setBuffer:w->buffer offset:0 atIndex:1];[encoder setBuffer:partial.data offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake((o.columns+31)/32,(a.rows+15)/16,parts) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
            struct Reduce{uint64_t rows,columns,iStride,oStride,parts;};Reduce r{a.rows,o.columns,partial.rowBytes/4,o.rowBytes/4,parts};
            encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Cannot create split key reduction encoder");
            [encoder setComputePipelineState:impl_->kernels->bf16SplitReduce];[encoder setBuffer:partial.data offset:0 atIndex:0];[encoder setBuffer:o.data offset:0 atIndex:1];
            [encoder setBytes:&r length:sizeof(r) atIndex:2];
            const auto total=Elements(o.rows,o.columns);[encoder dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(total,size_t(impl_->kernels->bf16SplitReduce.maxTotalThreadsPerThreadgroup)),1,1)];[encoder endEncoding];
            output.impl_->ready=false;output.impl_->pending=true;if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
            return;
        }
#endif
#ifdef GARNET_METAL_EXACT_HALF_PREFILL
        if((a.rows>2||w->visionPrefill)&&w->exactHalf&&w->halfScales) {
#ifdef GARNET_METAL_GLOBAL_HALF_ALPHA
            if(w->globalHalf) {
                const std::array<size_t,4> key{a.rows,o.columns,a.columns,size_t(w->globalHalfScale+64)};
                auto found=impl_->globalHalfMatmul.find(key);
                if(found==impl_->globalHalfMatmul.end()) {
                    auto product=[[MPSMatrixMultiplication alloc] initWithDevice:impl_->device transposeLeft:NO transposeRight:YES
                        resultRows:a.rows resultColumns:o.columns interiorColumns:a.columns alpha:std::ldexp(1.0,-w->globalHalfScale) beta:0];
                    if(!product)throw std::runtime_error("Cannot create global half-alpha MPS product");
                    found=impl_->globalHalfMatmul.emplace(key,product).first;
                }
                [found->second encodeToCommandBuffer:impl_->command leftMatrix:a rightMatrix:w->exactHalf resultMatrix:o];
                output.impl_->ready=false;output.impl_->pending=true;
                if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
                ++impl_->globalHalfAlphaDispatches;if(w->visionPrefill)++impl_->visionGlobalAlphaDispatches;return;
            }
#endif
            const std::array<size_t,3> key{a.rows,o.columns,a.columns};
            auto found=impl_->halfMatmul.find(key);
            if(found==impl_->halfMatmul.end()) {
                auto product=[[MPSMatrixMultiplication alloc] initWithDevice:impl_->device transposeLeft:NO
                    transposeRight:YES resultRows:a.rows resultColumns:o.columns interiorColumns:a.columns alpha:1 beta:0];
                found=impl_->halfMatmul.emplace(key,product).first;
            }
            const std::array<size_t,2> shape{a.rows,o.columns};auto scratch=impl_->halfScratch.find(shape);
            if(scratch==impl_->halfScratch.end())scratch=impl_->halfScratch.emplace(shape,Allocate(a.rows,o.columns)).first;
            auto temporary=scratch->second.impl_->matrix;
            [found->second encodeToCommandBuffer:impl_->command leftMatrix:a rightMatrix:w->exactHalf resultMatrix:temporary];
            struct Params{uint64_t rows,columns,iStride,oStride;};Params p{a.rows,o.columns,temporary.rowBytes/4,o.rowBytes/4};
            auto encoder=[impl_->command computeCommandEncoder];
            if(!encoder)throw std::runtime_error("Cannot create exact half rescale encoder");
            [encoder setComputePipelineState:impl_->kernels->halfRescale];
            [encoder setBuffer:temporary.data offset:0 atIndex:0];[encoder setBuffer:w->halfScales offset:w->halfScalesOffset atIndex:1];
            [encoder setBuffer:o.data offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];
            const auto total=Elements(o.rows,o.columns);
            [encoder dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(total,size_t(impl_->kernels->halfRescale.maxTotalThreadsPerThreadgroup)),1,1)];
            [encoder endEncoding];output.impl_->ready=false;output.impl_->pending=true;
            if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
            return;
        }
#endif
        // Wide tiles share more input/weight loading at chunk sizes 16/32.
        // Keep the established small-row/small-output path and fall back if
        // a device cannot encode the optional 256-thread pipelines.
        bool wide=false;
        auto widePipeline=a.rows>16?impl_->kernels->bf16Gemm32Wide:impl_->kernels->bf16Gemm16Wide;
        if(a.rows>=16&&o.columns>=64&&widePipeline&&widePipeline.threadExecutionWidth==32&&
            widePipeline.maxTotalThreadsPerThreadgroup>=256&&widePipeline.staticThreadgroupMemoryLength<=impl_->device.maxThreadgroupMemoryLength)wide=true;
        auto pipeline=a.rows<=2?impl_->kernels->bf16Decode:wide?widePipeline:impl_->kernels->bf16Gemm;
        size_t threads=wide?256:128,tileRows=wide&&a.rows>16?32:16,tileColumns=wide?64:32;
        size_t decodeColumns=16;
#ifdef GARNET_METAL_SMALL_OUTPUT_DECODE
        if(a.rows<=2&&o.columns<=256&&impl_->kernels->bf16DecodeSmall.threadExecutionWidth==32&&impl_->kernels->bf16DecodeSmall.maxTotalThreadsPerThreadgroup>=64) {
            pipeline=impl_->kernels->bf16DecodeSmall;threads=64;decodeColumns=4;
        }
#endif
#ifdef GARNET_METAL_NATIVE_BFLOAT_BODY
        // Preserve the small-output and vocabulary-head kernels. Native BF16
        // body weights retain their original bytes, with FP32 arithmetic.
        if(a.rows<=2&&a.columns%4==0&&o.columns>256&&o.columns<100000) {
            pipeline=a.rows==1?impl_->kernels->nativeBFSerial:impl_->kernels->nativeBFDot;
            threads=a.rows==1?64:128;decodeColumns=a.rows==1?8:16;
        }
#endif
#ifdef GARNET_METAL_CONSTANT_BF16_HEAD
        const bool constantHead=a.rows==1&&((a.columns==896&&o.columns==151936)||
            (a.columns==1024&&(o.columns==151936||o.columns==248320)));
        if(constantHead){pipeline=impl_->kernels->constantHead;threads=128;decodeColumns=16;}
#endif
#ifdef GARNET_METAL_BODY_SHAPE
        if(a.rows<=2&&o.columns<100000) {
            const std::array<size_t,5> key{a.rows,a.columns,o.columns,a.rowBytes/4,o.rowBytes/4};
            std::shared_lock<std::shared_mutex> lock(impl_->kernels->bodyShapeMutex);
            auto found=impl_->kernels->bodyShape.find(key);
            if(found!=impl_->kernels->bodyShape.end()&&found->second) {
                pipeline=found->second;++impl_->bodyShapeDispatches;
            } else ++impl_->bodyShapeMisses;
        }
#endif
#ifdef GARNET_METAL_SINGLE_CODER_HEAD
        const bool singleCoderHead=a.rows==1&&a.columns==896&&o.columns==151936;
        if(singleCoderHead){pipeline=impl_->kernels->singleCoderHead;threads=128;decodeColumns=16;}
#endif
        if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<threads)
            throw std::runtime_error("BF16 projection requires 32-lane SIMD groups");
        struct Params{uint64_t rows,inner,columns,aStride,oStride;};
        Params p{a.rows,a.columns,o.columns,a.rowBytes/4,o.rowBytes/4};
        auto encoder=[impl_->command computeCommandEncoder];
        if(!encoder)throw std::runtime_error("Cannot create BF16 projection encoder");
        [encoder setComputePipelineState:pipeline];[encoder setBuffer:a.data offset:0 atIndex:0];
        [encoder setBuffer:w->buffer offset:0 atIndex:1];[encoder setBuffer:o.data offset:0 atIndex:2];
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        [encoder dispatchThreadgroups:a.rows<=2?MTLSizeMake((o.columns+decodeColumns-1)/decodeColumns,1,1):MTLSizeMake((o.columns+tileColumns-1)/tileColumns,(o.rows+tileRows-1)/tileRows,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
#ifdef GARNET_METAL_CONSTANT_BF16_HEAD
        if(constantHead)++impl_->constantHeadDispatches;
#endif
#ifdef GARNET_METAL_SINGLE_CODER_HEAD
        if(singleCoderHead)++impl_->singleCoderHeadDispatches;
#endif
        [encoder endEncoding];output.impl_->ready=false;output.impl_->pending=true;if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
    }
}
MetalMatrix MetalContext::Upload(const std::vector<float>& values,size_t rows,size_t columns) {
    if(values.size()!=Elements(rows,columns))throw std::invalid_argument("Metal matrix input size mismatch");
    auto result=Allocate(rows,columns);
    Update(result,values);
    return result;
}
void MetalContext::Update(const MetalMatrix& matrix,const std::vector<float>& values) {
    Update(matrix,values.data(),values.size());
}
void MetalContext::Update(const MetalMatrix& matrix,const float* values,size_t count) {
    if(impl_->command)throw std::logic_error("Host update requires a closed command batch");
    if(!matrix.impl_||matrix.impl_->matrix.device!=impl_->device)
        throw std::invalid_argument("Metal update matrix belongs to another device or is empty");
#ifdef GARNET_METAL_DEFERRED_STATE_ZERO
    if(matrix.impl_->pending&&!impl_->zeroQueued.count(matrix.impl_.get()))throw std::logic_error("Cannot update pending Metal output");
#else
    if(matrix.impl_->pending)throw std::logic_error("Cannot update pending Metal output");
#endif
    auto native=matrix.impl_->matrix;
    if(!values||count!=Elements(native.rows,native.columns))throw std::invalid_argument("Metal update size mismatch");
#ifdef GARNET_METAL_DEFERRED_STATE_ZERO
    if(impl_->zeroQueued.erase(matrix.impl_.get()))matrix.impl_->pending=false;
#endif
    for(size_t i=0;i<native.rows;++i)
        std::memcpy(static_cast<char*>(native.data.contents)+i*native.rowBytes,
            values+i*native.columns,native.columns*sizeof(float));
    matrix.impl_->ready=true;
}
void MetalContext::Zero(const MetalMatrix& matrix) {
    if(impl_->command)throw std::logic_error("Host zero requires a closed command batch");
    if(!matrix.impl_||matrix.impl_->matrix.device!=impl_->device)
        throw std::invalid_argument("Metal zero matrix belongs to another device or is empty");
#ifdef GARNET_METAL_DEFERRED_STATE_ZERO
    if(matrix.impl_->pending&&!impl_->zeroQueued.count(matrix.impl_.get()))throw std::logic_error("Cannot zero pending Metal output");
    if(impl_->zeroQueued.erase(matrix.impl_.get()))matrix.impl_->pending=false;
#else
    if(matrix.impl_->pending)throw std::logic_error("Cannot zero pending Metal output");
#endif
    auto native=matrix.impl_->matrix;
    // Match Update: change logical columns only, preserving row padding.
    for(size_t i=0;i<native.rows;++i)
        std::memset(static_cast<char*>(native.data.contents)+i*native.rowBytes,
            0,native.columns*sizeof(float));
    matrix.impl_->ready=true;
}
void MetalContext::ZeroPrefixGroups(const MetalMatrix& matrix,size_t groups,size_t capacity,size_t width,size_t dirtyTokens) {
    if(impl_->command)throw std::logic_error("Host prefix zero requires a closed command batch");
    if(!matrix.impl_||matrix.impl_->matrix.device!=impl_->device)
        throw std::invalid_argument("Metal prefix zero matrix belongs to another device or is empty");
    if(matrix.impl_->pending||!matrix.impl_->ready)throw std::logic_error("Prefix zero requires initialized, completed state");
    auto native=matrix.impl_->matrix;
    if(!groups||!capacity||!width||dirtyTokens>capacity||capacity>SIZE_MAX/width||
        groups>SIZE_MAX/(capacity*width)||groups*capacity*width!=native.columns)
        throw std::invalid_argument("Metal prefix zero layout/bounds mismatch");
    for(size_t row=0;row<native.rows;++row)for(size_t group=0;group<groups;++group)
        std::memset(static_cast<char*>(native.data.contents)+row*native.rowBytes+group*capacity*width*sizeof(float),
            0,dirtyTokens*width*sizeof(float));
}
size_t MetalContext::LastDeferredZeroBytes() const {return impl_->deferredZeroBytes;}
size_t MetalContext::LastDeferredZeroStates() const {return impl_->deferredZeroStates;}
void MetalContext::DeferZero(const MetalMatrix& matrix) {
#ifndef GARNET_METAL_DEFERRED_STATE_ZERO
    Zero(matrix);
#else
    if(impl_->command||impl_->submitted.count)throw std::logic_error("Deferred reset requires a closed, drained command batch");
    if(!matrix.impl_||matrix.impl_->matrix.device!=impl_->device)throw std::invalid_argument("Invalid deferred reset matrix/device");
    if(matrix.impl_->pending&&!impl_->zeroQueued.count(matrix.impl_.get()))throw std::logic_error("Cannot defer reset of pending GPU work");
    impl_->zeroQueued.emplace(matrix.impl_.get(),matrix.impl_);
    // Pending/readability checks keep other contexts from stale physical state.
    matrix.impl_->ready=false;matrix.impl_->pending=true;
#endif
}
void MetalContext::Begin() {
    @autoreleasepool {
    if(impl_->command)throw std::logic_error("Metal command batch already open");
    if(impl_->submitted.count)throw std::logic_error("Undrained Metal partial batches");
    impl_->screenedHeadDispatches=0;
    impl_->screenedHeadSeedRows=0;
    impl_->interleavedScreenDispatches=0;
    impl_->visionPrefetchDispatches=0;
#ifdef GARNET_METAL_SCREENED_HEAD
    impl_->screenDiagnosticReady=false;
#endif
    impl_->partialSubmissions=0;
    impl_->bodyShapeDispatches=impl_->bodyShapeMisses=0;
    impl_->joinedPrefillPairs=0;
    impl_->fusedHeadArgmaxDispatches=0;
    impl_->deferredZeroBytes=impl_->deferredZeroStates=0;
    impl_->constantHeadDispatches=0;impl_->singleCoderHeadDispatches=0;impl_->residualNormDispatches=0;impl_->packedPrefillNormDispatches=0;impl_->globalHalfAlphaDispatches=0;impl_->wideBatchPrefillDispatches=0;impl_->visionGlobalAlphaDispatches=0;impl_->visionParallelDispatches=0;impl_->joinedGlobalMlpBlocks=0;
    impl_->command=[impl_->queue commandBuffer];
    if(!impl_->command)throw std::runtime_error("Cannot create Metal command buffer");
#ifdef GARNET_METAL_DEFERRED_STATE_ZERO
    if(!impl_->zeroQueued.empty()) {
        try {
            auto encoder=[impl_->command blitCommandEncoder];
            if(!encoder)throw std::runtime_error("Cannot create deferred state clear encoder");
            for(const auto& entry:impl_->zeroQueued) {
                const auto& target=entry.second;auto matrix=target->matrix;
                if(impl_->pendingOutputs.insert(target.get()).second)impl_->outputs.push_back(target);
                for(size_t row=0;row<matrix.rows;++row)
                    [encoder fillBuffer:matrix.data range:NSMakeRange(row*matrix.rowBytes,matrix.columns*sizeof(float)) value:0];
                impl_->deferredZeroBytes+=matrix.rows*matrix.columns*sizeof(float);++impl_->deferredZeroStates;
            }
            [encoder endEncoding];impl_->zeroQueued.clear();
        }catch(...){Abort();throw;}
    }
#endif
    }
}
void MetalContext::EncodeMatMul(const MetalMatrix& left,const MetalMatrix& right,
    const MetalMatrix& output,bool transposeRight) {
    @autoreleasepool {
    if(!impl_->command)throw std::logic_error("Begin a Metal command batch first");
    if(!left.impl_||!right.impl_||!output.impl_)throw std::invalid_argument("Empty Metal matrix");
    auto a=left.impl_->matrix,b=right.impl_->matrix,c=output.impl_->matrix;
    if(a.device!=impl_->device||b.device!=impl_->device||c.device!=impl_->device)
        throw std::invalid_argument("Metal matrices belong to another device");
    auto usable=[&](const MetalMatrix& matrix){
        return matrix.impl_->ready || (matrix.impl_->pending &&
            impl_->pendingOutputs.count(matrix.impl_.get())!=0);
    };
    if(!usable(left)||!usable(right))throw std::logic_error("Uninitialized Metal input");
    if(a.columns!=(transposeRight?b.columns:b.rows)||c.rows!=a.rows||
        c.columns!=(transposeRight?b.rows:b.columns))throw std::invalid_argument("Metal matmul shape mismatch");
    if(c.data==a.data||c.data==b.data)throw std::invalid_argument("In-place Metal matmul is unsupported");
    if(output.impl_->pending && impl_->pendingOutputs.count(output.impl_.get())==0)
        throw std::logic_error("Metal output is in another command batch");
    const std::array<size_t,4> key{c.rows,c.columns,a.columns,size_t(transposeRight)};
    auto found=impl_->matmul.find(key);
    if(found==impl_->matmul.end()) {
        auto op=[[MPSMatrixMultiplication alloc] initWithDevice:impl_->device transposeLeft:NO
            transposeRight:transposeRight resultRows:c.rows resultColumns:c.columns
            interiorColumns:a.columns alpha:1.0 beta:0.0];
        found=impl_->matmul.emplace(key,op).first;
    }
    auto op=found->second;
    [op encodeToCommandBuffer:impl_->command leftMatrix:a rightMatrix:b resultMatrix:c];
    output.impl_->ready=false;output.impl_->pending=true;
    if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
    }
}
void MetalContext::EncodePointwise(const MetalMatrix& left,const MetalMatrix& right,
    const MetalMatrix& output,unsigned operation,float epsilon,float offset) {
    @autoreleasepool {
    if(operation>4)throw std::invalid_argument("Unknown Metal pointwise operation");
    const bool rms=operation==0,mask=operation==3,bias=operation==4;
    if(!impl_->command)throw std::logic_error("Begin a Metal command batch first");
    if(!left.impl_||!right.impl_||!output.impl_)throw std::invalid_argument("Empty Metal matrix");
    auto a=left.impl_->matrix,b=right.impl_->matrix,c=output.impl_->matrix;
    if(a.device!=impl_->device||b.device!=impl_->device||c.device!=impl_->device)
        throw std::invalid_argument("Metal matrices belong to another device");
    auto usable=[&](const MetalMatrix& matrix){return matrix.impl_->ready ||
        (matrix.impl_->pending && impl_->pendingOutputs.count(matrix.impl_.get())!=0);};
    if(!usable(left)||!usable(right))throw std::logic_error("Uninitialized Metal input");
    if(a.rows!=c.rows||a.columns!=c.columns||b.columns!=(mask?1:a.columns)||b.rows!=((rms||bias)?1:a.rows))
        throw std::invalid_argument("Qwen Metal operation shape mismatch");
    if(rms && (!(epsilon>0)||!std::isfinite(epsilon)||!std::isfinite(offset)))
        throw std::invalid_argument("Invalid RMS normalization parameters");
    if(c.data==a.data||c.data==b.data)throw std::invalid_argument("In-place operation is unsupported");
    if(output.impl_->pending && impl_->pendingOutputs.count(output.impl_.get())==0)
        throw std::logic_error("Metal output is in another command batch");
    struct Params {uint64_t rows,columns,aStride,bStride,cStride;float epsilon,offset;};
    Params p{a.rows,a.columns,a.rowBytes/4,b.rowBytes/4,c.rowBytes/4,epsilon,offset};
    auto pipeline=rms?impl_->kernels->rms:(operation==1?impl_->kernels->silu:((operation==2||bias)?impl_->kernels->add:impl_->kernels->mask));
    id<MTLComputeCommandEncoder> encoder=[impl_->command computeCommandEncoder];
    if(!encoder)throw std::runtime_error("Cannot create Metal compute encoder");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a.data offset:0 atIndex:0];[encoder setBuffer:b.data offset:0 atIndex:1];
    [encoder setBuffer:c.data offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];
    const size_t count=rms?a.rows:Elements(a.rows,a.columns);
    const size_t width=std::min(size_t(pipeline.maxTotalThreadsPerThreadgroup),count);
    if(rms) {
        if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128)
            throw std::runtime_error("Parallel RMS requires 32-lane SIMD groups");
        [encoder dispatchThreadgroups:MTLSizeMake(a.rows,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    }else [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
    [encoder endEncoding];output.impl_->ready=false;output.impl_->pending=true;
    if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
    }
}
void MetalContext::EncodeRMSNorm(const MetalMatrix& input,const MetalMatrix& weight,
    const MetalMatrix& output,float epsilon,float weightOffset) {
    EncodePointwise(input,weight,output,0,epsilon,weightOffset);
}
void MetalContext::EncodeResidualRMSNorm(const MetalMatrix& left,const MetalMatrix& right,
    const MetalMatrix& weight,const MetalMatrix& residual,const MetalMatrix& normalized,float epsilon,float offset) {
#ifdef GARNET_METAL_RESIDUAL_NORM
    @autoreleasepool {
    if(!impl_->command)throw std::logic_error("Begin a Metal command batch first");
    const std::array<MetalMatrix,5> matrices={left,right,weight,residual,normalized};
    for(const auto& m:matrices)if(!m.impl_||m.impl_->matrix.device!=impl_->device)
        throw std::invalid_argument("Invalid residual normalization matrix/device");
    const auto a=left.impl_->matrix,w=weight.impl_->matrix;
    if(!left.CompatibleLayout(right)||!left.CompatibleLayout(residual)||!left.CompatibleLayout(normalized)||
        w.rows!=1||w.columns!=a.columns||!(epsilon>0)||!std::isfinite(epsilon)||!std::isfinite(offset))
        throw std::invalid_argument("Residual normalization shape/layout/parameters mismatch");
    for(size_t i=0;i<3;++i)if(!matrices[i].impl_->ready&&
        !(matrices[i].impl_->pending&&impl_->pendingOutputs.count(matrices[i].impl_.get())))
        throw std::logic_error("Uninitialized residual normalization input");
    for(size_t i=3;i<5;++i) {
        for(size_t j=0;j<i;++j)if(matrices[i].SharesStorage(matrices[j]))
            throw std::invalid_argument("Aliased residual normalization output");
        if(matrices[i].impl_->pending&&!impl_->pendingOutputs.count(matrices[i].impl_.get()))
            throw std::logic_error("Residual normalization output belongs to another batch");
    }
    auto pipeline=impl_->kernels->residualNorm;
    if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128)
        throw std::runtime_error("Residual normalization requires 32-lane SIMD groups");
    struct Params {uint64_t rows,columns,aStride,bStride,cStride;float epsilon,offset;};
    Params p{a.rows,a.columns,a.rowBytes/4,w.rowBytes/4,a.rowBytes/4,epsilon,offset};
    auto encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Cannot create residual normalization encoder");
    [encoder setComputePipelineState:pipeline];
    for(size_t i=0;i<5;++i)[encoder setBuffer:matrices[i].impl_->matrix.data offset:0 atIndex:i];
    [encoder setBytes:&p length:sizeof(p) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(a.rows,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [encoder endEncoding];
    for(size_t i=3;i<5;++i){auto state=matrices[i].impl_;state->ready=false;state->pending=true;
        if(impl_->pendingOutputs.insert(state.get()).second)impl_->outputs.push_back(state);}
    ++impl_->residualNormDispatches;
    }
#else
    throw std::logic_error("Residual normalization experiment is disabled");
#endif
}
void MetalContext::EncodePackedPrefillNorm(const MetalMatrix& left,const MetalMatrix& right,
    const MetalMatrix& weight,const MetalMatrix& packed,const MetalMatrix& reshaped,const MetalMatrix& normalized,float epsilon,float offset) {
#ifdef GARNET_METAL_PACKED_PREFILL_NORM
    @autoreleasepool {
    if(!impl_->command)throw std::logic_error("Begin a Metal command batch first");
    const std::array<MetalMatrix,6> matrices={left,right,weight,packed,reshaped,normalized};
    for(const auto& m:matrices)if(!m.impl_||m.impl_->matrix.device!=impl_->device)
        throw std::invalid_argument("Invalid packed prefill normalization matrix/device");
    const auto a=left.impl_->matrix,w=weight.impl_->matrix,n=reshaped.impl_->matrix;
    if(!left.CompatibleLayout(right)||!left.CompatibleLayout(packed)||!reshaped.CompatibleLayout(normalized)||
        a.rows>2||!n.columns||a.columns%n.columns||w.rows!=1||w.columns!=n.columns||
        !(epsilon>0)||!std::isfinite(epsilon)||!std::isfinite(offset))
        throw std::invalid_argument("Packed prefill normalization shape/layout/parameters mismatch");
    const size_t chunk=a.columns/n.columns;
    if(chunk<2||chunk>kMetalPrefillChunkLimit||n.rows!=Elements(a.rows,chunk))
        throw std::invalid_argument("Packed prefill normalization chunk mismatch");
    for(size_t i=0;i<3;++i)if(!matrices[i].impl_->ready&&
        !(matrices[i].impl_->pending&&impl_->pendingOutputs.count(matrices[i].impl_.get())))
        throw std::logic_error("Uninitialized packed prefill normalization input");
    for(size_t i=3;i<6;++i) {
        for(size_t j=0;j<i;++j)if(matrices[i].SharesStorage(matrices[j]))
            throw std::invalid_argument("Aliased packed prefill normalization output");
        if(matrices[i].impl_->pending&&!impl_->pendingOutputs.count(matrices[i].impl_.get()))
            throw std::logic_error("Packed prefill normalization output belongs to another batch");
    }
    auto pipeline=impl_->kernels->packedPrefillNorm;
    if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128)
        throw std::runtime_error("Packed prefill normalization requires 32-lane SIMD groups");
    struct Params {uint64_t rows,columns,packedStride,normStride,chunk;float epsilon,offset;};
    Params p{n.rows,n.columns,a.rowBytes/4,n.rowBytes/4,chunk,epsilon,offset};
    auto encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Cannot create packed prefill normalization encoder");
    [encoder setComputePipelineState:pipeline];
    for(size_t i=0;i<6;++i)[encoder setBuffer:matrices[i].impl_->matrix.data offset:0 atIndex:i];
    [encoder setBytes:&p length:sizeof(p) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(n.rows,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
    for(size_t i=3;i<6;++i){auto state=matrices[i].impl_;state->ready=false;state->pending=true;
        if(impl_->pendingOutputs.insert(state.get()).second)impl_->outputs.push_back(state);}
    ++impl_->packedPrefillNormDispatches;
    }
#else
    throw std::logic_error("Packed prefill normalization experiment is disabled");
#endif
}
void MetalContext::EncodeSiluMul(const MetalMatrix& gate,const MetalMatrix& up,const MetalMatrix& output) {
    EncodePointwise(gate,up,output,1,0,0);
}
void MetalContext::EncodeAdd(const MetalMatrix& a,const MetalMatrix& b,const MetalMatrix& out) {
    EncodePointwise(a,b,out,2,0,0);
}
void MetalContext::EncodeBias(const MetalMatrix& a,const MetalMatrix& b,const MetalMatrix& out) {
    EncodePointwise(a,b,out,4,0,1);
}
void MetalContext::EncodeMaskRows(const MetalMatrix& a,const MetalMatrix& mask,const MetalMatrix& out) {
    EncodePointwise(a,mask,out,3,0,0);
}
void MetalContext::EncodeStateKernel(unsigned operation,const std::vector<MetalMatrix>& inputs,
    const MetalMatrix& output,const void* parameters,size_t parameterBytes,size_t threads,const MetalMatrix* secondOutput) {
    @autoreleasepool {
    if(!impl_->command)throw std::logic_error("Begin a Metal command batch first");
    if(!output.impl_||!threads||threads>std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Invalid Metal state kernel output/grid");
    auto c=output.impl_->matrix;
    if(c.device!=impl_->device)throw std::invalid_argument("Metal output belongs to another device");
    if(output.impl_->pending && impl_->pendingOutputs.count(output.impl_.get())==0)
        throw std::logic_error("Metal output is in another command batch");
    for(size_t inputIndex=0;inputIndex<inputs.size();++inputIndex) {
        const auto& input=inputs[inputIndex];
        if(!input.impl_||input.impl_->matrix.device!=impl_->device)
            throw std::invalid_argument("Invalid Metal state kernel input/device");
        if(!input.impl_->ready && !(input.impl_->pending &&
            impl_->pendingOutputs.count(input.impl_.get())!=0))
            throw std::logic_error("Uninitialized Metal state kernel input");
        if(c.data==input.impl_->matrix.data) {
#ifdef GARNET_METAL_RESIDENT_KV_INPLACE
            if(operation==20&&inputs.size()==4&&inputIndex==1&&output.CompatibleLayout(inputs[1]))continue;
#endif
            throw std::invalid_argument("In-place state kernel is unsupported");
        }
    }
    if((operation==16||operation==18||operation==19)!=(secondOutput!=nullptr))throw std::invalid_argument("Metal secondary output operation mismatch");
    if(secondOutput) {
        if(!secondOutput->impl_||secondOutput->impl_->matrix.device!=impl_->device)
            throw std::invalid_argument("Invalid Metal secondary output/device");
        const auto& second=secondOutput->impl_;
        if(second->pending&&impl_->pendingOutputs.count(second.get())==0)
            throw std::logic_error("Metal secondary output is in another command batch");
        if(second->matrix.data==c.data)throw std::invalid_argument("Metal outputs overlap");
        for(const auto& input:inputs)if(second->matrix.data==input.impl_->matrix.data)
            throw std::invalid_argument("In-place secondary state output is unsupported");
    }
#ifdef GARNET_METAL_RESIDENT_KV_INPLACE
    constexpr unsigned maxOperation=20;
#else
    constexpr unsigned maxOperation=19;
#endif
    if(operation>maxOperation)throw std::invalid_argument("Unknown Metal state kernel");
    id<MTLComputePipelineState> pipelines[]={impl_->kernels->conv,impl_->kernels->delta,impl_->kernels->slice,impl_->kernels->prepare,impl_->kernels->gatedNorm,
        impl_->kernels->concat,impl_->kernels->normRope,impl_->kernels->kvUpdate,impl_->kernels->scores,impl_->kernels->attentionContext,impl_->kernels->vision,impl_->kernels->layout,impl_->kernels->kvChunk,impl_->kernels->attentionChunk,impl_->kernels->kvUpdateVector,impl_->kernels->sliceVector,impl_->kernels->deltaSplit,impl_->kernels->kvChunkVector,impl_->kernels->convChunk,impl_->kernels->deltaChunk,impl_->kernels->kvInplace};
    auto pipeline=pipelines[operation];
    if(!pipeline)throw std::runtime_error("Metal state pipeline is unavailable in this build");
#ifdef GARNET_METAL_PARALLEL_DELTA_SPLIT
    const auto* deltaFields=static_cast<const uint64_t*>(parameters);
    const bool parallelDelta=operation==16&&deltaFields[2]==128&&deltaFields[3]>=16&&deltaFields[3]%16==0&&
        deltaFields[9]%4==0&&deltaFields[11]%4==0&&deltaFields[12]%4==0;
    if(parallelDelta)pipeline=impl_->kernels->deltaSplitParallel;
#endif
    const auto* visionFields=static_cast<const uint64_t*>(parameters);
    const bool visionNorm=operation==10&&visionFields[12]==1;
#ifdef GARNET_METAL_FUSED_VISION_CONTEXT
    const bool fusedVisionContext=operation==10&&visionFields[12]==10;
#else
    constexpr bool fusedVisionContext=false;
#endif
    const bool visionScores=operation==10&&(visionFields[12]==7||fusedVisionContext)&&visionFields[8]<=4096;
    if(visionNorm)pipeline=impl_->kernels->visionNorm;
    if(visionScores)pipeline=impl_->kernels->visionScores;
    bool serialVisionPrefetch=false;
#ifdef GARNET_METAL_SERIAL_VISION_PREFETCH
    auto prefetch=impl_->kernels->visionPrefetch;
    if(operation==10&&visionFields[12]==8&&visionFields[8]==216&&visionFields[9]==768&&visionFields[10]==12&&
        (visionFields[0]==216||visionFields[0]==432)&&prefetch&&prefetch.threadExecutionWidth==32&&
        prefetch.maxTotalThreadsPerThreadgroup>=128&&prefetch.staticThreadgroupMemoryLength<=impl_->device.maxThreadgroupMemoryLength){
        pipeline=prefetch;serialVisionPrefetch=true;++impl_->visionPrefetchDispatches;
    }
#endif
#ifdef GARNET_METAL_MPS_VISION_SCORES
    // Bounded FP32 scratch isolates heads and batches before batched MPS Q*K^T.
    // Softmax retains the existing SIMD max/sum/exp implementation.
    MetalMatrix mpsScores;
    std::array<uint64_t,13> mpsFields{};
    const bool mpsVisionScores=visionScores&&visionFields[0]<=1024&&visionFields[8]<=512&&
        visionFields[10]<=16&&visionFields[9]/visionFields[10]<=128;
    if(mpsVisionScores) {
        const size_t rows=visionFields[0],tokens=visionFields[8],hidden=visionFields[9],heads=visionFields[10],dim=hidden/heads;
        const std::array<size_t,4> key{rows,tokens,hidden,heads};
        auto found=impl_->visionScores.find(key);
        if(found==impl_->visionScores.end()) {
            Impl::VisionScoreScratch item;
            item.query=Allocate(rows*heads,dim);item.key=Allocate(rows*heads,dim);item.scores=Allocate(rows*heads,tokens);
            const size_t matrices=rows/tokens*heads;
            auto view=[&](const MetalMatrix& matrix,size_t columns){auto m=matrix.impl_->matrix;
                auto descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:tokens columns:columns matrices:matrices
                    rowBytes:m.rowBytes matrixBytes:tokens*m.rowBytes dataType:MPSDataTypeFloat32];
                return [[MPSMatrix alloc] initWithBuffer:m.data descriptor:descriptor];};
            item.queryView=view(item.query,dim);item.keyView=view(item.key,dim);item.scoreView=view(item.scores,tokens);
            item.product=[[MPSMatrixMultiplication alloc] initWithDevice:impl_->device transposeLeft:NO transposeRight:YES
                resultRows:tokens resultColumns:tokens interiorColumns:dim alpha:1 beta:0];
            item.product.batchSize=matrices;
            if(!item.queryView||!item.keyView||!item.scoreView||!item.product)throw std::runtime_error("MPS vision score allocation failed");
            found=impl_->visionScores.emplace(key,std::move(item)).first;
        }
        auto& item=found->second;
        std::copy_n(visionFields,13,mpsFields.begin());
        mpsFields[5]=item.query.impl_->matrix.rowBytes/4;mpsFields[6]=item.key.impl_->matrix.rowBytes/4;
        auto pack=[impl_->command computeCommandEncoder];
        if(!pack)throw std::runtime_error("Cannot create vision Q/K packing encoder");
        [pack setComputePipelineState:impl_->kernels->visionPackQK];
        [pack setBuffer:inputs[0].impl_->matrix.data offset:0 atIndex:0];
        [pack setBuffer:item.query.impl_->matrix.data offset:0 atIndex:1];
        [pack setBuffer:item.key.impl_->matrix.data offset:0 atIndex:2];
        [pack setBytes:mpsFields.data() length:sizeof(mpsFields) atIndex:3];
        const size_t count=Elements(rows,hidden);
        [pack dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(count,size_t(impl_->kernels->visionPackQK.maxTotalThreadsPerThreadgroup)),1,1)];
        [pack endEncoding];
        [item.product encodeToCommandBuffer:impl_->command leftMatrix:item.queryView rightMatrix:item.keyView resultMatrix:item.scoreView];
        mpsScores=item.scores;mpsFields[5]=item.scores.impl_->matrix.rowBytes/4;
        parameters=mpsFields.data();pipeline=fusedVisionContext?impl_->kernels->visionFusedContext:impl_->kernels->visionSoftmax;
#ifdef GARNET_METAL_PARALLEL_VISION_CONTEXT
        auto candidate=impl_->kernels->visionParallelContext;
        const size_t scratchBytes=(tokens*sizeof(float)+15)&~size_t(15);
        if(fusedVisionContext&&tokens==216&&hidden==768&&heads==12&&(rows==216||rows==432)&&
            candidate&&candidate.threadExecutionWidth==32&&candidate.maxTotalThreadsPerThreadgroup>=128&&
            candidate.staticThreadgroupMemoryLength+scratchBytes<=impl_->device.maxThreadgroupMemoryLength) {
            pipeline=candidate;++impl_->visionParallelDispatches;
        }
#endif
    }
#endif
    auto encoder=[impl_->command computeCommandEncoder];
    if(!encoder)throw std::runtime_error("Cannot create Metal state kernel encoder");
    [encoder setComputePipelineState:pipeline];
    for(size_t i=0;i<inputs.size();++i)[encoder setBuffer:inputs[i].impl_->matrix.data offset:0 atIndex:i];
#ifdef GARNET_METAL_MPS_VISION_SCORES
    if(mpsVisionScores)[encoder setBuffer:mpsScores.impl_->matrix.data offset:0 atIndex:1];
#endif
    [encoder setBuffer:c.data offset:0 atIndex:inputs.size()];
    if(secondOutput)[encoder setBuffer:secondOutput->impl_->matrix.data offset:0 atIndex:inputs.size()+1];
    [encoder setBytes:parameters length:parameterBytes atIndex:inputs.size()+1+(secondOutput?1:0)];
    if(serialVisionPrefetch) {
        [encoder dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(threads,size_t(128)),1,1)];
    } else if(visionNorm||visionScores) {
        if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128)throw std::runtime_error("Parallel vision requires 32-lane SIMD groups");
        if(visionScores)[encoder setThreadgroupMemoryLength:((visionFields[8]*sizeof(float)+15)&~size_t(15)) atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(visionFields[0]*(visionScores?visionFields[10]:1),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];

#ifdef GARNET_METAL_PARALLEL_DELTA_SPLIT
    } else if(parallelDelta) {
        if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128)throw std::runtime_error("Parallel recurrent reduction requires 32-lane SIMD groups");
        [encoder dispatchThreadgroups:MTLSizeMake(threads/16,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
#endif
    } else if(operation==19) {
        if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128)throw std::runtime_error("Recurrent chunk requires 32-lane SIMD groups");
        [encoder dispatchThreadgroups:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    } else if(operation==1||operation==16||operation==4||operation==6||operation==8||operation==13) {
        const auto* fields=static_cast<const uint64_t*>(parameters);
        const size_t capacity=operation==8?fields[4]:(operation==13?fields[5]:0);
        if(operation==8&&capacity>4096) {
            [encoder setComputePipelineState:impl_->kernels->scoresSerial];
            [encoder dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(std::min(threads,size_t(impl_->kernels->scoresSerial.maxTotalThreadsPerThreadgroup)),1,1)];
        } else {
            const size_t width=(operation==1||operation==16||operation==8||operation==13)?128:32;
            if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<width)throw std::runtime_error("Parallel attention requires 32-lane SIMD groups");
            if(operation==8||operation==13)[encoder setThreadgroupMemoryLength:((capacity*sizeof(float)+15)&~size_t(15)) atIndex:0];
            if(operation==13)[encoder setThreadgroupMemoryLength:((capacity*sizeof(int)+15)&~size_t(15)) atIndex:1];
            [encoder dispatchThreadgroups:MTLSizeMake((operation==1||operation==16)?threads/fields[3]:threads,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
        }
    } else [encoder dispatchThreads:MTLSizeMake(threads,1,1)
        threadsPerThreadgroup:MTLSizeMake(std::min(threads,size_t(pipeline.maxTotalThreadsPerThreadgroup)),1,1)];
    [encoder endEncoding];output.impl_->ready=false;output.impl_->pending=true;if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
    if(secondOutput){const auto& second=secondOutput->impl_;second->ready=false;second->pending=true;if(impl_->pendingOutputs.insert(second.get()).second)impl_->outputs.push_back(second);}
    }
}
void MetalContext::EncodeLayout(const MetalMatrix& input,const MetalMatrix& flags,const MetalMatrix& output,size_t tokenWidth) {
 if(!input.impl_||!flags.impl_||!output.impl_)throw std::invalid_argument("Missing layout matrix");
 auto a=input.impl_->matrix,b=flags.impl_->matrix,o=output.impl_->matrix;
 if(Elements(a.rows,a.columns)!=Elements(o.rows,o.columns)||
    (tokenWidth&&(a.columns%tokenWidth||b.rows!=a.rows||b.columns!=a.columns/tokenWidth)))
    throw std::invalid_argument("Metal layout shape mismatch");
 struct Params{uint64_t aRows,aColumns,aStride,bStride,oRows,oColumns,oStride,tokenWidth;};
 Params p{a.rows,a.columns,a.rowBytes/4,b.rowBytes/4,o.rows,o.columns,o.rowBytes/4,tokenWidth};
 EncodeStateKernel(11,{input,flags},output,&p,sizeof(p),Elements(o.rows,o.columns));
}
void MetalContext::EncodeVision(unsigned operation,const std::vector<MetalMatrix>& inputs,
    const MetalMatrix& output,size_t tokens,size_t hidden,size_t heads,size_t mergeUnit) {
#ifdef GARNET_METAL_FUSED_VISION_CONTEXT
    constexpr unsigned maxMode=10;
#else
    constexpr unsigned maxMode=9;
#endif
    if(inputs.size()!=3||operation>maxMode||!output.impl_||!tokens||!hidden||!heads||hidden%heads||!mergeUnit)
        throw std::invalid_argument("Invalid Metal vision dispatch");
    for(auto& input:inputs)if(!input.impl_)throw std::invalid_argument("Missing Metal vision input");
    auto a=inputs[0].impl_->matrix,b=inputs[1].impl_->matrix,c=inputs[2].impl_->matrix,o=output.impl_->matrix;
    const bool same=a.rows==o.rows&&a.columns==o.columns;
    bool valid=false;
    switch(operation) {
    case 0:valid=same&&b.rows==1&&b.columns==a.columns;break;
    case 1:valid=same&&b.rows==1&&c.rows==1&&b.columns==a.columns&&c.columns==a.columns;break;
    case 2:case 3:valid=same;break;
    case 4:valid=a.rows==o.rows*mergeUnit&&o.columns==a.columns*mergeUnit;break;
    case 5:valid=same&&a.rows%tokens==0&&b.columns==a.columns&&c.rows==tokens&&c.columns==8;break;
    case 6:valid=same&&a.rows%tokens==0&&a.columns==3*hidden&&b.rows==tokens&&b.columns==hidden/heads&&(hidden/heads)%4==0;break;
    case 7:valid=a.rows==o.rows&&a.rows%tokens==0&&a.columns==3*hidden&&o.columns==heads*tokens;break;
    case 8:valid=a.rows==o.rows&&a.rows==b.rows&&a.rows%tokens==0&&a.columns==heads*tokens&&b.columns==3*hidden&&o.columns==hidden;break;
    case 9:valid=a.rows*tokens==o.rows&&a.columns==o.columns*tokens;break;
#ifdef GARNET_METAL_FUSED_VISION_CONTEXT
    case 10:valid=a.rows==o.rows&&a.rows%tokens==0&&a.columns==3*hidden&&o.columns==hidden&&
        a.rows<=1024&&tokens<=512&&heads<=16&&hidden/heads<=128;break;
#endif
    }
    if(!valid)throw std::invalid_argument("Metal vision matrix dimensions mismatch");
    // Positional indices are graph-owned constants validated during lowering.
    struct Params {uint64_t rows,columns,outRows,outColumns,aStride,bStride,cStride,oStride,tokens,hidden,heads,unit,mode;};
    Params p{a.rows,a.columns,o.rows,o.columns,a.rowBytes/4,b.rowBytes/4,c.rowBytes/4,o.rowBytes/4,tokens,hidden,heads,mergeUnit,operation};
    EncodeStateKernel(10,inputs,output,&p,sizeof(p),Elements(o.rows,o.columns));
}
void MetalContext::EncodeCausalConv(const MetalMatrix& token,const MetalMatrix& weight,
    const MetalMatrix& state,const MetalMatrix& active,const MetalMatrix& packed,size_t kernel) {
    if(!token.impl_||!weight.impl_||!state.impl_||!active.impl_||!packed.impl_||!kernel)
        throw std::invalid_argument("Empty Metal causal convolution argument");
    auto t=token.impl_->matrix,w=weight.impl_->matrix,s=state.impl_->matrix,a=active.impl_->matrix,o=packed.impl_->matrix;
    const auto stateWidth=Elements(t.columns,kernel);
    if(stateWidth>std::numeric_limits<size_t>::max()-t.columns||w.rows!=t.columns||w.columns!=kernel||
        s.rows!=t.rows||s.columns!=stateWidth||a.rows!=t.rows||a.columns!=1||o.rows!=t.rows||o.columns!=t.columns+stateWidth)
        throw std::invalid_argument("Metal causal convolution shape mismatch");
    struct Params {uint64_t batch,channels,kernel,tStride,wStride,sStride,aStride,oStride;};
    Params p{t.rows,t.columns,kernel,t.rowBytes/4,w.rowBytes/4,s.rowBytes/4,a.rowBytes/4,o.rowBytes/4};
    EncodeStateKernel(0,{token,weight,state,active},packed,&p,sizeof(p),Elements(t.rows,t.columns));
}
void MetalContext::EncodeSliceGroups(const MetalMatrix& input,const MetalMatrix& output,
    size_t groups,size_t sourceWidth,size_t start,size_t count) {
    if(!input.impl_||!output.impl_||!count||start>sourceWidth||count>sourceWidth-start)
        throw std::invalid_argument("Invalid Metal grouped slice");
    auto i=input.impl_->matrix,o=output.impl_->matrix;
    if(i.rows!=o.rows||i.columns!=Elements(groups,sourceWidth)||o.columns!=Elements(groups,count))
        throw std::invalid_argument("Metal grouped slice shape mismatch");
    struct Params {uint64_t batch,groups,sourceWidth,start,count,iStride,oStride;};
    Params p{i.rows,groups,sourceWidth,start,count,i.rowBytes/4,o.rowBytes/4};
    const size_t limit=std::numeric_limits<uint32_t>::max();
    const bool vectorized=sourceWidth%4==0&&start%4==0&&count%4==0&&p.iStride%4==0&&p.oStride%4==0&&
        i.rows<=limit/p.iStride&&o.rows<=limit/p.oStride&&impl_->kernels->sliceVector;
    EncodeStateKernel(vectorized?15:2,{input},output,&p,sizeof(p),Elements(o.rows,o.columns)/(vectorized?4:1));
}
void MetalContext::EncodeCausalConvChunk(const MetalMatrix& tokens,const MetalMatrix& weight,
    const MetalMatrix& state,const MetalMatrix& active,const MetalMatrix& output,
    const MetalMatrix& nextState,size_t chunk,size_t kernel) {
    if(!tokens.impl_||!weight.impl_||!state.impl_||!active.impl_||!output.impl_||!nextState.impl_||!chunk||chunk>kMetalPrefillChunkLimit||!kernel||kernel>8)
        throw std::invalid_argument("Invalid Metal chunk convolution arguments");
    auto t=tokens.impl_->matrix,w=weight.impl_->matrix,s=state.impl_->matrix,a=active.impl_->matrix,
        o=output.impl_->matrix,n=nextState.impl_->matrix;
    const size_t batch=s.rows,stateWidth=Elements(t.columns,kernel);
    if(t.rows!=Elements(batch,chunk)||w.rows!=t.columns||w.columns!=kernel||s.columns!=stateWidth||
        a.rows!=batch||a.columns!=chunk||o.rows!=t.rows||o.columns!=t.columns||n.rows!=batch||n.columns!=stateWidth)
        throw std::invalid_argument("Metal chunk convolution shape mismatch");
    struct Params {uint64_t batch,chunk,channels,kSize,tStride,wStride,sStride,aStride,oStride,nStride;};
    Params p{batch,chunk,t.columns,kernel,t.rowBytes/4,w.rowBytes/4,s.rowBytes/4,a.rowBytes/4,o.rowBytes/4,n.rowBytes/4};
    EncodeStateKernel(18,{tokens,weight,state,active},output,&p,sizeof(p),Elements(batch,t.columns),&nextState);
}
void MetalContext::EncodeGatedDelta(const MetalMatrix& query,const MetalMatrix& key,const MetalMatrix& value,
    const MetalMatrix& decay,const MetalMatrix& beta,const MetalMatrix& state,const MetalMatrix& active,
    const MetalMatrix& packed,size_t heads,size_t keyDim,size_t valueDim) {
    if(!query.impl_||!key.impl_||!value.impl_||!decay.impl_||!beta.impl_||!state.impl_||!active.impl_||!packed.impl_)
        throw std::invalid_argument("Empty Metal gated delta argument");
    auto q=query.impl_->matrix,k=key.impl_->matrix,v=value.impl_->matrix,g=decay.impl_->matrix,
        b=beta.impl_->matrix,s=state.impl_->matrix,a=active.impl_->matrix,o=packed.impl_->matrix;
    const auto qWidth=Elements(heads,keyDim),vWidth=Elements(heads,valueDim),sWidth=Elements(qWidth,valueDim);
    if(sWidth>std::numeric_limits<size_t>::max()-vWidth||q.columns!=qWidth||k.rows!=q.rows||k.columns!=qWidth||
        v.rows!=q.rows||v.columns!=vWidth||g.rows!=q.rows||g.columns!=heads||b.rows!=q.rows||b.columns!=heads||
        s.rows!=q.rows||s.columns!=sWidth||a.rows!=q.rows||a.columns!=1||o.rows!=q.rows||o.columns!=vWidth+sWidth)
        throw std::invalid_argument("Metal gated delta shape mismatch");
    struct Params {uint64_t batch,heads,keyDim,valueDim,qStride,kStride,vStride,gStride,bStride,sStride,aStride,oStride;};
    Params p{q.rows,heads,keyDim,valueDim,q.rowBytes/4,k.rowBytes/4,v.rowBytes/4,g.rowBytes/4,b.rowBytes/4,
        s.rowBytes/4,a.rowBytes/4,o.rowBytes/4};
    EncodeStateKernel(1,{query,key,value,decay,beta,state,active},packed,&p,sizeof(p),Elements(q.rows,vWidth));
}
void MetalContext::EncodeGatedDeltaSplit(const MetalMatrix& query,const MetalMatrix& key,const MetalMatrix& value,
    const MetalMatrix& decay,const MetalMatrix& beta,const MetalMatrix& state,const MetalMatrix& active,
    const MetalMatrix& result,const MetalMatrix& nextState,size_t heads,size_t keyDim,size_t valueDim) {
    if(!query.impl_||!key.impl_||!value.impl_||!decay.impl_||!beta.impl_||!state.impl_||!active.impl_||!result.impl_||!nextState.impl_)
        throw std::invalid_argument("Empty Metal gated delta argument");
    auto q=query.impl_->matrix,k=key.impl_->matrix,v=value.impl_->matrix,g=decay.impl_->matrix,
        b=beta.impl_->matrix,s=state.impl_->matrix,a=active.impl_->matrix,o=result.impl_->matrix,next=nextState.impl_->matrix;
    const auto qWidth=Elements(heads,keyDim),vWidth=Elements(heads,valueDim),sWidth=Elements(qWidth,valueDim);
    if(sWidth>std::numeric_limits<size_t>::max()-vWidth||q.columns!=qWidth||k.rows!=q.rows||k.columns!=qWidth||
        v.rows!=q.rows||v.columns!=vWidth||g.rows!=q.rows||g.columns!=heads||b.rows!=q.rows||b.columns!=heads||
        s.rows!=q.rows||s.columns!=sWidth||a.rows!=q.rows||a.columns!=1||o.rows!=q.rows||o.columns!=vWidth||next.rows!=q.rows||next.columns!=sWidth)
        throw std::invalid_argument("Metal gated delta shape mismatch");
    struct Params {uint64_t batch,heads,keyDim,valueDim,qStride,kStride,vStride,gStride,bStride,sStride,aStride,oStride,nextStride;};
    Params p{q.rows,heads,keyDim,valueDim,q.rowBytes/4,k.rowBytes/4,v.rowBytes/4,g.rowBytes/4,b.rowBytes/4,
        s.rowBytes/4,a.rowBytes/4,o.rowBytes/4,next.rowBytes/4};
    EncodeStateKernel(16,{query,key,value,decay,beta,state,active},result,&p,sizeof(p),Elements(q.rows,vWidth),&nextState);
}
void MetalContext::EncodeGatedDeltaChunk(const MetalMatrix& prepared,const MetalMatrix& state,const MetalMatrix& active,
    const MetalMatrix& output,const MetalMatrix& nextState,size_t chunk,size_t heads,size_t keyDim,size_t valueDim) {
    if(!prepared.impl_||!state.impl_||!active.impl_||!output.impl_||!nextState.impl_||!chunk||chunk>kMetalPrefillChunkLimit||!heads||heads>128||keyDim!=128||!valueDim||valueDim>4096||valueDim%16)
        throw std::invalid_argument("Invalid Metal recurrent chunk arguments");
    auto a=prepared.impl_->matrix,s=state.impl_->matrix,mask=active.impl_->matrix,o=output.impl_->matrix,n=nextState.impl_->matrix;
    const size_t batch=s.rows,qw=Elements(heads,keyDim),vw=Elements(heads,valueDim),stateWidth=Elements(qw,valueDim);
    if(a.rows!=Elements(batch,chunk)||a.columns!=2*qw+vw+2*heads||s.columns!=stateWidth||mask.rows!=batch||mask.columns!=chunk||
        o.rows!=a.rows||o.columns!=vw||n.rows!=batch||n.columns!=stateWidth||a.rowBytes%16||s.rowBytes%16||o.rowBytes%16||n.rowBytes%16)
        throw std::invalid_argument("Metal recurrent chunk shape mismatch");
    struct Params {uint64_t batch,chunk,heads,keyDim,valueDim,pStride,sStride,aStride,oStride,nextStride;};
    Params p{batch,chunk,heads,keyDim,valueDim,a.rowBytes/4,s.rowBytes/4,mask.rowBytes/4,o.rowBytes/4,n.rowBytes/4};
    EncodeStateKernel(19,{prepared,state,active},output,&p,sizeof(p),Elements(Elements(batch,heads),valueDim/16),&nextState);
}
void MetalContext::EncodeLinearPrepare(const MetalMatrix& conv,const MetalMatrix& a,const MetalMatrix& b,
    const MetalMatrix& aLog,const MetalMatrix& dtBias,const MetalMatrix& output,
    size_t keyHeads,size_t valueHeads,size_t keyDim,size_t valueDim,size_t kernel) {
    if(!conv.impl_||!a.impl_||!b.impl_||!aLog.impl_||!dtBias.impl_||!output.impl_||!keyHeads||!valueHeads||
        valueHeads%keyHeads||!keyDim||!valueDim||keyHeads>128||valueHeads>128||keyDim>4096||valueDim>4096||kernel>64)
        throw std::invalid_argument("Invalid Metal linear preparation arguments");
    auto c=conv.impl_->matrix,ap=a.impl_->matrix,bp=b.impl_->matrix,al=aLog.impl_->matrix,dt=dtBias.impl_->matrix,o=output.impl_->matrix;
    const size_t channels=2*keyHeads*keyDim+valueHeads*valueDim,width=2*valueHeads*keyDim+valueHeads*valueDim+2*valueHeads;
    if(c.columns!=channels*(kernel+1)||ap.rows!=c.rows||bp.rows!=c.rows||ap.columns!=valueHeads||bp.columns!=valueHeads||
        al.rows!=1||al.columns!=valueHeads||dt.rows!=1||dt.columns!=valueHeads||o.rows!=c.rows||o.columns!=width)
        throw std::invalid_argument("Metal linear preparation shape mismatch");
    struct Params {uint64_t batch,keyHeads,valueHeads,keyDim,valueDim,kSize,cStride,aStride,bStride,oStride,width;};
    Params p{c.rows,keyHeads,valueHeads,keyDim,valueDim,kernel,c.rowBytes/4,ap.rowBytes/4,bp.rowBytes/4,o.rowBytes/4,width};
    EncodeStateKernel(3,{conv,a,b,aLog,dtBias},output,&p,sizeof(p),Elements(o.rows,o.columns));
}
void MetalContext::EncodeGatedNorm(const MetalMatrix& core,const MetalMatrix& z,const MetalMatrix& norm,
    const MetalMatrix& output,size_t heads,size_t valueDim,float epsilon,size_t columnOffset) {
    if(!core.impl_||!z.impl_||!norm.impl_||!output.impl_||!(epsilon>0)||!std::isfinite(epsilon))
        throw std::invalid_argument("Invalid Metal gated normalization arguments");
    auto c=core.impl_->matrix,gate=z.impl_->matrix,w=norm.impl_->matrix,o=output.impl_->matrix;
    if(c.columns!=Elements(heads,valueDim)||gate.rows!=c.rows||gate.columns!=c.columns||w.rows!=1||w.columns!=valueDim||
        o.rows!=c.rows||columnOffset>o.columns||c.columns>o.columns-columnOffset)throw std::invalid_argument("Metal gated normalization shape mismatch");
    struct Params {uint64_t batch,heads,valueDim,cStride,zStride,oStride,columnOffset;float epsilon;};
    Params p{c.rows,heads,valueDim,c.rowBytes/4,gate.rowBytes/4,o.rowBytes/4,columnOffset,epsilon};
    EncodeStateKernel(4,{core,z,norm},output,&p,sizeof(p),Elements(c.rows,heads));
}
void MetalContext::EncodeConcat(const MetalMatrix& first,const MetalMatrix& second,const MetalMatrix& third,
    const MetalMatrix& output,bool includeThird) {
    if(!first.impl_||!second.impl_||!third.impl_||!output.impl_)throw std::invalid_argument("Empty Metal concat argument");
    auto a=first.impl_->matrix,b=second.impl_->matrix,c=third.impl_->matrix,o=output.impl_->matrix;
    const auto cw=includeThird?c.columns:0;
    if(b.columns>std::numeric_limits<size_t>::max()-a.columns||cw>std::numeric_limits<size_t>::max()-a.columns-b.columns||
        a.rows!=b.rows||a.rows!=c.rows||a.rows!=o.rows||o.columns!=a.columns+b.columns+cw)
        throw std::invalid_argument("Metal concat shape mismatch");
    struct Params {uint64_t batch,aWidth,bWidth,cWidth,aStride,bStride,cStride,oStride;};
    Params p{a.rows,a.columns,b.columns,cw,a.rowBytes/4,b.rowBytes/4,c.rowBytes/4,o.rowBytes/4};
    EncodeStateKernel(5,{first,second,third},output,&p,sizeof(p),Elements(o.rows,o.columns));
}
void MetalContext::EncodeNormRope(const MetalMatrix& input,const MetalMatrix& weight,const MetalMatrix& positions,
    const MetalMatrix& frequencies,const MetalMatrix& output,size_t heads,size_t dim,size_t rotary,float epsilon,float weightOffset,bool normalize) {
    if(!input.impl_||!weight.impl_||!positions.impl_||!frequencies.impl_||!output.impl_||!rotary||rotary%2||rotary>dim||
        !(epsilon>0)||!std::isfinite(epsilon)||!std::isfinite(weightOffset))throw std::invalid_argument("Invalid Metal norm/RoPE arguments");
    auto i=input.impl_->matrix,w=weight.impl_->matrix,p=positions.impl_->matrix,f=frequencies.impl_->matrix,o=output.impl_->matrix;
    if(i.columns!=Elements(heads,dim)||w.rows!=1||w.columns!=dim||p.rows!=i.rows||p.columns!=3||f.rows!=2||f.columns!=rotary/2||
        o.rows!=i.rows||o.columns!=i.columns)throw std::invalid_argument("Metal norm/RoPE shape mismatch");
    struct Params {uint64_t batch,heads,dim,rotary,iStride,pStride,fStride,oStride;float epsilon,weightOffset;uint64_t normalize;};
    Params params{i.rows,heads,dim,rotary,i.rowBytes/4,p.rowBytes/4,f.rowBytes/4,o.rowBytes/4,epsilon,weightOffset,uint64_t(normalize)};
    EncodeStateKernel(6,{input,weight,positions,frequencies},output,&params,sizeof(params),Elements(i.rows,heads));
}
void MetalContext::EncodeKVUpdate(const MetalMatrix& token,const MetalMatrix& cache,const MetalMatrix& slots,
    const MetalMatrix& active,const MetalMatrix& output,size_t kvHeads,size_t dim,size_t capacity) {
    if(!token.impl_||!cache.impl_||!slots.impl_||!active.impl_||!output.impl_)throw std::invalid_argument("Empty Metal KV argument");
    auto t=token.impl_->matrix,c=cache.impl_->matrix,s=slots.impl_->matrix,a=active.impl_->matrix,o=output.impl_->matrix;
    const auto width=Elements(Elements(kvHeads,dim),capacity);
    if(t.rows!=c.rows||t.columns!=kvHeads*dim||c.columns!=width||s.rows!=t.rows||s.columns!=1||a.rows!=t.rows||a.columns!=1||
        o.rows!=c.rows||o.columns!=width)throw std::invalid_argument("Metal KV update shape mismatch");
    struct Params {uint64_t batch,heads,dim,capacity,tStride,cStride,sStride,aStride,oStride;};
    Params p{t.rows,kvHeads,dim,capacity,t.rowBytes/4,c.rowBytes/4,s.rowBytes/4,a.rowBytes/4,o.rowBytes/4};
#ifdef GARNET_METAL_RESIDENT_KV_INPLACE
    if(output.SharesStorage(cache)&&output.CompatibleLayout(cache)) {
        EncodeStateKernel(20,{token,cache,slots,active},output,&p,sizeof(p),Elements(t.rows,t.columns));
        return;
    }
#endif
    const size_t limit32=std::numeric_limits<uint32_t>::max();
    const auto count=Elements(c.rows,width);
    const bool vector=dim%4==0 && t.rowBytes%16==0 && c.rowBytes%16==0 && o.rowBytes%16==0 &&
        count<=limit32 && p.tStride<=limit32/p.batch && p.cStride<=limit32/p.batch &&
        p.oStride<=limit32/p.batch && p.sStride<=limit32/p.batch && p.aStride<=limit32/p.batch;
    EncodeStateKernel(vector?14:7,{token,cache,slots,active},output,&p,sizeof(p),vector?count/4:count);
}
void MetalContext::EncodeKVChunk(const MetalMatrix& token,const MetalMatrix& cache,const MetalMatrix& slots,
    const MetalMatrix& active,const MetalMatrix& output,size_t heads,size_t dim,size_t capacity,size_t chunk,bool contiguousSlots) {
    if(!token.impl_||!cache.impl_||!slots.impl_||!active.impl_||!output.impl_||!chunk||chunk>kMetalPrefillChunkLimit)throw std::invalid_argument("Invalid chunk KV inputs");
    auto t=token.impl_->matrix,c=cache.impl_->matrix,s=slots.impl_->matrix,a=active.impl_->matrix,o=output.impl_->matrix;
    const auto width=Elements(Elements(heads,capacity),dim);
    if(t.rows!=c.rows*chunk||t.columns!=Elements(heads,dim)||c.columns!=width||s.rows!=c.rows||s.columns!=chunk||a.rows!=c.rows||a.columns!=chunk||o.rows!=c.rows||o.columns!=width)
        throw std::invalid_argument("Chunk KV layout mismatch");
    struct Params {uint64_t batch,heads,dim,capacity,chunk,tStride,cStride,sStride,aStride,oStride,contiguous;};
    Params p{c.rows,heads,dim,capacity,chunk,t.rowBytes/4,c.rowBytes/4,s.rowBytes/4,a.rowBytes/4,o.rowBytes/4,uint64_t(contiguousSlots)};
    const size_t count=Elements(c.rows,width),limit=std::numeric_limits<uint32_t>::max();
    const bool vector=dim%4==0&&t.rowBytes%16==0&&c.rowBytes%16==0&&o.rowBytes%16==0&&count<=limit&&
        p.tStride<=limit/t.rows&&p.cStride<=limit/c.rows&&p.oStride<=limit/o.rows&&
        p.sStride<=limit/s.rows&&p.aStride<=limit/a.rows&&impl_->kernels->kvChunkVector;
    EncodeStateKernel(vector?17:12,{token,cache,slots,active},output,&p,sizeof(p),vector?count/4:count);
}
void MetalContext::EncodeAttentionChunk(const std::vector<MetalMatrix>& inputs,const MetalMatrix& output,
    size_t heads,size_t kvHeads,size_t dim,size_t capacity,size_t chunk,bool gated) {
    if(inputs.size()!=8||!output.impl_||!chunk||chunk>kMetalPrefillChunkLimit||!capacity||capacity>2048||!kvHeads||heads%kvHeads)throw std::invalid_argument("Invalid chunk attention profile");
    for(auto& input:inputs)if(!input.impl_)throw std::invalid_argument("Missing chunk attention input");
    auto q=inputs[0].impl_->matrix,k=inputs[1].impl_->matrix,v=inputs[2].impl_->matrix,kc=inputs[3].impl_->matrix,vc=inputs[4].impl_->matrix,
        s=inputs[5].impl_->matrix,a=inputs[6].impl_->matrix,g=inputs[7].impl_->matrix,o=output.impl_->matrix;
    const auto cacheWidth=Elements(Elements(kvHeads,capacity),dim);
    if(q.rows!=kc.rows*chunk||q.columns!=Elements(heads,dim)||k.rows!=q.rows||k.columns!=Elements(kvHeads,dim)||v.rows!=k.rows||v.columns!=k.columns||
        kc.columns!=cacheWidth||vc.rows!=kc.rows||vc.columns!=cacheWidth||s.rows!=kc.rows||s.columns!=chunk||a.rows!=s.rows||a.columns!=chunk||
        g.rows!=q.rows||g.columns!=q.columns||o.rows!=q.rows||o.columns!=q.columns)throw std::invalid_argument("Chunk attention layout mismatch");
    struct Params {uint64_t batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,kcStride,vcStride,sStride,aStride,gStride,oStride,gated;};
    Params p{kc.rows,chunk,heads,kvHeads,dim,capacity,q.rowBytes/4,k.rowBytes/4,v.rowBytes/4,kc.rowBytes/4,vc.rowBytes/4,s.rowBytes/4,a.rowBytes/4,g.rowBytes/4,o.rowBytes/4,uint64_t(gated)};
    EncodeStateKernel(13,inputs,output,&p,sizeof(p),Elements(q.rows,heads));
}
void MetalContext::EncodeAttentionScores(const MetalMatrix& query,const MetalMatrix& key,const MetalMatrix& slots,
    const MetalMatrix& active,const MetalMatrix& output,size_t heads,size_t kvHeads,size_t dim,size_t capacity) {
    if(!query.impl_||!key.impl_||!slots.impl_||!active.impl_||!output.impl_||!kvHeads||!heads||heads%kvHeads)
        throw std::invalid_argument("Invalid Metal attention score arguments");
    auto q=query.impl_->matrix,k=key.impl_->matrix,s=slots.impl_->matrix,a=active.impl_->matrix,o=output.impl_->matrix;
    if(q.columns!=Elements(heads,dim)||k.rows!=q.rows||k.columns!=Elements(Elements(kvHeads,capacity),dim)||s.rows!=q.rows||
        s.columns!=1||a.rows!=q.rows||a.columns!=1||o.rows!=q.rows||o.columns!=Elements(heads,capacity))
        throw std::invalid_argument("Metal attention score shape mismatch");
    struct Params {uint64_t batch,heads,kvHeads,dim,capacity,qStride,kStride,sStride,aStride,oStride;};
    Params p{q.rows,heads,kvHeads,dim,capacity,q.rowBytes/4,k.rowBytes/4,s.rowBytes/4,a.rowBytes/4,o.rowBytes/4};
    EncodeStateKernel(8,{query,key,slots,active},output,&p,sizeof(p),Elements(q.rows,heads));
}
void MetalContext::EncodeAttentionContext(const MetalMatrix& probabilities,const MetalMatrix& value,const MetalMatrix& gate,
    const MetalMatrix& slots,const MetalMatrix& output,size_t heads,size_t kvHeads,size_t dim,size_t capacity,bool gated) {
    if(!probabilities.impl_||!value.impl_||!gate.impl_||!slots.impl_||!output.impl_||!kvHeads||!heads||heads%kvHeads)
        throw std::invalid_argument("Invalid Metal attention context arguments");
    auto p=probabilities.impl_->matrix,v=value.impl_->matrix,g=gate.impl_->matrix,s=slots.impl_->matrix,o=output.impl_->matrix;
    if(p.columns!=Elements(heads,capacity)||v.rows!=p.rows||v.columns!=Elements(Elements(kvHeads,capacity),dim)||g.rows!=p.rows||
        g.columns!=Elements(heads,dim)||s.rows!=p.rows||s.columns!=1||o.rows!=p.rows||o.columns!=g.columns)throw std::invalid_argument("Metal attention context shape mismatch");
    struct Params {uint64_t batch,heads,kvHeads,dim,capacity,pStride,vStride,gStride,sStride,oStride,gated;};
    Params params{p.rows,heads,kvHeads,dim,capacity,p.rowBytes/4,v.rowBytes/4,g.rowBytes/4,s.rowBytes/4,o.rowBytes/4,uint64_t(gated)};
    EncodeStateKernel(9,{probabilities,value,gate,slots},output,&params,sizeof(params),Elements(o.rows,o.columns));
}
void MetalContext::Flush() {
    @autoreleasepool {
    if(!impl_->command)throw std::logic_error("No Metal command batch to flush");
    auto next=[impl_->queue commandBuffer];
    if(!next)throw std::runtime_error("Cannot create Metal continuation command");
    if(!impl_->submitted)impl_->submitted=[NSMutableArray new];
    if(!impl_->submitted)throw std::runtime_error("Cannot retain Metal partial commands");
    [impl_->submitted addObject:impl_->command];
    [impl_->command commit];impl_->command=next;++impl_->partialSubmissions;
    // Pending owners stay retained and usable only by this context. Shared
    // tracked resources on one queue preserve producer/consumer ordering.
    }
}
size_t MetalContext::LastPartialSubmissions() const {return impl_->partialSubmissions;}
void MetalContext::SubmitAndWait() {
    @autoreleasepool {
    if(!impl_->command)throw std::logic_error("No Metal command batch to submit");
    auto command=impl_->command;
    [command commit];[command waitUntilCompleted];
    bool success=command.status==MTLCommandBufferStatusCompleted;
    NSString* failure=command.error.localizedDescription;
    double first=std::numeric_limits<double>::infinity(),last=0;
    auto observe=[&](id<MTLCommandBuffer> c) {
        if(c.GPUStartTime>0&&c.GPUEndTime>=c.GPUStartTime){first=std::min(first,c.GPUStartTime);last=std::max(last,c.GPUEndTime);}
    };
    observe(command);
    for(id<MTLCommandBuffer> prior in impl_->submitted) {
        [prior waitUntilCompleted];observe(prior);
        if(prior.status!=MTLCommandBufferStatusCompleted){success=false;if(!failure)failure=prior.error.localizedDescription;}
    }
    // One observed GPU span, including inter-command gaps; never sum overlaps.
    impl_->gpuMs=success&&std::isfinite(first)?(last-first)*1000.0:0;
    for(auto& output:impl_->outputs){output->pending=false;output->ready=success;}
    impl_->outputs.clear();impl_->pendingOutputs.clear();impl_->command=nil;impl_->submitted=nil;
    if(!success)throw std::runtime_error(failure?std::string(failure.UTF8String):"Metal execution did not complete");
    }
}
double MetalContext::LastGPUMs() const {return impl_->gpuMs;}
void MetalContext::Abort() {
    impl_->screenedHeadDispatches=0;
    impl_->screenedHeadSeedRows=0;
    impl_->interleavedScreenDispatches=0;
    impl_->visionPrefetchDispatches=0;
#ifdef GARNET_METAL_SCREENED_HEAD
    impl_->screenDiagnosticReady=false;
#endif
    // Flushed work may already be executing. Drain it before exposing buffers
    // for reset/reuse; the uncommitted continuation is discarded.
    for(id<MTLCommandBuffer> prior in impl_->submitted)[prior waitUntilCompleted];
    impl_->submitted=nil;impl_->partialSubmissions=0;
    impl_->bodyShapeDispatches=impl_->bodyShapeMisses=0;
    impl_->joinedPrefillPairs=0;
    impl_->fusedHeadArgmaxDispatches=0;
    impl_->deferredZeroBytes=impl_->deferredZeroStates=0;
#ifdef GARNET_METAL_DEFERRED_STATE_ZERO
    for(const auto& entry:impl_->zeroQueued){entry.second->pending=false;entry.second->ready=false;}
    impl_->zeroQueued.clear();
#endif
    impl_->constantHeadDispatches=0;impl_->singleCoderHeadDispatches=0;impl_->residualNormDispatches=0;impl_->packedPrefillNormDispatches=0;impl_->globalHalfAlphaDispatches=0;impl_->wideBatchPrefillDispatches=0;impl_->visionGlobalAlphaDispatches=0;impl_->visionParallelDispatches=0;impl_->joinedGlobalMlpBlocks=0;
    for(auto& output:impl_->outputs){output->pending=false;output->ready=false;}
    impl_->outputs.clear();impl_->pendingOutputs.clear();impl_->command=nil;
}
std::vector<float> MetalContext::Download(const MetalMatrix& value) const {
#ifdef GARNET_METAL_DEFERRED_STATE_ZERO
    if(value.impl_&&impl_->zeroQueued.count(value.impl_.get())) {
        auto matrix=value.impl_->matrix;
        return std::vector<float>(Elements(matrix.rows,matrix.columns),0.0f);
    }
#endif
    if(!value.impl_||!value.impl_->ready||value.impl_->pending)
        throw std::logic_error("Metal matrix is not completed");
    auto matrix=value.impl_->matrix;
    if(matrix.device!=impl_->device)throw std::invalid_argument("Metal matrix belongs to another device");
    std::vector<float> result(Elements(matrix.rows,matrix.columns));
    for(size_t i=0;i<matrix.rows;++i)
        std::memcpy(result.data()+i*matrix.columns,static_cast<char*>(matrix.data.contents)+i*matrix.rowBytes,
            matrix.columns*sizeof(float));
    return result;
}
std::vector<float> MetalContext::MatMul(const std::vector<float>& left,
    const std::vector<float>& right,size_t rows,size_t inner,size_t columns,bool transposeRight) {
    auto a=Upload(left,rows,inner);
    auto b=Upload(right,transposeRight?columns:inner,transposeRight?inner:columns);
    auto c=Allocate(rows,columns);
    Begin();
    try {EncodeMatMul(a,b,c,transposeRight);SubmitAndWait();}
    catch(...) {Abort();throw;}
    return Download(c);
}
}
