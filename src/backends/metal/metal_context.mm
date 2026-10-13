// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
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
#include <mutex>
namespace Garnet {
struct MetalMatrix::Impl {
    MPSMatrix* matrix;
    bool ready=false;
    bool pending=false;
};
struct MetalBF16Weight::Impl {
    id<MTLBuffer> buffer;
    size_t rows,columns;
};
struct MetalPipelines {
    id<MTLLibrary> library;
    id<MTLComputePipelineState> argmax;
    id<MTLComputePipelineState> bf16MatVec;
    id<MTLComputePipelineState> rms;
    id<MTLComputePipelineState> silu;
    id<MTLComputePipelineState> add;
    id<MTLComputePipelineState> mask;
    id<MTLComputePipelineState> conv;
    id<MTLComputePipelineState> delta;
    id<MTLComputePipelineState> slice;
    id<MTLComputePipelineState> prepare;
    id<MTLComputePipelineState> gatedNorm;
    id<MTLComputePipelineState> concat;
    id<MTLComputePipelineState> normRope,kvUpdate,scores,attentionContext;
    id<MTLComputePipelineState> vision,layout;
};
static std::mutex pipelineMutex;
static std::map<uint64_t,std::weak_ptr<MetalPipelines>> pipelineCache;
struct MetalContext::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> command;
    std::vector<std::shared_ptr<MetalMatrix::Impl>> outputs;
    std::unordered_set<const MetalMatrix::Impl*> pendingOutputs;
    std::shared_ptr<MetalPipelines> kernels;
    std::map<std::array<size_t,4>,MPSMatrixMultiplication*> matmul;
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
kernel void bf16_matvec(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group.x)*4+simd,row=group.y;
    if(column>=p.columns||row>=p.rows)return;
    float sum=0.0f;
    for(ulong k=lane;k<p.inner;k+=32)
        sum+=a[row*p.aStride+k]*as_type<float>(uint(weight[column*p.inner+k])<<16);
    sum=simd_sum(sum);
    if(lane==0)out[row*p.oStride+column]=sum;
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
kernel void silu_mul(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    float gate=a[row*p.aStride+col];
    c[row*p.cStride+col]=(gate/(1.0f+exp(-gate)))*b[row*p.bStride+col];
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
    device float* output [[buffer(4)]],constant RopeParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.heads)return;
    ulong row=id/p.heads,head=id%p.heads,base=row*p.iStride+head*p.dim;
    float sum=0.0f;
    for(ulong j=0;j<p.dim;++j){float x=input[base+j];sum+=x*x;}
    float inverse=p.normalize?rsqrt(sum/float(p.dim)+p.epsilon):1.0f;
    for(ulong j=0;j<p.dim;++j) {
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
struct ScoresParams {ulong batch,heads,kvHeads,dim,capacity,qStride,kStride,sStride,aStride,oStride;};
kernel void attention_scores(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
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
struct AttentionParams {ulong batch,heads,kvHeads,dim,capacity,pStride,vStride,gStride,oStride,gated;};
kernel void attention_context(device const float* probabilities [[buffer(0)]],device const float* value [[buffer(1)]],
    device const float* gate [[buffer(2)]],device float* output [[buffer(3)]],constant AttentionParams& p [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.heads*p.dim)return;
    ulong row=id/(p.heads*p.dim),head=(id/p.dim)%p.heads,col=id%p.dim,kv=head/(p.heads/p.kvHeads);
    float result=0.0f;
    for(ulong s=0;s<p.capacity;++s)result+=probabilities[row*p.pStride+head*p.capacity+s]*value[row*p.vStride+(kv*p.capacity+s)*p.dim+col];
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
struct GateParams {ulong batch,heads,valueDim,cStride,zStride,oStride;float epsilon;};
kernel void gated_norm(device const float* core [[buffer(0)]],device const float* z [[buffer(1)]],
    device const float* norm [[buffer(2)]],device float* output [[buffer(3)]],
    constant GateParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.heads)return;
    ulong row=id/p.heads,head=id%p.heads;
    float sum=0.0f;
    for(ulong j=0;j<p.valueDim;++j){float x=core[row*p.cStride+head*p.valueDim+j];sum+=x*x;}
    float inverse=rsqrt(sum/float(p.valueDim)+p.epsilon);
    for(ulong j=0;j<p.valueDim;++j) {
        float gate=z[row*p.zStride+head*p.valueDim+j];
        output[row*p.oStride+head*p.valueDim+j]=core[row*p.cStride+head*p.valueDim+j]*inverse*norm[j]*gate/(1.0f+exp(-gate));
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
struct DeltaParams {ulong batch,heads,keyDim,valueDim,qStride,kStride,vStride,gStride,bStride,sStride,aStride,oStride;};
kernel void gated_delta(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],device const float* decay [[buffer(3)]],
    device const float* beta [[buffer(4)]],device const float* state [[buffer(5)]],
    device const float* active [[buffer(6)]],device float* packed [[buffer(7)]],
    constant DeltaParams& p [[buffer(8)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.heads*p.valueDim)return;
    ulong column=id%p.valueDim,head=(id/p.valueDim)%p.heads,row=id/(p.heads*p.valueDim);
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim;
    ulong outputBase=row*p.oStride+head*(p.keyDim+1)*p.valueDim;
    ulong outBase=outputBase+p.valueDim;
    if(active[row*p.aStride]==0.0f) {
        packed[outputBase+column]=0.0f;
        for(ulong j=0;j<p.keyDim;++j)packed[outBase+j*p.valueDim+column]=state[stateBase+j*p.valueDim+column];
        return;
    }
    float qs=1e-6f,ks=1e-6f;
    for(ulong j=0;j<p.keyDim;++j) {
        float q=query[row*p.qStride+head*p.keyDim+j],k=key[row*p.kStride+head*p.keyDim+j];
        qs+=q*q;ks+=k*k;
    }
    float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks);
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
        auto library=[impl_->device newLibraryWithSource:source options:nil error:&error];
        if(!library)throw std::runtime_error(std::string(error.localizedDescription.UTF8String));
        impl_->kernels->library=library;
        impl_->kernels->argmax=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"argmax_rows"] error:&error];
        impl_->kernels->bf16MatVec=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_matvec"] error:&error];
        impl_->kernels->rms=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"rms_norm"] error:&error];
        impl_->kernels->silu=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"silu_mul"] error:&error];
        impl_->kernels->add=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"add_matrix"] error:&error];
        impl_->kernels->mask=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"mask_rows"] error:&error];
        impl_->kernels->conv=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"causal_conv"] error:&error];
        impl_->kernels->delta=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gated_delta"] error:&error];
        impl_->kernels->slice=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"slice_groups"] error:&error];
        impl_->kernels->prepare=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"linear_prepare"] error:&error];
        impl_->kernels->gatedNorm=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gated_norm"] error:&error];
        impl_->kernels->concat=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"concat_rows"] error:&error];
        impl_->kernels->normRope=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"norm_rope"] error:&error];
        impl_->kernels->kvUpdate=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"kv_update"] error:&error];
        impl_->kernels->scores=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"attention_scores"] error:&error];
        impl_->kernels->attentionContext=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"attention_context"] error:&error];
        impl_->kernels->vision=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"vision_op"] error:&error];
        impl_->kernels->layout=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"chunk_layout"] error:&error];
        if(!impl_->kernels->argmax||!impl_->kernels->bf16MatVec||!impl_->kernels->rms||!impl_->kernels->silu||!impl_->kernels->add||!impl_->kernels->mask||!impl_->kernels->conv||!impl_->kernels->delta||!impl_->kernels->slice||
            !impl_->kernels->prepare||!impl_->kernels->gatedNorm||!impl_->kernels->concat||!impl_->kernels->normRope||!impl_->kernels->kvUpdate||!impl_->kernels->scores||!impl_->kernels->attentionContext||!impl_->kernels->vision||!impl_->kernels->layout)
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
MetalBF16Weight MetalContext::UploadBF16(const void* data,size_t bytes,size_t rows,size_t columns) {
    @autoreleasepool {
        if(!data||bytes!=Elements(rows,columns)*sizeof(uint16_t))throw std::invalid_argument("BF16 weight size mismatch");
        if(bytes>impl_->device.maxBufferLength)throw std::invalid_argument("BF16 weight exceeds device buffer limit");
        // Inspect raw bits without introducing alignment requirements on mapped files.
        for(size_t i=0;i<rows*columns;++i){uint16_t bits;std::memcpy(&bits,static_cast<const char*>(data)+2*i,2);
            if((bits&0x7f80)==0x7f80)throw std::invalid_argument("Nonfinite BF16 checkpoint weight");}
        auto value=std::make_shared<MetalBF16Weight::Impl>();value->rows=rows;value->columns=columns;
        value->buffer=[impl_->device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        if(!value->buffer)throw std::runtime_error("BF16 weight allocation failed");
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
void MetalContext::EncodeBF16MatVec(const MetalMatrix& input,const MetalBF16Weight& weight,const MetalMatrix& output) {
    @autoreleasepool {
        if(!impl_->command||!input.impl_||!weight.impl_||!output.impl_)throw std::invalid_argument("Missing BF16 projection batch/input");
        auto a=input.impl_->matrix,o=output.impl_->matrix;auto w=weight.impl_;
        if(a.device!=impl_->device||o.device!=impl_->device||w->buffer.device!=impl_->device||
            a.columns!=w->columns||o.rows!=a.rows||o.columns!=w->rows||a.rows>2||o.data==a.data)
            throw std::invalid_argument("BF16 projection shape/device mismatch (batch 1/2)");
        auto usable=[&](const MetalMatrix& m){return m.impl_->ready||(m.impl_->pending&&
            impl_->pendingOutputs.count(m.impl_.get())!=0);};
        if(!usable(input))throw std::logic_error("Uninitialized BF16 projection input");
        if(output.impl_->pending&&impl_->pendingOutputs.count(output.impl_.get())==0)
            throw std::logic_error("BF16 output belongs to another command batch");
        auto pipeline=impl_->kernels->bf16MatVec;
        if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128)
            throw std::runtime_error("BF16 projection requires 32-lane SIMD groups");
        struct Params{uint64_t rows,inner,columns,aStride,oStride;};
        Params p{a.rows,a.columns,o.columns,a.rowBytes/4,o.rowBytes/4};
        auto encoder=[impl_->command computeCommandEncoder];
        if(!encoder)throw std::runtime_error("Cannot create BF16 projection encoder");
        [encoder setComputePipelineState:pipeline];[encoder setBuffer:a.data offset:0 atIndex:0];
        [encoder setBuffer:w->buffer offset:0 atIndex:1];[encoder setBuffer:o.data offset:0 atIndex:2];
        [encoder setBytes:&p length:sizeof(p) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake((o.columns+3)/4,o.rows,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
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
    if(matrix.impl_->pending)throw std::logic_error("Cannot update pending Metal output");
    auto native=matrix.impl_->matrix;
    if(!values||count!=Elements(native.rows,native.columns))throw std::invalid_argument("Metal update size mismatch");
    for(size_t i=0;i<native.rows;++i)
        std::memcpy(static_cast<char*>(native.data.contents)+i*native.rowBytes,
            values+i*native.columns,native.columns*sizeof(float));
    matrix.impl_->ready=true;
}
void MetalContext::Begin() {
    @autoreleasepool {
    if(impl_->command)throw std::logic_error("Metal command batch already open");
    impl_->command=[impl_->queue commandBuffer];
    if(!impl_->command)throw std::runtime_error("Cannot create Metal command buffer");
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
    const MetalMatrix& output,const void* parameters,size_t parameterBytes,size_t threads) {
    @autoreleasepool {
    if(!impl_->command)throw std::logic_error("Begin a Metal command batch first");
    if(!output.impl_||!threads||threads>std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Invalid Metal state kernel output/grid");
    auto c=output.impl_->matrix;
    if(c.device!=impl_->device)throw std::invalid_argument("Metal output belongs to another device");
    if(output.impl_->pending && impl_->pendingOutputs.count(output.impl_.get())==0)
        throw std::logic_error("Metal output is in another command batch");
    for(const auto& input:inputs) {
        if(!input.impl_||input.impl_->matrix.device!=impl_->device)
            throw std::invalid_argument("Invalid Metal state kernel input/device");
        if(!input.impl_->ready && !(input.impl_->pending &&
            impl_->pendingOutputs.count(input.impl_.get())!=0))
            throw std::logic_error("Uninitialized Metal state kernel input");
        if(c.data==input.impl_->matrix.data)throw std::invalid_argument("In-place state kernel is unsupported");
    }
    if(operation>11)throw std::invalid_argument("Unknown Metal state kernel");
    id<MTLComputePipelineState> pipelines[]={impl_->kernels->conv,impl_->kernels->delta,impl_->kernels->slice,impl_->kernels->prepare,impl_->kernels->gatedNorm,
        impl_->kernels->concat,impl_->kernels->normRope,impl_->kernels->kvUpdate,impl_->kernels->scores,impl_->kernels->attentionContext,impl_->kernels->vision,impl_->kernels->layout};
    auto pipeline=pipelines[operation];
    auto encoder=[impl_->command computeCommandEncoder];
    if(!encoder)throw std::runtime_error("Cannot create Metal state kernel encoder");
    [encoder setComputePipelineState:pipeline];
    for(size_t i=0;i<inputs.size();++i)[encoder setBuffer:inputs[i].impl_->matrix.data offset:0 atIndex:i];
    [encoder setBuffer:c.data offset:0 atIndex:inputs.size()];
    [encoder setBytes:parameters length:parameterBytes atIndex:inputs.size()+1];
    [encoder dispatchThreads:MTLSizeMake(threads,1,1)
        threadsPerThreadgroup:MTLSizeMake(std::min(threads,size_t(pipeline.maxTotalThreadsPerThreadgroup)),1,1)];
    [encoder endEncoding];output.impl_->ready=false;output.impl_->pending=true;if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
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
    if(inputs.size()!=3||operation>9||!output.impl_||!tokens||!hidden||!heads||hidden%heads||!mergeUnit)
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
    EncodeStateKernel(2,{input},output,&p,sizeof(p),Elements(o.rows,o.columns));
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
void MetalContext::EncodeLinearPrepare(const MetalMatrix& conv,const MetalMatrix& a,const MetalMatrix& b,
    const MetalMatrix& aLog,const MetalMatrix& dtBias,const MetalMatrix& output,
    size_t keyHeads,size_t valueHeads,size_t keyDim,size_t valueDim,size_t kernel) {
    if(!conv.impl_||!a.impl_||!b.impl_||!aLog.impl_||!dtBias.impl_||!output.impl_||!keyHeads||!valueHeads||
        valueHeads%keyHeads||!keyDim||!valueDim||!kernel||keyHeads>128||valueHeads>128||keyDim>4096||valueDim>4096||kernel>64)
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
    const MetalMatrix& output,size_t heads,size_t valueDim,float epsilon) {
    if(!core.impl_||!z.impl_||!norm.impl_||!output.impl_||!(epsilon>0)||!std::isfinite(epsilon))
        throw std::invalid_argument("Invalid Metal gated normalization arguments");
    auto c=core.impl_->matrix,gate=z.impl_->matrix,w=norm.impl_->matrix,o=output.impl_->matrix;
    if(c.columns!=Elements(heads,valueDim)||gate.rows!=c.rows||gate.columns!=c.columns||w.rows!=1||w.columns!=valueDim||
        o.rows!=c.rows||o.columns!=c.columns)throw std::invalid_argument("Metal gated normalization shape mismatch");
    struct Params {uint64_t batch,heads,valueDim,cStride,zStride,oStride;float epsilon;};
    Params p{c.rows,heads,valueDim,c.rowBytes/4,gate.rowBytes/4,o.rowBytes/4,epsilon};
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
    EncodeStateKernel(7,{token,cache,slots,active},output,&p,sizeof(p),Elements(c.rows,width));
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
    const MetalMatrix& output,size_t heads,size_t kvHeads,size_t dim,size_t capacity,bool gated) {
    if(!probabilities.impl_||!value.impl_||!gate.impl_||!output.impl_||!kvHeads||!heads||heads%kvHeads)
        throw std::invalid_argument("Invalid Metal attention context arguments");
    auto p=probabilities.impl_->matrix,v=value.impl_->matrix,g=gate.impl_->matrix,o=output.impl_->matrix;
    if(p.columns!=Elements(heads,capacity)||v.rows!=p.rows||v.columns!=Elements(Elements(kvHeads,capacity),dim)||g.rows!=p.rows||
        g.columns!=Elements(heads,dim)||o.rows!=p.rows||o.columns!=g.columns)throw std::invalid_argument("Metal attention context shape mismatch");
    struct Params {uint64_t batch,heads,kvHeads,dim,capacity,pStride,vStride,gStride,oStride,gated;};
    Params params{p.rows,heads,kvHeads,dim,capacity,p.rowBytes/4,v.rowBytes/4,g.rowBytes/4,o.rowBytes/4,uint64_t(gated)};
    EncodeStateKernel(9,{probabilities,value,gate},output,&params,sizeof(params),Elements(o.rows,o.columns));
}
void MetalContext::SubmitAndWait() {
    @autoreleasepool {
    if(!impl_->command)throw std::logic_error("No Metal command batch to submit");
    auto command=impl_->command;
    [command commit];[command waitUntilCompleted];
    const bool success=command.status==MTLCommandBufferStatusCompleted;
    for(auto& output:impl_->outputs){output->pending=false;output->ready=success;}
    impl_->outputs.clear();impl_->pendingOutputs.clear();impl_->command=nil;
    if(!success)throw std::runtime_error(command.error ?
        std::string(command.error.localizedDescription.UTF8String):"Metal execution did not complete");
    }
}
void MetalContext::Abort() {
    for(auto& output:impl_->outputs){output->pending=false;output->ready=false;}
    impl_->outputs.clear();impl_->pendingOutputs.clear();impl_->command=nil;
}
std::vector<float> MetalContext::Download(const MetalMatrix& value) const {
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
