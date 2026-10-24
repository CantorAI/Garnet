// SPDX-License-Identifier: Apache-2.0
// Existing FP32 attention path and a separate unwired compact research candidate.
// No fast-math or lower precision reductions. Serving changes require GPU gates.
#include "qwen35_attention_core.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
namespace Garnet {
namespace {
bool valid(Qwen35AttentionCoreConfig c) {
    return c.batch>0&&c.batch<=8&&c.tokens>0&&c.tokens<=128&&c.heads>0&&c.heads<=32&&
        c.kv>0&&c.heads%c.kv==0&&c.capacity>0&&c.capacity<=8192&&
        c.dim>0&&c.dim<=256&&c.dim%32==0;
}
__device__ float warpSum(float v){
    for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);return v;
}
__device__ float blockReduce(float v,bool maximum,float* temporary,float* result){
    for(int d=16;d;d>>=1){float other=__shfl_down_sync(0xffffffff,v,d);v=maximum?fmaxf(v,other):v+other;}
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    if(!lane)temporary[warp]=v;
    __syncthreads();
    if(!warp){
        v=lane<8?temporary[lane]:(maximum?-CUDART_INF_F:0.f);
        for(int d=16;d;d>>=1){float other=__shfl_down_sync(0xffffffff,v,d);v=maximum?fmaxf(v,other):v+other;}
        if(!lane)*result=v;
    }
    __syncthreads();return *result;
}
__global__ void writeKeys(Qwen35AttentionCoreConfig c,const float* key,const float* value,
    const float* slots,const float* active,float* nextKey,float* nextValue){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=c.batch*c.tokens*c.kv*c.dim)return;
    int d=i%c.dim,head=(i/c.dim)%c.kv,bt=i/(c.kv*c.dim),b=bt/c.tokens;
    if(active[bt]==0)return;
    int slot=int(slots[bt]);if(slot<0||slot>=c.capacity)return;
    size_t target=((size_t(b)*c.kv+head)*c.capacity+slot)*c.dim+d;
    nextKey[target]=key[i];nextValue[target]=value[i];
}
// Each warp computes one key score; independent key tiles expose parallelism
// even at batch1. Read KV directly with head mapping, never repeat-interleave it.
__global__ void scores(Qwen35AttentionCoreConfig c,const float* query,const float* key,
    const float* slots,const float* active,float* scratch){
    int queryRow=blockIdx.y,b=queryRow/(c.tokens*c.heads),t=(queryRow/c.heads)%c.tokens;
    if(active[b*c.tokens+t]==0)return;
    int head=queryRow%c.heads,kh=head/(c.heads/c.kv),lane=threadIdx.x%32,warp=threadIdx.x/32;
    int position=blockIdx.x*8+warp,slot=int(slots[b*c.tokens+t]);
    if(position>=c.capacity)return; // warp-uniform branch
    float sum=0;
    if(position<=slot){
        size_t kbase=((size_t(b)*c.kv+kh)*c.capacity+position)*c.dim;
        for(int d=lane;d<c.dim;d+=32)sum=fmaf(query[size_t(queryRow)*c.dim+d],key[kbase+d],sum);
        sum=warpSum(sum);
    }
    if(!lane)scratch[size_t(queryRow)*c.capacity+position]=position<=slot?sum*rsqrtf(float(c.dim)):-1e30f;
}
__global__ void normalize(Qwen35AttentionCoreConfig c,const float* active,float* scratch){
    int row=blockIdx.x,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
    if(active[b*c.tokens+t]==0)return;
    __shared__ float temporary[8],result;float maximum=-CUDART_INF_F;
    float* values=scratch+size_t(row)*c.capacity;
    for(int k=threadIdx.x;k<c.capacity;k+=256)maximum=fmaxf(maximum,values[k]);
    maximum=blockReduce(maximum,true,temporary,&result);
    // All threads finish reading the max before reusing shared reduction data.
    __syncthreads();float sum=0;
    for(int k=threadIdx.x;k<c.capacity;k+=256){float v=expf(values[k]-maximum);values[k]=v;sum+=v;}
    sum=blockReduce(sum,false,temporary,&result);
    for(int k=threadIdx.x;k<c.capacity;k+=256)values[k]/=sum;
}
// A block produces32 context dimensions. Warps split keys and lanes split D,
// keeping V loads coalesced; shared partial sums avoid atomic accumulation.
__global__ void context(Qwen35AttentionCoreConfig c,const float* value,const float* gate,
    const float* slots,const float* active,const float* scratch,float* output){
    int row=blockIdx.y,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
    int lane=threadIdx.x%32,warp=threadIdx.x/32,d=blockIdx.x*32+lane;
    if(active[b*c.tokens+t]==0){if(!warp)output[size_t(row)*c.dim+d]=0;return;}
    int head=row%c.heads,kh=head/(c.heads/c.kv);float sum=0;
    int end=min(c.capacity,int(slots[b*c.tokens+t])+1);
    for(int k=warp;k<end;k+=8){
        float weight=scratch[size_t(row)*c.capacity+k];
        sum=fmaf(weight,value[((size_t(b)*c.kv+kh)*c.capacity+k)*c.dim+d],sum);
    }
    __shared__ float partial[8][32];partial[warp][lane]=sum;__syncthreads();
    if(!warp){
        sum=0;for(int w=0;w<8;++w)sum+=partial[w][lane];
        output[size_t(row)*c.dim+d]=sum/(1.f+expf(-gate[size_t(row)*c.dim+d]));
    }
}
// Research candidate: one block per query, scores and softmax only over visible keys.
// Keep the original three-stage kernel available for correctness/performance A/B.
__global__ void compact(Qwen35AttentionCoreConfig c,const float* query,const float* key,
 const float* value,const float* gate,const float* slots,const float* active,float* output){
 const int row=blockIdx.x,b=row/(c.tokens*c.heads),t=(row/c.heads)%c.tokens;
 if(active[b*c.tokens+t]==0){if(threadIdx.x<c.dim)output[size_t(row)*c.dim+threadIdx.x]=0;return;}
 const int end=min(c.capacity,int(slots[b*c.tokens+t])+1);
 if(end<1){if(threadIdx.x<c.dim)output[size_t(row)*c.dim+threadIdx.x]=0;return;}
 const int head=row%c.heads,kh=head/(c.heads/c.kv),lane=threadIdx.x%32,warp=threadIdx.x/32;
 __shared__ float values[8192],temporary[8],result,partial[8][32];
 for(int position=warp;position<end;position+=8){
  float sum=0;const size_t base=((size_t(b)*c.kv+kh)*c.capacity+position)*c.dim;
  for(int d=lane;d<c.dim;d+=32)sum=fmaf(query[size_t(row)*c.dim+d],key[base+d],sum);
  sum=warpSum(sum);if(!lane)values[position]=sum*rsqrtf(float(c.dim));
 }
 __syncthreads();float maximum=-CUDART_INF_F;
 for(int k=threadIdx.x;k<end;k+=256)maximum=fmaxf(maximum,values[k]);
 maximum=blockReduce(maximum,true,temporary,&result);__syncthreads();float total=0;
 for(int k=threadIdx.x;k<end;k+=256){float v=expf(values[k]-maximum);values[k]=v;total+=v;}
 total=blockReduce(total,false,temporary,&result);
 for(int k=threadIdx.x;k<end;k+=256)values[k]/=total;
 __syncthreads();
 for(int tile=0;tile<c.dim/32;++tile){
  const int d=tile*32+lane;float sum=0;
  for(int k=warp;k<end;k+=8)sum=fmaf(values[k],value[((size_t(b)*c.kv+kh)*c.capacity+k)*c.dim+d],sum);
  partial[warp][lane]=sum;__syncthreads();
  if(!warp){sum=0;for(int w=0;w<8;++w)sum+=partial[w][lane];output[size_t(row)*c.dim+d]=sum/(1.f+expf(-gate[size_t(row)*c.dim+d]));}
  __syncthreads();
 }
}
}
size_t Qwen35AttentionCoreWorkspace(const Qwen35AttentionCoreConfig& c) noexcept {
    return valid(c)?size_t(c.batch)*c.tokens*c.heads*c.capacity*sizeof(float):0;
}
int RunQwen35AttentionCore(const Qwen35AttentionCoreConfig& c,const void* const* in,
    void* const* out,void* workspace,cudaStream_t stream) noexcept {
    if(!valid(c)||!in||!out||!workspace)return 1;
    for(int i=0;i<8;++i)if(!in[i])return 1;
    for(int i=0;i<3;++i)if(!out[i])return 1;
    size_t bytes=size_t(c.batch)*c.kv*c.capacity*c.dim*sizeof(float);
    if(cudaMemcpyAsync(out[1],in[4],bytes,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess||
        cudaMemcpyAsync(out[2],in[5],bytes,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 1;
    int updates=c.batch*c.tokens*c.kv*c.dim,rows=c.batch*c.tokens*c.heads;
    writeKeys<<<(updates+255)/256,256,0,stream>>>(c,(const float*)in[1],(const float*)in[2],
        (const float*)in[6],(const float*)in[7],(float*)out[1],(float*)out[2]);
    scores<<<dim3((c.capacity+7)/8,rows),256,0,stream>>>(c,(const float*)in[0],(const float*)out[1],
        (const float*)in[6],(const float*)in[7],(float*)workspace);
    normalize<<<rows,256,0,stream>>>(c,(const float*)in[7],(float*)workspace);
    context<<<dim3(c.dim/32,rows),256,0,stream>>>(c,(const float*)out[2],(const float*)in[3],
        (const float*)in[6],(const float*)in[7],(const float*)workspace,(float*)out[0]);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
int RunQwen35AttentionCompact(const Qwen35AttentionCoreConfig& c,const void* const* in,
 void* const* out,void*,cudaStream_t stream) noexcept {
 if(!valid(c)||!in||!out)return 1;
 for(int i=0;i<8;++i)if(!in[i])return 1;
 for(int i=0;i<3;++i)if(!out[i])return 1;
 size_t bytes=size_t(c.batch)*c.kv*c.capacity*c.dim*sizeof(float);
 if(cudaMemcpyAsync(out[1],in[4],bytes,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess||
    cudaMemcpyAsync(out[2],in[5],bytes,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)return 1;
 int updates=c.batch*c.tokens*c.kv*c.dim,rows=c.batch*c.tokens*c.heads;
 writeKeys<<<(updates+255)/256,256,0,stream>>>(c,(const float*)in[1],(const float*)in[2],(const float*)in[6],(const float*)in[7],(float*)out[1],(float*)out[2]);
 compact<<<rows,256,0,stream>>>(c,(const float*)in[0],(const float*)out[1],(const float*)out[2],(const float*)in[3],(const float*)in[6],(const float*)in[7],(float*)out[0]);
 return cudaPeekAtLastError()==cudaSuccess?0:1;
}

}
