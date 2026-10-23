// SPDX-License-Identifier: Apache-2.0
#include "qwen35_indexed_head.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <math_constants.h>
#include <cstdint>
namespace Garnet { namespace {
struct Best {float value;int id;};
struct Better {
    __device__ Best operator()(Best a,Best b) const {
        if(a.id<0)return b;if(b.id<0)return a;
        return b.value>a.value||(b.value==a.value&&b.id<a.id)?b:a;
    }
};
__global__ void indexedDot(Qwen35IndexedHeadConfig c,const float* query,const __nv_bfloat16* table,
    const int* indices,__nv_bfloat16* scores){
    int item=blockIdx.x,row=item/c.candidates,id=indices[item];
    if(id<0||id>=c.vocab){if(!threadIdx.x)scores[item]=__float2bfloat16_rn(-CUDART_INF_F);return;}
    float sum=0;
    for(int d=threadIdx.x;d<c.hidden;d+=256){
        float x=__bfloat162float(__float2bfloat16_rn(query[size_t(row)*c.hidden+d]));
        sum=fmaf(x,__bfloat162float(table[size_t(id)*c.hidden+d]),sum);
    }
    using Reduce=cub::BlockReduce<float,256>;__shared__ typename Reduce::TempStorage temporary;
    sum=Reduce(temporary).Sum(sum);
    if(!threadIdx.x)scores[item]=__float2bfloat16_rn(sum);
}
__global__ void indexedBest(Qwen35IndexedHeadConfig c,const int* indices,const __nv_bfloat16* scores,
    long long* ids,float* values){
    int row=blockIdx.x;Best best{-CUDART_INF_F,-1};Better better;
    for(int i=threadIdx.x;i<c.candidates;i+=256){
        int id=indices[row*c.candidates+i];float value=__bfloat162float(scores[row*c.candidates+i]);
        if(id>=0&&id<c.vocab&&isfinite(value))best=better(best,{value,id});
    }
    using Reduce=cub::BlockReduce<Best,256>;__shared__ typename Reduce::TempStorage temporary;
    best=Reduce(temporary).Reduce(best,better);
    if(!threadIdx.x){ids[row]=best.id;if(values)values[row]=best.value;}
}
struct Range{uintptr_t pointer;size_t bytes;};
bool overlap(Range a,Range b){
    if(!a.pointer||!b.pointer)return false;
    return a.pointer>=b.pointer?a.pointer-b.pointer<b.bytes:b.pointer-a.pointer<a.bytes;
}
__global__ void candidateKeys(const __nv_bfloat16* logits,float* keys,int* ids,int vocab){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=vocab)return;
    float value=__bfloat162float(logits[i]);keys[i]=isfinite(value)?value:-CUDART_INF_F;ids[i]=i;
}
__global__ void candidateCopy(const float* keys,const int* ids,int* output,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=isfinite(keys[i])?ids[i]:-1;
}
size_t aligned(size_t n){return (n+255)&~size_t(255);}
size_t sortWorkspace(int vocab){
    size_t bytes=0;
    if(cub::DeviceRadixSort::SortPairsDescending(nullptr,bytes,(float*)nullptr,(float*)nullptr,(int*)nullptr,(int*)nullptr,vocab)!=cudaSuccess)return 0;
    return bytes;
}
}}
namespace Garnet {
int RunQwen35IndexedHead(const Qwen35IndexedHeadConfig& c,const float* query,const void* table,const int* indices,
    void* scores,long long* ids,float* values,cudaStream_t stream) noexcept {
    if(c.rows<1||c.rows>8||c.hidden<1||c.hidden>16384||c.vocab<1||c.candidates<1||c.candidates>65536||c.candidates>c.vocab||
        !query||!table||!indices||!scores||!ids)return 1;
    Range ranges[]={{reinterpret_cast<uintptr_t>(query),size_t(c.rows)*c.hidden*4},
        {reinterpret_cast<uintptr_t>(table),size_t(c.vocab)*c.hidden*2},
        {reinterpret_cast<uintptr_t>(indices),size_t(c.rows)*c.candidates*4},
        {reinterpret_cast<uintptr_t>(scores),size_t(c.rows)*c.candidates*2},
        {reinterpret_cast<uintptr_t>(ids),size_t(c.rows)*8},
        {reinterpret_cast<uintptr_t>(values),size_t(c.rows)*4}};
    for(int out=3;out<6;++out)for(int prior=0;prior<out;++prior)if(overlap(ranges[out],ranges[prior]))return 1;
    indexedDot<<<c.rows*c.candidates,256,0,stream>>>(c,query,(const __nv_bfloat16*)table,indices,(__nv_bfloat16*)scores);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    indexedBest<<<c.rows,256,0,stream>>>(c,indices,(const __nv_bfloat16*)scores,ids,values);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
size_t Qwen35HeadCandidatesWorkspace(int vocab) noexcept {
    if(vocab<1||vocab>1048576)return 0;auto bytes=sortWorkspace(vocab);
    return bytes?4*aligned(size_t(vocab)*4)+bytes:0;
}
int RunQwen35HeadCandidates(int rows,int sourceRows,int vocab,int count,const int* selectedRows,
    const void* logits,int* output,void* workspace,size_t bytes,cudaStream_t stream) noexcept {
    if(rows<1||rows>8||sourceRows<1||sourceRows>128||vocab<1||vocab>1048576||count<1||count>vocab||count>65536||
        !selectedRows||!logits||!output||!workspace)return 1;
    for(int row=0;row<rows;++row)if(selectedRows[row]<0||selectedRows[row]>=sourceRows)return 1;
    auto required=Qwen35HeadCandidatesWorkspace(vocab);if(!required||bytes<required)return 1;
    Range in{reinterpret_cast<uintptr_t>(logits),size_t(sourceRows)*vocab*2},out{reinterpret_cast<uintptr_t>(output),size_t(rows)*count*4},
        scratch{reinterpret_cast<uintptr_t>(workspace),required};
    if(overlap(in,out)||overlap(in,scratch)||overlap(out,scratch))return 1;
    size_t stride=aligned(size_t(vocab)*4);auto* base=(char*)workspace;
    auto* keys=(float*)base;auto* sortedKeys=(float*)(base+stride);auto* ids=(int*)(base+2*stride);auto* sortedIds=(int*)(base+3*stride);
    size_t temporaryBytes=bytes-4*stride;
    for(int row=0;row<rows;++row){
        candidateKeys<<<(vocab+255)/256,256,0,stream>>>((const __nv_bfloat16*)logits+size_t(selectedRows[row])*vocab,keys,ids,vocab);
        if(cudaPeekAtLastError()!=cudaSuccess)return 1;
        if(cub::DeviceRadixSort::SortPairsDescending(base+4*stride,temporaryBytes,keys,sortedKeys,ids,sortedIds,vocab,0,32,stream)!=cudaSuccess)return 1;
        candidateCopy<<<(count+255)/256,256,0,stream>>>(sortedKeys,sortedIds,output+size_t(row)*count,count);
        if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    }return 0;
}
}
