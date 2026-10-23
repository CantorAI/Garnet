// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <cuda_bf16.h>
namespace Garnet {
namespace {
__global__ void CopyActive(const float* input,const float* active,float* output,int rows,int width){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(rows)*width&&active[i/width]!=0)output[i]=input[i];
}
__global__ void CopyLast(const float* input,const float* active,float* output,int rows,int tokens,int width){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*width)return;
    int row=i/width;
    for(int t=tokens-1;t>=0;--t)if(active[row*tokens+t]!=0){output[i]=input[(size_t(row)*tokens+t)*width+i%width];break;}
}
struct RequestPayload {float active[128],position[384],slots[128];};
__global__ void WriteRequest(float* active,float* position,float* slots,RequestPayload values,int start,int count){
    int i=threadIdx.x;if(i>=count)return;active[start+i]=values.active[i];slots[start+i]=values.slots[i];
    for(int axis=0;axis<3;++axis)position[(start+i)*3+axis]=values.position[i*3+axis];
}
struct EmbeddingPayload {int ids[1024];};
__global__ void GatherChunk(const __nv_bfloat16* table,float* output,EmbeddingPayload values,int rows,int tokens,int count,int width){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(rows)*tokens*width)return;
    int row=i/(tokens*width),token=(i/width)%tokens,column=i%width;
    output[i]=token<count?__bfloat162float(table[size_t(values.ids[token*rows+row])*width+column]):0;
}
}
cudaError_t WriteQwen35Request(float* active,float* position,float* slots,const float* flags,const float* positions,const float* indices,int count,cudaStream_t stream){
    if(!active||!position||!slots||!flags||!positions||!indices||count<1||count>1024)return cudaErrorInvalidValue;
    for(int start=0;start<count;start+=128){
        RequestPayload payload{};int n=(count-start)<128?count-start:128;
        for(int i=0;i<n;++i){payload.active[i]=flags[start+i];payload.slots[i]=indices[start+i];for(int axis=0;axis<3;++axis)payload.position[3*i+axis]=positions[3*(start+i)+axis];}
        WriteRequest<<<1,128,0,stream>>>(active,position,slots,payload,start,n);auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
    }
    return cudaSuccess;
}
cudaError_t GatherQwen35Chunk(const void* table,float* output,const int* ids,int rows,int tokens,int count,int width,cudaStream_t stream){
    if(!table||!output||!ids||(rows!=1&&rows!=2&&rows!=4&&rows!=8)||tokens<1||tokens>128||count<1||count>tokens||width<1||width>16384)return cudaErrorInvalidValue;
    EmbeddingPayload payload{};for(int i=0;i<rows*count;++i)payload.ids[i]=ids[i];
    GatherChunk<<<(size_t(rows)*tokens*width+255)/256,256,0,stream>>>((const __nv_bfloat16*)table,output,payload,rows,tokens,count,width);return cudaGetLastError();
}
cudaError_t CopyQwen35ActiveRows(const float* input,const float* active,float* output,int rows,int width,cudaStream_t stream){
    if(!input||!active||!output||rows<=0||width<=0)return cudaErrorInvalidValue;
    CopyActive<<<(size_t(rows)*width+255)/256,256,0,stream>>>(input,active,output,rows,width);return cudaGetLastError();
}
cudaError_t CopyQwen35LastChunk(const float* input,const float* active,float* output,int rows,int tokens,int width,cudaStream_t stream){
    if(!input||!active||!output||rows<=0||tokens<=0||width<=0)return cudaErrorInvalidValue;
    CopyLast<<<(size_t(rows)*width+255)/256,256,0,stream>>>(input,active,output,rows,tokens,width);return cudaGetLastError();
}
}
