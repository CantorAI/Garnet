// SPDX-License-Identifier: Apache-2.0
#include "../backends/tensorrt/qwen35_delta_plugin.h"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
namespace Garnet {
bool Qwen35WholeProgramCUDA() noexcept {
#ifdef __CUDACC_RDC__
    return false;
#else
    return true;
#endif
}
namespace {
__device__ float silu(float x){return x/(1.0f+expf(-x));}
__global__ void convolution(Qwen35DeltaConfig c,int batch,int tokens,const float* input,const float* old,
    const float* active,const float* weights,float* conv,float* next){
    const int channels=2*c.kh*c.kd+c.vh*c.vd;
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=batch*channels)return;
    int b=index/channels,ch=index%channels;float h[4];
    for(int k=0;k<c.kernel;++k)h[k]=old[index*c.kernel+k];
    for(int t=0;t<tokens;++t){
        float v=0;
        if(active[b*tokens+t]!=0){
            for(int k=0;k<c.kernel-1;++k)h[k]=h[k+1];
            h[c.kernel-1]=input[(b*tokens+t)*channels+ch];
            for(int k=0;k<c.kernel;++k)v=__fadd_rn(v,__fmul_rn(h[k],weights[ch*c.kernel+k]));
            v=silu(v);
        }
        conv[(b*tokens+t)*channels+ch]=v;
    }
    for(int k=0;k<c.kernel;++k)next[index*c.kernel+k]=h[k];
}
__global__ void recurrence(Qwen35DeltaConfig c,int tokens,const float* conv,const float* z,const float* beta,
    const float* a,const float* old,const float* active,const float* dt,const float* alog,const float* norm,
    float* output,float* next){
    using Reduce=cub::BlockReduce<float,128>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ float q[128],k[128],factor;
    const int lane=threadIdx.x,head=blockIdx.x%c.vh,b=blockIdx.x/c.vh;
    const int kh=head/(c.vh/c.kh),channels=2*c.kh*c.kd+c.vh*c.vd;
    float state[128];
    for(int d=0;d<c.kd;++d)state[d]=lane<c.vd?old[((b*c.vh+head)*c.kd+d)*c.vd+lane]:0;
    for(int t=0;t<tokens;++t){
        const int bt=b*tokens+t,offset=bt*channels;
        if(active[bt]==0){if(lane<c.vd)output[(bt*c.vh+head)*c.vd+lane]=0;continue;}
        float qv=lane<c.kd?conv[offset+kh*c.kd+lane]:0;
        float kv=lane<c.kd?conv[offset+c.kh*c.kd+kh*c.kd+lane]:0;
        float sum=Reduce(temp).Sum(qv*qv);
        if(lane==0)factor=1.0f/sqrtf(sum+1e-6f);__syncthreads();
        q[lane]=__fmul_rn(__fmul_rn(qv,factor),1.0f/sqrtf(float(c.kd)));__syncthreads();
        sum=Reduce(temp).Sum(kv*kv);
        if(lane==0)factor=1.0f/sqrtf(sum+1e-6f);__syncthreads();
        k[lane]=kv*factor;__syncthreads();
        const float av=a[bt*c.vh+head]+dt[head];
        const float softplus=fmaxf(av,0.0f)+logf(1.0f+expf(-fabsf(av)));
        const float decay=expf(-expf(alog[head])*softplus);
        const float gate=1.0f/(1.0f+expf(-beta[bt*c.vh+head]));
        float memory=0;
        for(int d=0;d<c.kd;++d){state[d]=__fmul_rn(state[d],decay);memory=__fadd_rn(memory,__fmul_rn(state[d],k[d]));}
        const float value=lane<c.vd?conv[offset+2*c.kh*c.kd+head*c.vd+lane]:0;
        const float delta=(value-memory)*gate;float core=0;
        for(int d=0;d<c.kd;++d){state[d]=__fadd_rn(state[d],__fmul_rn(k[d],delta));core=__fadd_rn(core,__fmul_rn(state[d],q[d]));}
        sum=Reduce(temp).Sum(lane<c.vd?core*core:0);
        if(lane==0)factor=1.0f/sqrtf(sum/float(c.vd)+c.epsilon);__syncthreads();
        if(lane<c.vd)output[(bt*c.vh+head)*c.vd+lane]=__fmul_rn(__fmul_rn(__fmul_rn(core,factor),norm[lane]),silu(z[(bt*c.vh+head)*c.vd+lane]));
        __syncthreads();
    }
    for(int d=0;d<c.kd;++d)if(lane<c.vd)next[((b*c.vh+head)*c.kd+d)*c.vd+lane]=state[d];
}
}
int RunQwen35Delta(const Qwen35DeltaConfig& c,int batch,int tokens,const void* const* in,void* const* out,void* workspace,cudaStream_t stream) noexcept {
    if(batch<1||batch>8||tokens<1||tokens>128||c.kh<1||c.vh<1||c.vh%c.kh||c.kd<1||c.kd>128||c.vd<1||c.vd>128||c.kernel<1||c.kernel>4||!workspace)return 1;
    const int channels=2*c.kh*c.kd+c.vh*c.vd;
    convolution<<<(batch*channels+127)/128,128,0,stream>>>(c,batch,tokens,(const float*)in[0],(const float*)in[4],(const float*)in[6],(const float*)in[7],(float*)workspace,(float*)out[1]);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    recurrence<<<batch*c.vh,128,0,stream>>>(c,tokens,(float*)workspace,(const float*)in[1],(const float*)in[2],(const float*)in[3],(const float*)in[5],(const float*)in[6],(const float*)in[8],(const float*)in[9],(const float*)in[10],(float*)out[0],(float*)out[2]);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
#include "qwen35_delta_register128.cuh"
