// SPDX-License-Identifier: Apache-2.0
#pragma once
// Included after qwen35_delta.cu's unchanged convolution/reference kernels.
// Fixed K/V dimensions let each thread retain its recurrent state in registers.
// Volatile shared Q/K prevents hoisting both128-entry vectors across the state
// lifetime. FP32 accumulation and reductions retain the reference order.
namespace Garnet { namespace {
__global__ void registerState128Kernel(Qwen35DeltaConfig c,int tokens,const float* conv,const float* z,const float* beta,
    const float* a,const float* old,const float* active,const float* dt,const float* alog,const float* norm,float* output,float* next){
    using Reduce=cub::BlockReduce<float,128>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ volatile float q[128],k[128];__shared__ float factor;
    const int lane=threadIdx.x,head=blockIdx.x%c.vh,b=blockIdx.x/c.vh;
    const int kh=head/(c.vh/c.kh),channels=2*c.kh*128+c.vh*128;
    float state[128];
    #pragma unroll
    for(int d=0;d<128;++d)state[d]=old[((b*c.vh+head)*128+d)*128+lane];
    for(int t=0;t<tokens;++t){
        const int bt=b*tokens+t,offset=bt*channels;
        if(active[bt]==0){output[(bt*c.vh+head)*128+lane]=0;continue;}
        float qv=conv[offset+kh*128+lane],kv=conv[offset+c.kh*128+kh*128+lane];
        float sum=Reduce(temp).Sum(qv*qv);
        if(lane==0)factor=1.0f/sqrtf(sum+1e-6f);__syncthreads();
        q[lane]=__fmul_rn(__fmul_rn(qv,factor),1.0f/sqrtf(128.0f));__syncthreads();
        sum=Reduce(temp).Sum(kv*kv);
        if(lane==0)factor=1.0f/sqrtf(sum+1e-6f);__syncthreads();
        k[lane]=kv*factor;__syncthreads();
        const float av=a[bt*c.vh+head]+dt[head];
        const float softplus=fmaxf(av,0.0f)+logf(1.0f+expf(-fabsf(av)));
        const float decay=expf(-expf(alog[head])*softplus),gate=1.0f/(1.0f+expf(-beta[bt*c.vh+head]));
        float memory=0;
        #pragma unroll
        for(int d=0;d<128;++d){state[d]=__fmul_rn(state[d],decay);memory=__fadd_rn(memory,__fmul_rn(state[d],k[d]));}
        const float value=conv[offset+2*c.kh*128+head*128+lane],delta=(value-memory)*gate;float core=0;
        #pragma unroll
        for(int d=0;d<128;++d){state[d]=__fadd_rn(state[d],__fmul_rn(k[d],delta));core=__fadd_rn(core,__fmul_rn(state[d],q[d]));}
        sum=Reduce(temp).Sum(core*core);
        if(lane==0)factor=1.0f/sqrtf(sum/128.0f+c.epsilon);__syncthreads();
        output[(bt*c.vh+head)*128+lane]=__fmul_rn(__fmul_rn(__fmul_rn(core,factor),norm[lane]),silu(z[(bt*c.vh+head)*128+lane]));__syncthreads();
    }
    #pragma unroll
    for(int d=0;d<128;++d)next[((b*c.vh+head)*128+d)*128+lane]=state[d];
}
}
int RunQwen35DeltaRegister128(const Qwen35DeltaConfig& c,int batch,int tokens,const void* const* in,void* const* out,void* workspace,cudaStream_t stream) noexcept {
    if(c.kd!=128||c.vd!=128||batch<1||batch>8||tokens<1||tokens>128||c.kh<1||c.kh>64||c.vh<1||c.vh>64||c.vh%c.kh||c.kernel<1||c.kernel>4||!(c.epsilon>0)||!in||!out||!workspace)return 1;
    for(int i=0;i<11;++i)if(!in[i])return 1;for(int i=0;i<3;++i)if(!out[i])return 1;
    const int channels=2*c.kh*128+c.vh*128;
    convolution<<<(batch*channels+127)/128,128,0,stream>>>(c,batch,tokens,(const float*)in[0],(const float*)in[4],(const float*)in[6],(const float*)in[7],(float*)workspace,(float*)out[1]);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    registerState128Kernel<<<batch*c.vh,128,0,stream>>>(c,tokens,(float*)workspace,(const float*)in[1],(const float*)in[2],(const float*)in[3],(const float*)in[5],(const float*)in[6],(const float*)in[8],(const float*)in[9],(const float*)in[10],(float*)out[0],(float*)out[2]);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
