// SPDX-License-Identifier: Apache-2.0
// Standalone experiment: include after the unchanged qwen35_delta.cu reference.
// Four warps partition K while each block covers32value columns. Carries pass
// between warps in the original K order; there is no reassociated reduction.
#include <type_traits>
namespace Garnet { namespace {
// Fixed128-state variant: identical block/reduction layout to the reference,
// compile-time K loops so the per-thread state can stay in registers. It uses
// only the original convolution workspace and leaves the old kernel intact.
template<bool Volatile> __global__ void registerRecurrence128(Qwen35DeltaConfig c,int tokens,const float* conv,const float* z,const float* beta,
    const float* a,const float* old,const float* active,const float* dt,const float* alog,const float* norm,float* output,float* next){
    using Reduce=cub::BlockReduce<float,128>;
    __shared__ typename Reduce::TempStorage temp;
    using SharedFloat=typename std::conditional<Volatile,volatile float,float>::type;
    __shared__ SharedFloat q[128],k[128];__shared__ float factor;
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
__global__ void tiledControls128(Qwen35DeltaConfig c,int tokens,const float* conv,const float* beta,
    const float* a,const float* active,const float* dt,const float* alog,float* controls){
    using Reduce=cub::BlockReduce<float,128>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ float factor;
    const int lane=threadIdx.x,head=blockIdx.x%c.vh,b=blockIdx.x/c.vh;
    const int kh=head/(c.vh/c.kh),channels=2*c.kh*128+c.vh*128;
    for(int t=0;t<tokens;++t){
        const int bt=b*tokens+t,offset=bt*channels,base=(bt*c.vh+head)*258;
        float qv=conv[offset+kh*128+lane],kv=conv[offset+c.kh*128+kh*128+lane];
        float sum=Reduce(temp).Sum(qv*qv);
        if(lane==0)factor=1.0f/sqrtf(sum+1e-6f);__syncthreads();
        controls[base+lane]=__fmul_rn(__fmul_rn(qv,factor),1.0f/sqrtf(128.0f));__syncthreads();
        sum=Reduce(temp).Sum(kv*kv);
        if(lane==0)factor=1.0f/sqrtf(sum+1e-6f);__syncthreads();
        controls[base+128+lane]=kv*factor;
        if(lane==0){
            const float av=a[bt*c.vh+head]+dt[head];
            const float softplus=fmaxf(av,0.0f)+logf(1.0f+expf(-fabsf(av)));
            controls[base+256]=expf(-expf(alog[head])*softplus);
            controls[base+257]=1.0f/(1.0f+expf(-beta[bt*c.vh+head]));
        }
        __syncthreads();
    }
}
__global__ void tiledRecurrence128(Qwen35DeltaConfig c,int tokens,const float* conv,const float* controls,
    const float* old,const float* active,float* core,float* next){
    __shared__ float carry[32];
    const int lane=threadIdx.x%32,partition=threadIdx.x/32;
    const int tile=blockIdx.x%4,head=(blockIdx.x/4)%c.vh,b=blockIdx.x/(4*c.vh),column=tile*32+lane;
    const int channels=2*c.kh*128+c.vh*128;
    float state[32];
    #pragma unroll
    for(int j=0;j<32;++j)state[j]=old[((b*c.vh+head)*128+partition*32+j)*128+column];
    for(int t=0;t<tokens;++t){
        const int bt=b*tokens+t,base=(bt*c.vh+head)*258;
        if(active[bt]==0){if(partition==0)core[(bt*c.vh+head)*128+column]=0;continue;}
        const float decay=controls[base+256],gate=controls[base+257];
        #pragma unroll
        for(int j=0;j<32;++j)state[j]=__fmul_rn(state[j],decay);
        if(partition==0)carry[lane]=0;__syncthreads();
        #pragma unroll
        for(int part=0;part<4;++part){
            if(partition==part){
                float memory=carry[lane];
                #pragma unroll
                for(int j=0;j<32;++j)memory=__fadd_rn(memory,__fmul_rn(state[j],controls[base+128+part*32+j]));
                carry[lane]=memory;
            }
            __syncthreads();
        }
        const float value=conv[bt*channels+2*c.kh*128+head*128+column];
        const float delta=(value-carry[lane])*gate;
        __syncthreads();if(partition==0)carry[lane]=0;__syncthreads();
        #pragma unroll
        for(int part=0;part<4;++part){
            if(partition==part){
                float result=carry[lane];
                #pragma unroll
                for(int j=0;j<32;++j){
                    state[j]=__fadd_rn(state[j],__fmul_rn(controls[base+128+part*32+j],delta));
                    result=__fadd_rn(result,__fmul_rn(state[j],controls[base+part*32+j]));
                }
                carry[lane]=result;
            }
            __syncthreads();
        }
        if(partition==3)core[(bt*c.vh+head)*128+column]=carry[lane];
        __syncthreads();
    }
    #pragma unroll
    for(int j=0;j<32;++j)next[((b*c.vh+head)*128+partition*32+j)*128+column]=state[j];
}
__global__ void tiledOutput128(Qwen35DeltaConfig c,int tokens,const float* core,const float* z,
    const float* active,const float* norm,float* output){
    using Reduce=cub::BlockReduce<float,128>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ float factor;
    const int lane=threadIdx.x,head=blockIdx.x%c.vh,b=blockIdx.x/c.vh;
    for(int t=0;t<tokens;++t){
        const int bt=b*tokens+t,index=(bt*c.vh+head)*128+lane;
        if(active[bt]==0){output[index]=0;continue;}
        const float value=core[index];float sum=Reduce(temp).Sum(value*value);
        if(lane==0)factor=1.0f/sqrtf(sum/128.0f+c.epsilon);__syncthreads();
        output[index]=__fmul_rn(__fmul_rn(__fmul_rn(value,factor),norm[lane]),silu(z[index]));
        __syncthreads();
    }
}
}
size_t TiledDeltaWorkspace(const Qwen35DeltaConfig& c,int batch,int tokens){
    return size_t(batch)*tokens*(2*c.kh*128+c.vh*128+c.vh*258+c.vh*128)*sizeof(float);
}
int RunTiledDelta128(const Qwen35DeltaConfig& c,int batch,int tokens,const void* const* in,void* const* out,void* workspace,cudaStream_t stream){
    if(c.kd!=128||c.vd!=128||batch<1||batch>8||tokens<1||tokens>128||c.kh<1||c.vh<1||c.vh%c.kh||c.kernel<1||c.kernel>4||!workspace)return 1;
    const int channels=2*c.kh*128+c.vh*128;
    auto* conv=static_cast<float*>(workspace);auto* controls=conv+size_t(batch)*tokens*channels;
    auto* core=controls+size_t(batch)*tokens*c.vh*258;
    convolution<<<(batch*channels+127)/128,128,0,stream>>>(c,batch,tokens,(const float*)in[0],(const float*)in[4],(const float*)in[6],(const float*)in[7],conv,(float*)out[1]);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    tiledControls128<<<batch*c.vh,128,0,stream>>>(c,tokens,conv,(const float*)in[2],(const float*)in[3],(const float*)in[6],(const float*)in[8],(const float*)in[9],controls);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    tiledRecurrence128<<<batch*c.vh*4,128,0,stream>>>(c,tokens,conv,controls,(const float*)in[5],(const float*)in[6],core,(float*)out[2]);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    tiledOutput128<<<batch*c.vh,128,0,stream>>>(c,tokens,core,(const float*)in[1],(const float*)in[6],(const float*)in[10],(float*)out[0]);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
int RunRegisterDelta128(const Qwen35DeltaConfig& c,int batch,int tokens,const void* const* in,void* const* out,void* workspace,cudaStream_t stream,bool volatileShared=false){
    if(c.kd!=128||c.vd!=128||batch<1||batch>8||tokens<1||tokens>128||c.kh<1||c.vh<1||c.vh%c.kh||c.kernel<1||c.kernel>4||!workspace)return 1;
    const int channels=2*c.kh*128+c.vh*128;
    convolution<<<(batch*channels+127)/128,128,0,stream>>>(c,batch,tokens,(const float*)in[0],(const float*)in[4],(const float*)in[6],(const float*)in[7],(float*)workspace,(float*)out[1]);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    if(volatileShared)registerRecurrence128<true><<<batch*c.vh,128,0,stream>>>(c,tokens,(float*)workspace,(const float*)in[1],(const float*)in[2],(const float*)in[3],(const float*)in[5],(const float*)in[6],(const float*)in[8],(const float*)in[9],(const float*)in[10],(float*)out[0],(float*)out[2]);
    else registerRecurrence128<false><<<batch*c.vh,128,0,stream>>>(c,tokens,(float*)workspace,(const float*)in[1],(const float*)in[2],(const float*)in[3],(const float*)in[5],(const float*)in[6],(const float*)in[8],(const float*)in[9],(const float*)in[10],(float*)out[0],(float*)out[2]);
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
