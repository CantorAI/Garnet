// SPDX-License-Identifier: Apache-2.0
#include "qwen35_packed_commit.cuh"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <initializer_list>
namespace Garnet { namespace {
__global__ void packedCommitConvolution(Qwen35PackedCommitConfig c,Qwen35PackedCommitSpans spans,
    const float* packed,float* state,const Qwen35PackedLinearLayout* layout,const float* parameters,float* conv){
    int channels=2*c.key_heads*128+c.value_heads*128;
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=c.batch*channels)return;
    int layer=blockIdx.y,b=i/channels,ch=i%channels;
    auto p=layout[layer];int stride=channels*c.kernel+2*c.value_heads;
    const float* weights=parameters+size_t(layer)*stride;
    float h[4];for(int k=0;k<c.kernel;++k)h[k]=state[size_t(b)*c.state_width+p.state_offset+ch*c.kernel+k];
    for(int t=0;t<c.tokens;++t){
        float v=0;
        if(t<spans.count[b]){
            for(int k=0;k<c.kernel-1;++k)h[k]=h[k+1];
            h[c.kernel-1]=packed[size_t(b)*c.packed_width+p.payload_offset+t*channels+ch];
            for(int k=0;k<c.kernel;++k)v=__fadd_rn(v,__fmul_rn(h[k],weights[ch*c.kernel+k]));
            v=v/(1.0f+expf(-v));
        }
        conv[((size_t(layer)*c.batch+b)*c.tokens+t)*channels+ch]=v;
    }
    for(int k=0;k<c.kernel;++k)state[size_t(b)*c.state_width+p.state_offset+ch*c.kernel+k]=h[k];
}
__global__ void packedCommitRecurrent(Qwen35PackedCommitConfig c,Qwen35PackedCommitSpans spans,
    const float* packed,float* canonical,const Qwen35PackedLinearLayout* layout,const float* parameters,const float* conv){
    using Reduce=cub::BlockReduce<float,128>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ volatile float k[128];__shared__ float factor;
    int lane=threadIdx.x,head=blockIdx.x%c.value_heads,b=blockIdx.x/c.value_heads,layer=blockIdx.y;
    int kh=head/(c.value_heads/c.key_heads),channels=2*c.key_heads*128+c.value_heads*128;
    auto p=layout[layer];int stride=channels*c.kernel+2*c.value_heads;
    const float* dt=parameters+size_t(layer)*stride+channels*c.kernel;
    const float* alog=dt+c.value_heads;
    int64_t stateBase=size_t(b)*c.state_width+p.state_offset+channels*c.kernel+head*128*128;
    float state[128];
    #pragma unroll
    for(int d=0;d<128;++d)state[d]=canonical[stateBase+d*128+lane];
    for(int t=0;t<spans.count[b];++t){
        const float* cv=conv+((size_t(layer)*c.batch+b)*c.tokens+t)*channels;
        float kv=cv[c.key_heads*128+kh*128+lane];
        float sum=Reduce(temp).Sum(kv*kv);
        if(!lane)factor=1.0f/sqrtf(sum+1e-6f);__syncthreads();
        k[lane]=kv*factor;__syncthreads();
        int64_t betaBase=size_t(b)*c.packed_width+p.payload_offset+c.tokens*(channels+c.value_heads*128);
        float av=packed[betaBase+c.tokens*c.value_heads+t*c.value_heads+head]+dt[head];
        float softplus=fmaxf(av,0.0f)+logf(1.0f+expf(-fabsf(av)));
        float decay=expf(-expf(alog[head])*softplus),gate=1.0f/(1.0f+expf(-packed[betaBase+t*c.value_heads+head]));
        float memory=0;
        #pragma unroll
        for(int d=0;d<128;++d){state[d]=__fmul_rn(state[d],decay);memory=__fadd_rn(memory,__fmul_rn(state[d],k[d]));}
        float value=cv[2*c.key_heads*128+head*128+lane],delta=(value-memory)*gate;
        #pragma unroll
        for(int d=0;d<128;++d)state[d]=__fadd_rn(state[d],__fmul_rn(k[d],delta));
        __syncthreads();
    }
    #pragma unroll
    for(int d=0;d<128;++d)canonical[stateBase+d*128+lane]=state[d];
}
__global__ void packedCommitKV(Qwen35PackedCommitConfig c,Qwen35PackedCommitSpans spans,
    const float* packed,float* state,const int64_t* layout,size_t elements){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=elements)return;
    int dim=i%c.head_dim;i/=c.head_dim;int t=i%c.tokens;i/=c.tokens;
    int head=i%c.kv_heads;i/=c.kv_heads;int which=i%2;i/=2;int layer=i%c.full_layers,b=i/c.full_layers;
    if(t>=spans.count[b])return;
    int slot=spans.start[b]+t;
    int64_t offset=layout[layer]+((int64_t(which)*c.kv_heads+head)*c.capacity+slot)*c.head_dim+dim;
    state[int64_t(b)*c.state_width+offset]=packed[int64_t(b)*c.packed_width+c.hidden_width+offset];
}
bool valid(const Qwen35PackedCommitConfig& c){
    return c.batch>=1&&c.batch<=8&&c.tokens>=1&&c.tokens<=16&&c.key_heads>=1&&c.key_heads<=64&&
        c.value_heads>=1&&c.value_heads<=64&&c.value_heads%c.key_heads==0&&c.kernel>=1&&c.kernel<=4&&
        c.linear_layers>=1&&c.linear_layers<=64&&c.full_layers>=0&&c.full_layers<=64&&
        c.kv_heads>=1&&c.kv_heads<=32&&c.capacity>=32&&c.capacity<=8192&&c.head_dim>=1&&c.head_dim<=256&&
        c.state_width>0&&c.state_width<=INT32_MAX&&c.packed_width>c.state_width&&c.packed_width<=INT32_MAX&&
        c.hidden_width>0&&c.hidden_width<=c.packed_width-c.state_width;
}
bool overlap(const void* a,size_t n,const void* b,size_t m){
    auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x>=y?x-y<m:y-x<n;
}
}
size_t Qwen35PackedCommitWorkspace(const Qwen35PackedCommitConfig& c) noexcept {
    return valid(c)?size_t(c.linear_layers)*c.batch*c.tokens*(2*c.key_heads*128+c.value_heads*128)*4:0;
}
int RunQwen35PackedCommit(const Qwen35PackedCommitConfig& c,Qwen35PackedCommitSpans spans,const float* packed,
    float* canonical,const Qwen35PackedLinearLayout* linear,const int64_t* full,const float* parameters,float* workspace,cudaStream_t stream) noexcept {
    if(!valid(c)||!packed||!canonical||!linear||!parameters||!workspace||(c.full_layers&&!full))return 1;
    for(int b=0;b<c.batch;++b)if(spans.start[b]<0||spans.start[b]>c.capacity||spans.count[b]<0||
        spans.count[b]>c.tokens||spans.count[b]>c.capacity-spans.start[b])return 1;
    size_t packedBytes=size_t(c.batch)*c.packed_width*4,stateBytes=size_t(c.batch)*c.state_width*4,workBytes=Qwen35PackedCommitWorkspace(c);
    size_t linearBytes=size_t(c.linear_layers)*sizeof(*linear),fullBytes=size_t(c.full_layers)*sizeof(*full);
    size_t paramBytes=size_t(c.linear_layers)*((2*c.key_heads*128+c.value_heads*128)*c.kernel+2*c.value_heads)*4;
    const void* pointers[]={packed,canonical,linear,full,parameters,workspace};
    size_t sizes[]={packedBytes,stateBytes,linearBytes,fullBytes,paramBytes,workBytes};
    for(int out: {1,5})for(int i=0;i<6;++i)if(i!=out&&sizes[i]&&overlap(pointers[out],sizes[out],pointers[i],sizes[i]))return 1;
    int channels=2*c.key_heads*128+c.value_heads*128;
    packedCommitConvolution<<<dim3((c.batch*channels+127)/128,c.linear_layers),128,0,stream>>>(c,spans,packed,canonical,linear,parameters,workspace);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    packedCommitRecurrent<<<dim3(c.batch*c.value_heads,c.linear_layers),128,0,stream>>>(c,spans,packed,canonical,linear,parameters,workspace);
    if(cudaPeekAtLastError()!=cudaSuccess)return 1;
    if(c.full_layers){
        size_t n=size_t(c.batch)*c.full_layers*2*c.kv_heads*c.tokens*c.head_dim;
        packedCommitKV<<<(n+255)/256,256,0,stream>>>(c,spans,packed,canonical,full,n);
    }
    return cudaPeekAtLastError()==cudaSuccess?0:1;
}
}
