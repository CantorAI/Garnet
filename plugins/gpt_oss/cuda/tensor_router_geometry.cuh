// SPDX-License-Identifier: Apache-2.0
// Opt-in geometry only; per-warp WMMA arithmetic matches the original.
#pragma once
template<int Warps>
__global__ void routeScoresTensorCoreGeometry(const float* x,const float* weight,const float* bias,
    float* logits,int tokens,GptOssOptions o) {
    static_assert(Warps==2);
    constexpr int Threads=Warps*32,Experts=Warps*16;
    const int first=blockIdx.x*16,firstExpert=blockIdx.y*Experts;
#if __CUDA_ARCH__ >= 800
    namespace W=nvcuda::wmma;
    __shared__ __align__(32) __nv_bfloat16 a[16*32],b[Experts*32];
    __shared__ __align__(32) float result[Warps][16*16];
    const int warp=threadIdx.x/32;
    W::fragment<W::matrix_a,16,16,16,__nv_bfloat16,W::row_major> af;
    W::fragment<W::matrix_b,16,16,16,__nv_bfloat16,W::col_major> bfFragment;
    W::fragment<W::accumulator,16,16,16,float> acc;
    W::fill_fragment(acc,0.f);
    for(int start=0;start<o.hidden;start+=32) {
        for(int i=threadIdx.x;i<16*32;i+=Threads) {
            const int token=first+i/32,k=start+i%32;
            a[i]=__float2bfloat16(token<tokens && k<o.hidden?x[size_t(token)*o.hidden+k]:0.f);
        }
        for(int i=threadIdx.x;i<Experts*32;i+=Threads) {
            const int expert=firstExpert+i/32,k=start+i%32;
            b[i]=__float2bfloat16(expert<o.experts && k<o.hidden?weight[size_t(expert)*o.hidden+k]:0.f);
        }
        __syncthreads();
        for(int offset=0;offset<32;offset+=16) {
            W::load_matrix_sync(af,a+offset,32);
            W::load_matrix_sync(bfFragment,b+warp*16*32+offset,32);
            W::mma_sync(acc,af,bfFragment,acc);
        }
        __syncthreads();
    }
    W::store_matrix_sync(result[warp],acc,16,W::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x;i<16*Experts;i+=Threads) {
        const int token=first+i/Experts,e=i%Experts,expert=firstExpert+e;
        if(token<tokens && expert<o.experts)
            logits[size_t(token)*o.experts+expert]=bf(result[e/16][(i/Experts)*16+e%16]+bias[expert]);
    }
#else
    // Older architectures retain functional scores, without a tensor-core gain.
    for(int i=threadIdx.x;i<16*Experts;i+=Threads) {
        const int token=first+i/Experts,expert=firstExpert+i%Experts;
        if(token<tokens && expert<o.experts) {
            float value=0;
            for(int k=0;k<o.hidden;++k)value+=bf(x[size_t(token)*o.hidden+k])*bf(weight[size_t(expert)*o.hidden+k]);
            logits[size_t(token)*o.experts+expert]=bf(value+bias[expert]);
        }
    }
#endif
}
