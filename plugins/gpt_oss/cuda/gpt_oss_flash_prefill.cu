// SPDX-License-Identifier: Apache-2.0
// AttentionSink variant adapted from FlashInfer v0.7.0.post1, Apache-2.0.
// Copyright (c) 2025 by FlashInfer team. See third_party/flashinfer/PROVENANCE.txt.
#include "gpt_oss_flash_prefill.h"
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <algorithm>

namespace Garnet {
namespace {
using Bf16=nv_bfloat16;
// FlashInfer's32-row specialization is reserved for wider value dimensions;
// head64 uses the ordinary four-query-warp64-row packed GQA tile.
constexpr int TileQ=64;
struct Layout {
    size_t bytes=0,q=0,out=0,qptr=0,pptr=0,last=0,requests=0,tiles=0,kvTiles=0,chunk=0,indices=0;
    int tileCount=0;
    size_t take(size_t n){const size_t offset=bytes;bytes=(bytes+n+255)&~size_t(255);return offset;}
    Layout(int batch,int tokens,int logical,int heads) {
        const size_t rows=size_t(batch)*tokens;
        q=take(rows*heads*64*2);out=take(rows*heads*64*2);
        qptr=take(size_t(batch+1)*4);pptr=take(size_t(batch+1)*4);last=take(size_t(batch)*4);
        tileCount=batch*((tokens*8+TileQ-1)/TileQ);
        requests=take(size_t(tileCount)*4);tiles=take(size_t(tileCount)*4);kvTiles=take(size_t(tileCount)*4);
        chunk=take(4);indices=take(size_t(batch)*logical*4);
    }
};
template<class T>T* at(void* workspace,size_t offset){return reinterpret_cast<T*>(static_cast<char*>(workspace)+offset);}
struct Params : flashinfer::BatchPrefillPagedParams<Bf16,Bf16,Bf16,int> {
    const float* sink=nullptr;
    const int* starts=nullptr;
    const int* active=nullptr;
    const int* rawTable=nullptr;
    int tokens=0,logical=0,physical=0,pageSize=0,window=0;
    __host__ __device__ uint32_t get_kv_len(uint32_t b) const {
        return uint32_t(min(logical*pageSize,max(1,starts[b]+tokens)));
    }
};
// Own the complete mask: causal offset, sliding window, missing pages and
// inactive/over-capacity rows. Safe indices prevent speculative loads through
// holes; the original table determines whether those values contribute.
struct SinkAttention : flashinfer::AttentionVariantBase {
    static constexpr bool use_softmax=true;
    uint32_t qo_len,kv_len,window_left;
    float sm_scale_log2;
    template<class P>__device__ __host__ SinkAttention(const P& p,uint32_t b,uint8_t*) {
        qo_len=p.get_qo_len(b);kv_len=p.get_kv_len(b);
        window_left=p.window && p.starts[b]>=0 && p.starts[b]+p.tokens<=p.logical*p.pageSize
            ? uint32_t(p.window-1) : kv_len;
        sm_scale_log2=p.sm_scale*flashinfer::math::log2e;
    }
    REGISTER_LOGITS_MASK(p,b,q,k,qh,kh,{
        const int end=p.starts[b]+int(q)+1;
        if(!p.active[b]||q>=uint32_t(p.tokens)||end<=0||end>p.logical*p.pageSize||k>=uint32_t(end))return false;
        if(p.window && int(k)<max(0,end-p.window))return false;
        const int page=p.rawTable[b*p.logical+k/p.pageSize];
        return page>=0 && page<p.physical;
    })
    REGISTER_M_D_UPDATE(p,kvTile,qh,m,d,scale,{
        const float sink=kvTile==0 && qh<p.num_qo_heads ? p.sink[qh]*flashinfer::math::log2e : -flashinfer::math::inf;
        const float next=sink>m?sink:m;
        scale=flashinfer::math::ptx_exp2(max(m-next,-flashinfer::math::inf));
        d=flashinfer::math::ptx_exp2(max(sink-next,-flashinfer::math::inf))+d*scale;
        m=next;
    })
    REGISTER_OUTPUT_TRANSFORM(p,output,b,q,qh,m,d,scale,{
        return output*scale*(m != -flashinfer::math::inf ? flashinfer::math::ptx_rcp(d) : 0.f);
    })
};
__global__ void prepare(const float* qkv,const int* table,Bf16* q,int* qptr,int* pptr,int* last,
    int* requests,int* tiles,int* kvTiles,int* chunk,int* indices,int batch,int tokens,int logical,
    int physical,int heads,int kvHeads) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t qCount=size_t(batch)*tokens*heads*64;
    if(i<qCount)q[i]=__float2bfloat16(qkv[(i/(heads*64))*(heads+2*kvHeads)*64+i%(heads*64)]);
    if(i<=size_t(batch)){qptr[i]=int(i)*tokens;pptr[i]=int(i)*logical;}
    if(i<size_t(batch))last[i]=16;
    const int perRequest=(tokens*8+TileQ-1)/TileQ;
    if(i<size_t(batch)*perRequest){requests[i]=int(i)/perRequest;tiles[i]=int(i)%perRequest;kvTiles[i]=0;}
    if(i<size_t(batch)*logical){const int page=table[i];indices[i]=page>=0&&page<physical?page:0;}
    if(!i)*chunk=logical*16;
}
__global__ void convertOutput(const Bf16* x,float* y,size_t count){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)y[i]=__bfloat162float(x[i]);
}
bool supported(int batch,int tokens,int logical,const GptOssOptions& o){
    return batch>0&&batch<=512&&tokens>0&&tokens<=4096&&logical>0&&logical<=256&&o.prefill&&
        o.headDim==64&&o.kvHeads>0&&o.kvHeads<=16&&o.qHeads==8*o.kvHeads&&o.pageSize==16&&o.window>=0;
}
}
size_t GptOssFlashPrefillWorkspace(int batch,int tokens,int logical,const GptOssOptions& o){
    return supported(batch,tokens,logical,o)?Layout(batch,tokens,logical,o.qHeads).bytes:0;
}
cudaError_t RunGptOssFlashPrefill(const void* const* in,float* y,void* workspace,int batch,int tokens,
    int logical,int physical,const GptOssOptions& o,cudaStream_t stream){
    if(!supported(batch,tokens,logical,o)||physical<=0||!in||!y||!workspace)return cudaErrorInvalidValue;
    const Layout l(batch,tokens,logical,o.qHeads);
    const size_t count=size_t(batch)*tokens*o.qHeads*64;
    const size_t work=std::max(count,std::max(size_t(batch)*logical,size_t(l.tileCount)));
    prepare<<<(work+255)/256,256,0,stream>>>((const float*)in[0],(const int*)in[3],at<Bf16>(workspace,l.q),
        at<int>(workspace,l.qptr),at<int>(workspace,l.pptr),at<int>(workspace,l.last),at<int>(workspace,l.requests),
        at<int>(workspace,l.tiles),at<int>(workspace,l.kvTiles),at<int>(workspace,l.chunk),at<int>(workspace,l.indices),
        batch,tokens,logical,physical,o.qHeads,o.kvHeads);
    auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
    Params p;p.q=at<Bf16>(workspace,l.q);p.o=at<Bf16>(workspace,l.out);
    const size_t layerOffset=size_t(o.layer)*physical*16*o.kvHeads*64;
    p.paged_kv=flashinfer::paged_kv_t<Bf16,int>(o.kvHeads,16,64,batch,flashinfer::QKVLayout::kNHD,
        (Bf16*)in[1]+layerOffset,(Bf16*)in[2]+layerOffset,at<int>(workspace,l.indices),at<int>(workspace,l.pptr),at<int>(workspace,l.last));
    p.q_indptr=at<int>(workspace,l.qptr);p.o_indptr=p.q_indptr;
    p.num_qo_heads=o.qHeads;p.group_size=flashinfer::uint_fastdiv(8);
    p.q_stride_n=o.qHeads*64;p.q_stride_h=64;p.sm_scale=.125f;
    p.request_indices=at<int>(workspace,l.requests);p.qo_tile_indices=at<int>(workspace,l.tiles);
    p.kv_tile_indices=at<int>(workspace,l.kvTiles);p.kv_chunk_size_ptr=at<int>(workspace,l.chunk);
    p.padded_batch_size=l.tileCount;p.max_total_num_rows=batch*tokens;
    p.sink=(const float*)in[7];p.starts=(const int*)in[5];p.active=(const int*)in[6];
    p.rawTable=(const int*)in[3];p.tokens=tokens;p.logical=logical;p.physical=physical;p.pageSize=16;p.window=o.window;
    try {
        status=flashinfer::BatchPrefillWithPagedKVCacheDispatched<true,TileQ,64,64,
            // Custom mode invokes our complete mask on every KV tile;
            // ordinary mode skips masks on interior tiles, including holes.
            flashinfer::PosEncodingMode::kNone,false,flashinfer::MaskMode::kCustom,SinkAttention,Params>(
            p,nullptr,nullptr,false,stream);
    }catch(const std::exception& error){
        std::fprintf(stderr,"GPT-OSS FlashInfer prefill rejected: %s\n",error.what());
        return cudaErrorNotSupported;
    }catch(...){return cudaErrorNotSupported;}
    if(status!=cudaSuccess)return status;
    convertOutput<<<(count+255)/256,256,0,stream>>>(p.o,y,count);
    return cudaGetLastError();
}
}
