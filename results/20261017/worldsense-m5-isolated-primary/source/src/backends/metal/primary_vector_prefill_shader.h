// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kPrimaryPrefillMinimalPreamble=R"PMIN(#include <metal_stdlib>
using namespace metal;
struct BF16Params {ulong rows,inner,columns,aStride,oStride;};
inline ushort4 primary_quad_kword(uint sm,ushort tags,uint base) {
    uint4 signMant=uint4(as_type<uchar4>(sm));
    uint4 exponents=uint4(tags,tags>>4,tags>>8,tags>>12)&15u;
    return ushort4((signMant&127u)|((signMant&128u)<<8)|((exponents+base)<<7));
}
)PMIN";
inline constexpr const char* kPrimaryVectorPrefillShader=R"PVEC(
// Admission proves row%4, k%2, inner%128 and full four-row groups.
// Each duo stays in one original 128-K block; no floating arithmetic.
struct PrimaryDuo {ushort4 k0,k1;};
inline PrimaryDuo primary_duo(device const uchar* payload,device const uint* metadata,uint row,uint k,uint inner) {
 uint meta=metadata[(row/4)*(inner/128)+k/128],at=(meta>>9)*256,index=(k%128)*4;
 PrimaryDuo d;
 if(meta&256u) {
  uint2 sm=*(device const uint2*)(payload+at+index);
  uint tags=*(device const uint*)(payload+at+512+index/2),base=meta&255u;
  d.k0=primary_quad_kword(sm.x,ushort(tags),base);
  d.k1=primary_quad_kword(sm.y,ushort(tags>>16),base);
 } else {
  uint4 raw=*(device const uint4*)(payload+at+index*2);
  d.k0=as_type<ushort4>(raw.xy);d.k1=as_type<ushort4>(raw.zw);
 }
 return d;
}
template<uint tileColumns,uint tileK,uint threads>
inline void primary_prefill_tile_fill(device const uchar* payload,device const uint* metadata,
 threadgroup float* right,uint colBase,uint kBase,uint columns,uint inner,uint tid) {
 for(uint index=tid;index<(tileColumns/4)*(tileK/2);index+=threads) {
  uint localK=(index/(tileColumns/4))*2,localColumn=(index%(tileColumns/4))*4;
  uint row=colBase+localColumn,k=kBase+localK;
  PrimaryDuo duo;
  if(row<columns&&k<inner)duo=primary_duo(payload,metadata,row,k,inner);
  else {duo.k0=ushort4(0);duo.k1=ushort4(0);}
  ushort4 words[2]={duo.k0,duo.k1};
  for(uint j=0;j<2;++j)
   *(threadgroup float4*)(right+(localK+j)*tileColumns+localColumn)=as_type<float4>(uint4(words[j])<<16);
 }
}
kernel void primary_vector_bf16_gemm(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const uint* metadata [[buffer(4)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float left[16*32],result[16*32];
    threadgroup float4 rightAligned[256];threadgroup float* right=(threadgroup float*)rightAligned;
    ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*32;
    simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
    simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(ulong base=0;base<p.inner;base+=32) {
        for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,k=base+i%32;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
        primary_prefill_tile_fill<32,32,128>(payload,metadata,right,uint(colBase),uint(base),uint(p.columns),uint(p.inner),tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint offset=0;offset<32;offset+=8) {
            simdgroup_float8x8 l0,l1,r;
            simdgroup_load(l0,left+offset,32);simdgroup_load(l1,left+8*32+offset,32);simdgroup_load(r,right+offset*32+simd*8,32);
            simdgroup_multiply_accumulate(accum0,l0,r,accum0);simdgroup_multiply_accumulate(accum1,l1,r,accum1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(accum0,result+simd*8,32);simdgroup_store(accum1,result+8*32+simd*8,32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,column=colBase+i%32;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=result[i];}
}

kernel void primary_vector_gemm_16_64_64(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const uint* metadata [[buffer(4)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float left[1024],result[1024];
threadgroup float4 rightAligned[1024];threadgroup float* right=(threadgroup float*)rightAligned;
ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*64;
simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=64) {
for(uint i=tid;i<1024;i+=256){ulong row=rowBase+i/64,k=base+i%64;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
primary_prefill_tile_fill<64,64,256>(payload,metadata,right,uint(colBase),uint(base),uint(p.columns),uint(p.inner),tid);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint offset=0;offset<64;offset+=8) {
simdgroup_float8x8 r;
simdgroup_load(r,right+offset*64+simd*8,64);
simdgroup_float8x8 l0;
simdgroup_load(l0,left+0+offset,64);
simdgroup_multiply_accumulate(accum0,l0,r,accum0);
simdgroup_float8x8 l1;
simdgroup_load(l1,left+512+offset,64);
simdgroup_multiply_accumulate(accum1,l1,r,accum1);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(accum0,result+0+simd*8,64);
simdgroup_store(accum1,result+512+simd*8,64);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<1024;i+=256){ulong row=rowBase+i/64,column=colBase+i%64;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=result[i];}
}

kernel void primary_vector_gemm_32_64_32(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const uint* metadata [[buffer(4)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float left[1024],result[2048];
threadgroup float4 rightAligned[512];threadgroup float* right=(threadgroup float*)rightAligned;
ulong rowBase=ulong(group.y)*32,colBase=ulong(group.x)*64;
simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum2=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum3=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=32) {
for(uint i=tid;i<1024;i+=256){ulong row=rowBase+i/32,k=base+i%32;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
primary_prefill_tile_fill<64,32,256>(payload,metadata,right,uint(colBase),uint(base),uint(p.columns),uint(p.inner),tid);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint offset=0;offset<32;offset+=8) {
simdgroup_float8x8 r;
simdgroup_load(r,right+offset*64+simd*8,64);
simdgroup_float8x8 l0;
simdgroup_load(l0,left+0+offset,32);
simdgroup_multiply_accumulate(accum0,l0,r,accum0);
simdgroup_float8x8 l1;
simdgroup_load(l1,left+256+offset,32);
simdgroup_multiply_accumulate(accum1,l1,r,accum1);
simdgroup_float8x8 l2;
simdgroup_load(l2,left+512+offset,32);
simdgroup_multiply_accumulate(accum2,l2,r,accum2);
simdgroup_float8x8 l3;
simdgroup_load(l3,left+768+offset,32);
simdgroup_multiply_accumulate(accum3,l3,r,accum3);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(accum0,result+0+simd*8,64);
simdgroup_store(accum1,result+512+simd*8,64);
simdgroup_store(accum2,result+1024+simd*8,64);
simdgroup_store(accum3,result+1536+simd*8,64);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<2048;i+=256){ulong row=rowBase+i/64,column=colBase+i%64;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=result[i];}
}

struct PrimaryPrefillTileProof {uint inner,columns,firstRow,rows,tileColumns,tileK,threads;};
kernel void primary_vector_prefill_expand(device const uchar* payload [[buffer(0)]],device const uint* metadata [[buffer(1)]],
 device uint* output [[buffer(2)]],constant PrimaryPrefillTileProof& p [[buffer(3)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float4 rightAligned[1024];threadgroup float* right=(threadgroup float*)rightAligned;
 uint rowBase=p.firstRow+group.x*p.tileColumns,kBase=group.y*p.tileK;
 if(p.tileColumns==32)primary_prefill_tile_fill<32,32,128>(payload,metadata,right,rowBase,kBase,p.columns,p.inner,tid);
 else if(p.tileK==32)primary_prefill_tile_fill<64,32,256>(payload,metadata,right,rowBase,kBase,p.columns,p.inner,tid);
 else primary_prefill_tile_fill<64,64,256>(payload,metadata,right,rowBase,kBase,p.columns,p.inner,tid);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<p.tileColumns*p.tileK;i+=p.threads) {
  uint row=rowBase+i%p.tileColumns,k=kBase+i/p.tileColumns;
  if(row<p.firstRow+p.rows&&k<p.inner)output[(row-p.firstRow)*p.inner+k]=as_type<uint>(right[i]);
 }
}
)PVEC";}
