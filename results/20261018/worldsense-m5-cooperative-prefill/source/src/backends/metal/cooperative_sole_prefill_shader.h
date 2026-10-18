// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
inline constexpr const char* kCooperativeSolePrefillShader=R"COOPMLP(

inline ushort sole_direct_word(device const ushort* weight,ulong column,ulong k,ulong inner) {
 return weight[(column/4)*inner*4+k*4+column%4];
}
inline ulong2 sole_cooperative_coordinate(ulong i,ulong tileK){return ulong2((i/4)%tileK,(i/(tileK*4))*4+i%4);}
inline ulong sole_cooperative_slot(ulong k,ulong column,ulong tileN){return k*(tileN+4)+column;}
kernel void sole_direct_bf16_gemm(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float left[16*32],right[32*(32+4)],result[16*32];
    ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*32;
    simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
    simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(ulong base=0;base<p.inner;base+=32) {
        for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,k=base+i%32;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
        for(uint i=tid;i<32*32;i+=128){ulong2 coordinate=sole_cooperative_coordinate(i,32);ulong k=base+coordinate.x,column=colBase+coordinate.y;right[sole_cooperative_slot(coordinate.x,coordinate.y,32)]=(k<p.inner&&column<p.columns)?as_type<float>(uint(sole_direct_word(weight,column,k,p.inner))<<16):0.0f;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint offset=0;offset<32;offset+=8) {
            simdgroup_float8x8 l0,l1,r;
            simdgroup_load(l0,left+offset,32);simdgroup_load(l1,left+8*32+offset,32);simdgroup_load(r,right+offset*(32+4)+simd*8,32+4);
            simdgroup_multiply_accumulate(accum0,l0,r,accum0);simdgroup_multiply_accumulate(accum1,l1,r,accum1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(accum0,result+simd*8,32);simdgroup_store(accum1,result+8*32+simd*8,32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,column=colBase+i%32;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=result[i];}
}
kernel void sole_direct_gemm_16_64_64(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float left[1024],right[64*(64+4)],result[1024];
ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*64;
simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=64) {
for(uint i=tid;i<1024;i+=256){ulong row=rowBase+i/64,k=base+i%64;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
for(uint i=tid;i<4096;i+=256){ulong2 coordinate=sole_cooperative_coordinate(i,64);ulong k=base+coordinate.x,column=colBase+coordinate.y;right[sole_cooperative_slot(coordinate.x,coordinate.y,64)]=(k<p.inner&&column<p.columns)?as_type<float>(uint(sole_direct_word(weight,column,k,p.inner))<<16):0.0f;}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint offset=0;offset<64;offset+=8) {
simdgroup_float8x8 r;
simdgroup_load(r,right+offset*(64+4)+simd*8,64+4);
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
kernel void sole_direct_gemm_32_64_32(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float left[1024],right[32*(64+4)],result[2048];
ulong rowBase=ulong(group.y)*32,colBase=ulong(group.x)*64;
simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum2=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum3=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=32) {
for(uint i=tid;i<1024;i+=256){ulong row=rowBase+i/32,k=base+i%32;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
for(uint i=tid;i<2048;i+=256){ulong2 coordinate=sole_cooperative_coordinate(i,32);ulong k=base+coordinate.x,column=colBase+coordinate.y;right[sole_cooperative_slot(coordinate.x,coordinate.y,64)]=(k<p.inner&&column<p.columns)?as_type<float>(uint(sole_direct_word(weight,column,k,p.inner))<<16):0.0f;}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint offset=0;offset<32;offset+=8) {
simdgroup_float8x8 r;
simdgroup_load(r,right+offset*(64+4)+simd*8,64+4);
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
struct SoleDirectTileParams {ulong rows,inner,first,count,tileN,tileK;};
kernel void sole_direct_tile_read(device const ushort* weight [[buffer(0)]],device uint* output [[buffer(1)]],
 constant SoleDirectTileParams& p [[buffer(2)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup uint right[64*(64+4)];
 ulong colBase=p.first+ulong(group.x)*p.tileN,base=ulong(group.y)*p.tileK;
 uint threads=p.tileN==32?128:256;
 for(uint i=tid;i<p.tileN*p.tileK;i+=threads){ulong2 coordinate=sole_cooperative_coordinate(i,p.tileK);ulong k=base+coordinate.x,column=colBase+coordinate.y;right[sole_cooperative_slot(coordinate.x,coordinate.y,p.tileN)]=(k<p.inner&&column<p.rows)?uint(sole_direct_word(weight,column,k,p.inner))<<16:0u;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<p.tileN*p.tileK;i+=threads){ulong k=base+i/p.tileN,column=colBase+i%p.tileN;if(k<p.inner&&column<p.first+p.count)output[(column-p.first)*p.inner+k]=right[sole_cooperative_slot(i/p.tileN,i%p.tileN,p.tileN)];}
}
)COOPMLP";
}
