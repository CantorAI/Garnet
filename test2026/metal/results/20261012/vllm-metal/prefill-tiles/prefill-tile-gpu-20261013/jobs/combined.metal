#include <metal_stdlib>
using namespace metal;
struct BF16Params {ulong rows,inner,columns,aStride,oStride;};
kernel void bf16_decode(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            ushort4 b0=*(device const ushort4*)(weight+column*p.inner+k),b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
            if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
            if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
            if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
                sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<p.inner;k+=32) {
        float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
        float4 w=0.0f;
        w.x=as_type<float>(uint(weight[column*p.inner+k])<<16);
        if(column+1<p.columns)w.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
        if(column+2<p.columns)w.z=as_type<float>(uint(weight[(column+2)*p.inner+k])<<16);
        if(column+3<p.columns)w.w=as_type<float>(uint(weight[(column+3)*p.inner+k])<<16);
        sum0+=x*w;if(p.rows==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
// BF16 weights are expanded only inside a small shared-memory tile. FP32
// SIMD-group matrix multiply accumulates two 8x8 output tiles per group.
kernel void bf16_gemm(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float left[16*32],right[32*32],result[16*32];
    ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*32;
    simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
    simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(ulong base=0;base<p.inner;base+=32) {
        for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,k=base+i%32;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
        for(uint i=tid;i<32*32;i+=128){ulong k=base+i/32,column=colBase+i%32;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}
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
kernel void gemm_16_32_64(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float left[1024],right[2048],result[512];
ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*32;
simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=64) {
for(uint i=tid;i<1024;i+=128){ulong row=rowBase+i/64,k=base+i%64;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
for(uint i=tid;i<2048;i+=128){ulong k=base+i/32,column=colBase+i%32;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint offset=0;offset<64;offset+=8) {
simdgroup_float8x8 r;
simdgroup_load(r,right+offset*32+simd*8,32);
simdgroup_float8x8 l0;
simdgroup_load(l0,left+0+offset,64);
simdgroup_multiply_accumulate(accum0,l0,r,accum0);
simdgroup_float8x8 l1;
simdgroup_load(l1,left+512+offset,64);
simdgroup_multiply_accumulate(accum1,l1,r,accum1);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(accum0,result+0+simd*8,32);
simdgroup_store(accum1,result+256+simd*8,32);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<512;i+=128){ulong row=rowBase+i/32,column=colBase+i%32;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=result[i];}
}
kernel void gemm_16_32_128(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float left[2048],right[4096],result[512];
ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*32;
simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=128) {
for(uint i=tid;i<2048;i+=128){ulong row=rowBase+i/128,k=base+i%128;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
for(uint i=tid;i<4096;i+=128){ulong k=base+i/32,column=colBase+i%32;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint offset=0;offset<128;offset+=8) {
simdgroup_float8x8 r;
simdgroup_load(r,right+offset*32+simd*8,32);
simdgroup_float8x8 l0;
simdgroup_load(l0,left+0+offset,128);
simdgroup_multiply_accumulate(accum0,l0,r,accum0);
simdgroup_float8x8 l1;
simdgroup_load(l1,left+1024+offset,128);
simdgroup_multiply_accumulate(accum1,l1,r,accum1);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(accum0,result+0+simd*8,32);
simdgroup_store(accum1,result+256+simd*8,32);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<512;i+=128){ulong row=rowBase+i/32,column=colBase+i%32;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=result[i];}
}
kernel void gemm_16_64_64(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float left[1024],right[4096],result[1024];
ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*64;
simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=64) {
for(uint i=tid;i<1024;i+=256){ulong row=rowBase+i/64,k=base+i%64;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
for(uint i=tid;i<4096;i+=256){ulong k=base+i/64,column=colBase+i%64;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}
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
kernel void gemm_32_32_64(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float left[2048],right[2048],result[1024];
ulong rowBase=ulong(group.y)*32,colBase=ulong(group.x)*32;
simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum2=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum3=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=64) {
for(uint i=tid;i<2048;i+=128){ulong row=rowBase+i/64,k=base+i%64;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
for(uint i=tid;i<2048;i+=128){ulong k=base+i/32,column=colBase+i%32;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint offset=0;offset<64;offset+=8) {
simdgroup_float8x8 r;
simdgroup_load(r,right+offset*32+simd*8,32);
simdgroup_float8x8 l0;
simdgroup_load(l0,left+0+offset,64);
simdgroup_multiply_accumulate(accum0,l0,r,accum0);
simdgroup_float8x8 l1;
simdgroup_load(l1,left+512+offset,64);
simdgroup_multiply_accumulate(accum1,l1,r,accum1);
simdgroup_float8x8 l2;
simdgroup_load(l2,left+1024+offset,64);
simdgroup_multiply_accumulate(accum2,l2,r,accum2);
simdgroup_float8x8 l3;
simdgroup_load(l3,left+1536+offset,64);
simdgroup_multiply_accumulate(accum3,l3,r,accum3);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(accum0,result+0+simd*8,32);
simdgroup_store(accum1,result+256+simd*8,32);
simdgroup_store(accum2,result+512+simd*8,32);
simdgroup_store(accum3,result+768+simd*8,32);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<1024;i+=128){ulong row=rowBase+i/32,column=colBase+i%32;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=result[i];}
}
kernel void gemm_32_64_32(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float left[1024],right[2048],result[2048];
ulong rowBase=ulong(group.y)*32,colBase=ulong(group.x)*64;
simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum2=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum3=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=32) {
for(uint i=tid;i<1024;i+=256){ulong row=rowBase+i/32,k=base+i%32;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
for(uint i=tid;i<2048;i+=256){ulong k=base+i/64,column=colBase+i%64;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}
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
