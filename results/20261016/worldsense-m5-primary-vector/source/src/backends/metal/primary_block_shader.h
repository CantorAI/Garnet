#pragma once
namespace Garnet {inline constexpr const char* kPrimaryBlockShader=R"PRIMARY(
inline ushort primary_word(device const uchar* payload,device const ulong* metadata,ulong row,ulong k,ulong inner) {
    ulong block=(row/4)*(inner/128)+k/128,meta=metadata[block],at=(meta>>9)*256,index=(k%128)*4+row%4;
    if(meta&256ul){uint sm=payload[at+index],tag=(payload[at+512+index/2]>>((index%2)*4))&15u;
        return ushort((sm&127u)|((sm&128u)<<8)|(((uint(meta)&255u)+tag)<<7));}
    return ushort(payload[at+index*2])|(ushort(payload[at+index*2+1])<<8);
}
inline ushort4 primary_row4(device const uchar* payload,device const ulong* metadata,ulong row,ulong k,ulong inner) {
    return ushort4(primary_word(payload,metadata,row,k,inner),primary_word(payload,metadata,row,k+1,inner),primary_word(payload,metadata,row,k+2,inner),primary_word(payload,metadata,row,k+3,inner));
}

// Packed admission proves row%4==0, k%4==0, inner%128==0 and four valid rows.
// At every decode iteration k..k+3 stays in a single 128-K block.
struct PrimaryQuad { ushort4 r0,r1,r2,r3; };
inline ushort4 primary_quad_row(uint4 sm,uint4 tags,uint base,uint row) {
    uint4 bytes=(sm>>(row*8))&255u;
    return ushort4((bytes&127u)|((bytes&128u)<<8)|((uint4(base)+((tags>>(row*4))&15u))<<7));
}
inline PrimaryQuad primary_quad(device const uchar* payload,device const ulong* metadata,ulong row,ulong k,ulong inner) {
    ulong meta=metadata[(row/4)*(inner/128)+k/128],at=(meta>>9)*256,index=(k%128)*4;
    PrimaryQuad q;
    if(meta&256ul) {
        uint4 sm=*(device const uint4*)(payload+at+index);
        uint2 packedTags=*(device const uint2*)(payload+at+512+index/2);
        uint4 tags=uint4(packedTags.x&65535u,packedTags.x>>16,packedTags.y&65535u,packedTags.y>>16);
        uint base=uint(meta)&255u;
        q.r0=primary_quad_row(sm,tags,base,0);q.r1=primary_quad_row(sm,tags,base,1);
        q.r2=primary_quad_row(sm,tags,base,2);q.r3=primary_quad_row(sm,tags,base,3);
    } else {
        uint4 lo=*(device const uint4*)(payload+at+index*2),hi=*(device const uint4*)(payload+at+index*2+16);
        uint4 even=uint4(lo.x,lo.z,hi.x,hi.z),odd=uint4(lo.y,lo.w,hi.y,hi.w);
        q.r0=ushort4(even&65535u);q.r1=ushort4(even>>16);
        q.r2=ushort4(odd&65535u);q.r3=ushort4(odd>>16);
    }
    return q;
}
// Correctness-only bounded readback exercises the exact tile helper used by decode.
kernel void primary_expand_vector_words(device const uchar* payload [[buffer(0)]],device const ulong* metadata [[buffer(1)]],
 device uint* output [[buffer(2)]],constant ulong& inner [[buffer(3)]],constant ulong& start [[buffer(4)]],constant ulong& count [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    if(ulong(id)>=count)return;
    ulong i=start+id,row=i/inner,k=i%inner;
    PrimaryQuad q=primary_quad(payload,metadata,row-row%4,k-k%4,inner);
    ushort4 values=row%4==0?q.r0:row%4==1?q.r1:row%4==2?q.r2:q.r3;
    output[id]=uint(values[k%4])<<16;
}

kernel void primary_expand_words(device const uchar* payload [[buffer(0)]],device const ulong* metadata [[buffer(1)]],
 device uint* output [[buffer(2)]],constant ulong& inner [[buffer(3)]],constant ulong& start [[buffer(4)]],constant ulong& count [[buffer(5)]],uint id [[thread_position_in_grid]]) {
 if(ulong(id)>=count)return;ulong i=start+id;output[id]=uint(primary_word(payload,metadata,i/inner,i%inner,inner))<<16;
}
kernel void primary_bf16_decode(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const ulong* metadata [[buffer(4)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            PrimaryQuad quad=primary_quad(payload,metadata,column,k,p.inner);
            ushort4 b0=quad.r0,b1=quad.r1,b2=quad.r2,b3=quad.r3;
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
                sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<p.inner;k+=32) {
        float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
        float4 w=0.0f;
        w.x=as_type<float>(uint(primary_word(payload,metadata,column,k,p.inner))<<16);
        if(column+1<p.columns)w.y=as_type<float>(uint(primary_word(payload,metadata,(column+1),k,p.inner))<<16);
        if(column+2<p.columns)w.z=as_type<float>(uint(primary_word(payload,metadata,(column+2),k,p.inner))<<16);
        if(column+3<p.columns)w.w=as_type<float>(uint(primary_word(payload,metadata,(column+3),k,p.inner))<<16);
        sum0+=x*w;if(p.rows==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
kernel void primary_bf16_gemm(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const ulong* metadata [[buffer(4)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float left[16*32],right[32*32],result[16*32];
    ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*32;
    simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
    simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(ulong base=0;base<p.inner;base+=32) {
        for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,k=base+i%32;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
        for(uint i=tid;i<32*32;i+=128){ulong k=base+i/32,column=colBase+i%32;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(primary_word(payload,metadata,column,k,p.inner))<<16):0.0f;}
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
kernel void primary_gemm_16_64_64(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const ulong* metadata [[buffer(4)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float left[1024],right[4096],result[1024];
ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*64;
simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=64) {
for(uint i=tid;i<1024;i+=256){ulong row=rowBase+i/64,k=base+i%64;left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
for(uint i=tid;i<4096;i+=256){ulong k=base+i/64,column=colBase+i%64;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(primary_word(payload,metadata,column,k,p.inner))<<16):0.0f;}
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
kernel void primary_gemm_32_64_32(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const ulong* metadata [[buffer(4)]],
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
for(uint i=tid;i<2048;i+=256){ulong k=base+i/64,column=colBase+i%64;right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(primary_word(payload,metadata,column,k,p.inner))<<16):0.0f;}
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
kernel void primary_specialized_bf16_decode(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const ulong* metadata [[buffer(4)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(!fixed_no_tail&&column>=ulong(fixed_columns))return;
    float4 sum0=0.0f,sum1=0.0f;
    if(ulong(fixed_inner)%4==0) {
        for(ulong k=ulong(lane)*4;k<ulong(fixed_inner);k+=128) {
            float4 x=*(device const float4*)(a+k),y=ulong(fixed_rows)==2?*(device const float4*)(a+ulong(fixed_aStride)+k):float4(0.0f);
            PrimaryQuad quad=primary_quad(payload,metadata,column,k,ulong(fixed_inner));
            ushort4 b0=quad.r0,b1=quad.r1,b2=quad.r2,b3=quad.r3;
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
                sum0+=x[j]*w;if(ulong(fixed_rows)==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<ulong(fixed_inner);k+=32) {
        float x=a[k],y=ulong(fixed_rows)==2?a[ulong(fixed_aStride)+k]:0.0f;
        float4 w=0.0f;
        w.x=as_type<float>(uint(primary_word(payload,metadata,column,k,ulong(fixed_inner)))<<16);
        if(fixed_no_tail||column+1<ulong(fixed_columns))w.y=as_type<float>(uint(primary_word(payload,metadata,(column+1),k,ulong(fixed_inner)))<<16);
        if(fixed_no_tail||column+2<ulong(fixed_columns))w.z=as_type<float>(uint(primary_word(payload,metadata,(column+2),k,ulong(fixed_inner)))<<16);
        if(fixed_no_tail||column+3<ulong(fixed_columns))w.w=as_type<float>(uint(primary_word(payload,metadata,(column+3),k,ulong(fixed_inner)))<<16);
        sum0+=x*w;if(ulong(fixed_rows)==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&(fixed_no_tail||column+j<ulong(fixed_columns));++j){out[column+j]=sum0[j];if(ulong(fixed_rows)==2)out[ulong(fixed_oStride)+column+j]=sum1[j];}
}
)PRIMARY";}
