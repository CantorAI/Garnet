
#include <metal_stdlib>
using namespace metal;
struct BF16Params {ulong rows,inner,columns,aStride,oStride;};

constant uint fixed_rows [[function_constant(0)]];
constant uint fixed_inner [[function_constant(1)]];
constant uint fixed_columns [[function_constant(2)]];
constant uint fixed_aStride [[function_constant(3)]];
constant uint fixed_oStride [[function_constant(4)]];
constant bool fixed_no_tail [[function_constant(5)]];

kernel void sole_bf16_decode(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            ulong packed=(column/4)*p.inner*4+k*4;
            uint4 pair0=*(device const uint4*)(weight+packed),pair1=*(device const uint4*)(weight+packed+8);
            ushort4 words[4]={as_type<ushort4>(pair0.xy),as_type<ushort4>(pair0.zw),as_type<ushort4>(pair1.xy),as_type<ushort4>(pair1.zw)};
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(words[j])<<16);
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

kernel void sole_specialized_bf16_decode(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=ulong(fixed_columns))return;
    float4 sum0=0.0f,sum1=0.0f;
    if(ulong(fixed_inner)%4==0) {
        for(ulong k=ulong(lane)*4;k<ulong(fixed_inner);k+=128) {
            float4 x=*(device const float4*)(a+k),y=ulong(fixed_rows)==2?*(device const float4*)(a+ulong(fixed_aStride)+k):float4(0.0f);
            ulong packed=(column/4)*ulong(fixed_inner)*4+k*4;
            uint4 pair0=*(device const uint4*)(weight+packed),pair1=*(device const uint4*)(weight+packed+8);
            ushort4 words[4]={as_type<ushort4>(pair0.xy),as_type<ushort4>(pair0.zw),as_type<ushort4>(pair1.xy),as_type<ushort4>(pair1.zw)};
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(words[j])<<16);
                sum0+=x[j]*w;if(ulong(fixed_rows)==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<ulong(fixed_inner);k+=32) {
        float x=a[k],y=ulong(fixed_rows)==2?a[ulong(fixed_aStride)+k]:0.0f;
        float4 w=0.0f;
        w.x=as_type<float>(uint(weight[column*ulong(fixed_inner)+k])<<16);
        if(column+1<ulong(fixed_columns))w.y=as_type<float>(uint(weight[(column+1)*ulong(fixed_inner)+k])<<16);
        if(column+2<ulong(fixed_columns))w.z=as_type<float>(uint(weight[(column+2)*ulong(fixed_inner)+k])<<16);
        if(column+3<ulong(fixed_columns))w.w=as_type<float>(uint(weight[(column+3)*ulong(fixed_inner)+k])<<16);
        sum0+=x*w;if(ulong(fixed_rows)==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<ulong(fixed_columns);++j){out[column+j]=sum0[j];if(ulong(fixed_rows)==2)out[ulong(fixed_oStride)+column+j]=sum1[j];}
}

kernel void sole_bf16_restore(device const ushort* input [[buffer(0)]],device ushort* out [[buffer(1)]],
 constant ulong2& shape [[buffer(2)]],uint id [[thread_position_in_grid]]) {
 ulong at=id;if(at>=shape.x*shape.y)return;
 ulong row=at/shape.y,k=at%shape.y;out[at]=input[(row/4)*shape.y*4+k*4+row%4];
}
kernel void sole_bf16_read(device const ushort* input [[buffer(0)]],device uint* out [[buffer(1)]],
 constant ulong4& p [[buffer(2)]],uint id [[thread_position_in_grid]]) {
 ulong i=id;if(i>=p.w)return;ulong at=p.z+i,row=at/p.y,k=at%p.y;
 out[i]=uint(input[(row/4)*p.y*4+k*4+row%4])<<16;
}


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

struct SoleDualWords {ushort4 words[4];};
inline SoleDualWords sole_dual_words(device const ushort* weight,ulong packed){
 uint4 pair0=*(device const uint4*)(weight+packed),pair1=*(device const uint4*)(weight+packed+8);
 SoleDualWords value;value.words[0]=as_type<ushort4>(pair0.xy);value.words[1]=as_type<ushort4>(pair0.zw);value.words[2]=as_type<ushort4>(pair1.xy);value.words[3]=as_type<ushort4>(pair1.zw);return value;
}
kernel void sole_dual_specialized_decode(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],device const ushort* upweight [[buffer(4)]],device float* upout [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=ulong(fixed_columns))return;
    float4 sum0=0.0f,sum1=0.0f;float4 up0=0.0f,up1=0.0f;
    if(ulong(fixed_inner)%4==0) {
        for(ulong k=ulong(lane)*4;k<ulong(fixed_inner);k+=128) {
            float4 x=*(device const float4*)(a+k),y=ulong(fixed_rows)==2?*(device const float4*)(a+ulong(fixed_aStride)+k):float4(0.0f);
            ulong packed=(column/4)*ulong(fixed_inner)*4+k*4;
            SoleDualWords gateWords=sole_dual_words(weight,packed),upWords=sole_dual_words(upweight,packed);
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(gateWords.words[j])<<16);
                sum0+=x[j]*w;if(ulong(fixed_rows)==2)sum1+=y[j]*w;
                float4 uw=as_type<float4>(uint4(upWords.words[j])<<16);up0+=x[j]*uw;if(ulong(fixed_rows)==2)up1+=y[j]*uw;
            }
        }
    } else {
    for(ulong k=lane;k<ulong(fixed_inner);k+=32) {
        float x=a[k],y=ulong(fixed_rows)==2?a[ulong(fixed_aStride)+k]:0.0f;
        float4 w=0.0f;
        w.x=as_type<float>(uint(weight[column*ulong(fixed_inner)+k])<<16);
        if(column+1<ulong(fixed_columns))w.y=as_type<float>(uint(weight[(column+1)*ulong(fixed_inner)+k])<<16);
        if(column+2<ulong(fixed_columns))w.z=as_type<float>(uint(weight[(column+2)*ulong(fixed_inner)+k])<<16);
        if(column+3<ulong(fixed_columns))w.w=as_type<float>(uint(weight[(column+3)*ulong(fixed_inner)+k])<<16);
        float4 uw=0.0f;
        uw.x=as_type<float>(uint(upweight[column*ulong(fixed_inner)+k])<<16);
        if(column+1<ulong(fixed_columns))uw.y=as_type<float>(uint(upweight[(column+1)*ulong(fixed_inner)+k])<<16);
        if(column+2<ulong(fixed_columns))uw.z=as_type<float>(uint(upweight[(column+2)*ulong(fixed_inner)+k])<<16);
        if(column+3<ulong(fixed_columns))uw.w=as_type<float>(uint(upweight[(column+3)*ulong(fixed_inner)+k])<<16);
        sum0+=x*w;if(ulong(fixed_rows)==2)sum1+=y*w;
        up0+=x*uw;if(ulong(fixed_rows)==2)up1+=y*uw;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    up0=float4(simd_sum(up0.x),simd_sum(up0.y),simd_sum(up0.z),simd_sum(up0.w));
    up1=float4(simd_sum(up1.x),simd_sum(up1.y),simd_sum(up1.z),simd_sum(up1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<ulong(fixed_columns);++j){out[column+j]=sum0[j];if(ulong(fixed_rows)==2)out[ulong(fixed_oStride)+column+j]=sum1[j];upout[column+j]=up0[j];if(ulong(fixed_rows)==2)upout[ulong(fixed_oStride)+column+j]=up1[j];}
}

kernel void sole_dual_read_pair(device const ushort* first [[buffer(0)]],device const ushort* second [[buffer(1)]],device uint* output [[buffer(2)]],constant ulong2& p [[buffer(3)]],uint id [[thread_position_in_grid]]){
 ulong i=id;if(i>=p.y)return;ulong packed=(p.x+i)*16;SoleDualWords g=sole_dual_words(first,packed),u=sole_dual_words(second,packed);
 for(uint k=0;k<4;++k)for(uint c=0;c<4;++c){output[i*32+k*4+c]=uint(g.words[k][c])<<16;output[i*32+16+k*4+c]=uint(u.words[k][c])<<16;}
}
