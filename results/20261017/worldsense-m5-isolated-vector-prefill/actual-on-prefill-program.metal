#define GARNET_ATTENTION_PREFIX 1

#include <metal_stdlib>
using namespace metal;
struct BF16Params {ulong rows,inner,columns,aStride,oStride;};
struct HalfRescaleParams {ulong rows,columns,iStride,oStride;};
kernel void half_rescale(device const float* input [[buffer(0)]],device const int* scales [[buffer(1)]],
    device float* output [[buffer(2)]],constant HalfRescaleParams& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,column=id%p.columns;
    output[row*p.oStride+column]=ldexp(input[row*p.iStride+column],-scales[column]);
}
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
kernel void bf16_decode_small(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*4+simd*2;
    if(column>=p.columns)return;
    float2 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            ushort4 b0=*(device const ushort4*)(weight+column*p.inner+k),b1=ushort4(0);
            if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
            for(uint j=0;j<4;++j) {
                float2 w=as_type<float2>(uint2(b0[j],b1[j])<<16);
                sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<p.inner;k+=32) {
        float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
        float2 w=0.0f;
        w.x=as_type<float>(uint(weight[column*p.inner+k])<<16);
        if(column+1<p.columns)w.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
        sum0+=x*w;if(p.rows==2)sum1+=y*w;
    }
    }
    sum0=float2(simd_sum(sum0.x),simd_sum(sum0.y));
    sum1=float2(simd_sum(sum1.x),simd_sum(sum1.y));
    if(lane==0)for(uint j=0;j<2&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
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
struct SplitParams {ulong rows,inner,columns,aStride,oStride,parts;};
kernel void bf16_split_key(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant SplitParams& p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float left[16*32],right[32*32],result[16*32];
    ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*32,begin=ulong(group.z)*64,end=min(begin+64,p.inner);
    simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
    simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(ulong base=begin;base<end;base+=32) {
        for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,k=base+i%32;left[i]=(row<p.rows&&k<end)?a[row*p.aStride+k]:0.0f;}
        for(uint i=tid;i<32*32;i+=128){ulong k=base+i/32,column=colBase+i%32;right[i]=(k<end&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}
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
    for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,column=colBase+i%32;if(row<p.rows&&column<p.columns)out[(ulong(group.z)*p.rows+row)*p.oStride+column]=result[i];}
}
struct SplitReduceParams {ulong rows,columns,iStride,oStride,parts;};
kernel void bf16_split_reduce(device const float* partial [[buffer(0)]],device float* out [[buffer(1)]],
    constant SplitReduceParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=ulong(id)/p.columns,col=ulong(id)%p.columns;
    float terms[16];for(uint i=0;i<16;++i)terms[i]=i<p.parts?partial[(ulong(i)*p.rows+row)*p.iStride+col]:0.0f;
    for(uint width=8;width>0;width/=2)for(uint i=0;i<width;++i)terms[i]=terms[2*i]+terms[2*i+1];
    out[row*p.oStride+col]=terms[0];
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
struct ArgmaxParams {ulong rows,columns,iStride,oStride;};
kernel void argmax_rows(device const float* input [[buffer(0)]],device float* output [[buffer(1)]],
    constant ArgmaxParams& p [[buffer(2)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    threadgroup float maximum[4];threadgroup uint indices[4];threadgroup uint invalid[4];
    float best=-INFINITY;uint index=0xffffffff,bad=0;
    for(ulong j=tid;j<p.columns;j+=128) {
        float value=input[row*p.iStride+j];bad|=!isfinite(value);
        if(value>best||(value==best&&j<index)){best=value;index=uint(j);}
    }
    float groupBest=simd_max(best);uint groupIndex=simd_min(best==groupBest?index:0xffffffff);
    uint groupBad=simd_max(bad);
    if(lane==0){maximum[simd]=groupBest;indices[simd]=groupIndex;invalid[simd]=groupBad;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    best=lane<4?maximum[lane]:-INFINITY;index=lane<4?indices[lane]:0xffffffff;bad=lane<4?invalid[lane]:0;
    groupBest=simd_max(best);groupIndex=simd_min(best==groupBest?index:0xffffffff);groupBad=simd_max(bad);
    if(tid==0){output[row*p.oStride]=float(groupIndex);output[row*p.oStride+1]=float(groupBad!=0);}
}
struct Params { ulong rows,columns,aStride,bStride,cStride;float epsilon,offset; };
kernel void rms_norm(device const float* a [[buffer(0)]],device const float* weight [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    threadgroup float partial[4];
    float sum=0;
    for(ulong j=tid;j<p.columns;j+=128){float x=a[row*p.aStride+j];sum+=x*x;}
    sum=simd_sum(sum);
    if(lane==0)partial[simd]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(lane<4?partial[lane]:0.0f);
    float inverse=rsqrt(sum/float(p.columns)+p.epsilon);
    for(ulong j=tid;j<p.columns;j+=128)c[row*p.cStride+j]=a[row*p.aStride+j]*inverse*(weight[j]+p.offset);
}
kernel void residual_norm(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device const float* weight [[buffer(2)]],device float* residual [[buffer(3)]],
    device float* c [[buffer(4)]],constant Params& p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.rows)return;
    threadgroup float partial[4];
    float sum=0;
    for(ulong j=tid;j<p.columns;j+=128){float x=a[row*p.aStride+j]+b[row*p.cStride+j];residual[row*p.cStride+j]=x;sum+=x*x;}
    sum=simd_sum(sum);
    if(lane==0)partial[simd]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    sum=simd_sum(lane<4?partial[lane]:0.0f);
    float inverse=rsqrt(sum/float(p.columns)+p.epsilon);
    for(ulong j=tid;j<p.columns;j+=128)c[row*p.cStride+j]=residual[row*p.cStride+j]*inverse*(weight[j]+p.offset);
}
struct PackedParams {ulong rows,columns,packedStride,normStride,chunk;float epsilon,offset;};
kernel void packed_residual_norm(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
 device const float* weight [[buffer(2)]],device float* packed [[buffer(3)]],device float* layout [[buffer(4)]],
 device float* out [[buffer(5)]],constant PackedParams& p [[buffer(6)]],uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 if(row>=p.rows)return;
 ulong base=(row/p.chunk)*p.packedStride+(row%p.chunk)*p.columns;
 threadgroup float partial[4];float sum=0;
 for(ulong j=tid;j<p.columns;j+=128){float x=a[base+j]+b[base+j];packed[base+j]=x;layout[row*p.normStride+j]=x;sum+=x*x;}
 sum=simd_sum(sum);if(lane==0)partial[simd]=sum;
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
 sum=simd_sum(lane<4?partial[lane]:0.0f);
 float inverse=rsqrt(sum/float(p.columns)+p.epsilon);
 for(ulong j=tid;j<p.columns;j+=128)out[row*p.normStride+j]=layout[row*p.normStride+j]*inverse*(weight[j]+p.offset);
}
kernel void silu_mul(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    float gate=a[row*p.aStride+col];
    c[row*p.cStride+col]=(gate/(1.0f+exp(-gate)))*b[row*p.bStride+col];
}
struct JoinedGlobalSiluParams {ulong rows,columns,iStride,gStride,uStride,oStride;};
kernel void joined_global_silu(device const float* joined [[buffer(0)]],device float* gates [[buffer(1)]],
    device float* ups [[buffer(2)]],device float* activated [[buffer(3)]],
    constant JoinedGlobalSiluParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    float gate=joined[row*p.iStride+col],up=joined[row*p.iStride+p.columns+col];
    gates[row*p.gStride+col]=gate;ups[row*p.uStride+col]=up;
    activated[row*p.oStride+col]=(gate/(1.0f+exp(-gate)))*up;
}
kernel void add_matrix(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    c[row*p.cStride+col]=a[row*p.aStride+col]+b[(p.offset!=0?0:row)*p.bStride+col];
}
kernel void mask_rows(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],constant Params& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;
    c[row*p.cStride+col]=b[row*p.bStride]!=0.0f?a[row*p.aStride+col]:0.0f;
}
struct RopeParams {ulong batch,heads,dim,rotary,iStride,pStride,fStride,oStride;float epsilon,weightOffset;ulong normalize;};
kernel void norm_rope(device const float* input [[buffer(0)]],device const float* weight [[buffer(1)]],
    device const float* positions [[buffer(2)]],device const float* frequencies [[buffer(3)]],
    device float* output [[buffer(4)]],constant RopeParams& p [[buffer(5)]],uint id [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    if(id>=p.batch*p.heads)return;
    ulong row=id/p.heads,head=id%p.heads,base=row*p.iStride+head*p.dim;
    float sum=0.0f;
    for(ulong j=lane;j<p.dim;j+=32){float x=input[base+j];sum+=x*x;}
    sum=simd_sum(sum);
    float inverse=p.normalize?rsqrt(sum/float(p.dim)+p.epsilon):1.0f;
    for(ulong j=lane;j<p.dim;j+=32) {
        float x=input[base+j]*inverse*(p.normalize?(p.weightOffset+weight[j]):1.0f);
        if(j<p.rotary) {
            ulong halfWidth=p.rotary/2,frequency=j%halfWidth,other=j<halfWidth?j+halfWidth:j-halfWidth;
            ulong axis=ulong(frequencies[p.fStride+frequency]);
            float angle=positions[row*p.pStride+axis]*frequencies[frequency];
            float y=input[base+other]*inverse*(p.normalize?(p.weightOffset+weight[other]):1.0f);
            x=x*precise::cos(angle)+(j<halfWidth?-y:y)*precise::sin(angle);
        }
        output[row*p.oStride+head*p.dim+j]=x;
    }
}
struct KVParams {ulong batch,heads,dim,capacity,tStride,cStride,sStride,aStride,oStride;};
kernel void kv_update(device const float* token [[buffer(0)]],device const float* cache [[buffer(1)]],
    device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],
    device float* output [[buffer(4)]],constant KVParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    ulong width=p.heads*p.capacity*p.dim;
    if(id>=p.batch*width)return;
    ulong row=id/width,col=id%width,dim=col%p.dim,slot=(col/p.dim)%p.capacity,head=col/(p.capacity*p.dim);
    bool replace=active[row*p.aStride]!=0.0f&&float(slot)==slots[row*p.sStride];
    output[row*p.oStride+col]=replace?token[row*p.tStride+head*p.dim+dim]:cache[row*p.cStride+col];
}
// A session-owned cache already contains every untouched slot.
kernel void kv_update_inplace(device const float* token [[buffer(0)]],device const float* cache [[buffer(1)]],
 device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
 constant KVParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
 ulong width=p.heads*p.dim;if(id>=p.batch*width)return;
 ulong row=id/width,head=(id%width)/p.dim,d=id%p.dim;
 if(active[row*p.aStride]==0.0f)return;
 float slot=slots[row*p.sStride];
 if(!isfinite(slot)||slot<0.0f||slot>=float(p.capacity))return;
 output[row*p.oStride+(head*p.capacity+ulong(slot))*p.dim+d]=token[row*p.tStride+head*p.dim+d];
}
struct KVChunkParams {ulong batch,heads,dim,capacity,chunk,tStride,cStride,sStride,aStride,oStride,contiguous;};
// Bit-preserving cache copies for aligned dimensions fitting uint indices.
kernel void kv_update_vector4(device const float* token [[buffer(0)]],device const float* cache [[buffer(1)]],
    device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],
    device float* output [[buffer(4)]],constant KVParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    const uint dim=uint(p.dim/4),capacity=uint(p.capacity),width=uint(p.heads)*capacity*dim;
    if(id>=uint(p.batch)*width)return;
    uint row=id/width,col=id%width,d=col%dim,slot=(col/dim)%capacity,head=col/(capacity*dim);
    bool replace=active[row*uint(p.aStride)]!=0.0f&&float(slot)==slots[row*uint(p.sStride)];
    device const float4* source=(device const float4*)token;
    device const float4* previous=(device const float4*)cache;
    device float4* next=(device float4*)output;
    next[row*uint(p.oStride/4)+col]=replace?source[row*uint(p.tStride/4)+head*dim+d]:previous[row*uint(p.cStride/4)+col];
}
kernel void kv_chunk(device const float* token [[buffer(0)]],device const float* cache [[buffer(1)]],
    device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
    constant KVChunkParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    ulong width=p.heads*p.capacity*p.dim;if(id>=p.batch*width)return;
    ulong row=id/width,col=id%width,d=col%p.dim,s=(col/p.dim)%p.capacity,h=col/(p.capacity*p.dim);
    int latest=-1;
    if(p.contiguous) {
        if(active[row*p.aStride]!=0.0f) {
            int candidate=int(s)-int(slots[row*p.sStride]);
            if(candidate>=0&&ulong(candidate)<p.chunk&&active[row*p.aStride+ulong(candidate)]!=0.0f)latest=candidate;
        }
    } else {
        for(uint t=0;t<p.chunk;++t)if(active[row*p.aStride+t]!=0.0f&&slots[row*p.sStride+t]==float(s))latest=int(t);
    }
    output[row*p.oStride+col]=latest<0?cache[row*p.cStride+col]:token[(row*p.chunk+ulong(latest))*p.tStride+h*p.dim+d];
}
kernel void kv_chunk_vector4(device const float* token [[buffer(0)]],device const float* cache [[buffer(1)]],
    device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
    constant KVChunkParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    const uint dim=uint(p.dim/4),capacity=uint(p.capacity),chunk=uint(p.chunk),width=uint(p.heads)*capacity*dim;
    if(id>=uint(p.batch)*width)return;
    uint row=id/width,col=id%width,d=col%dim,s=(col/dim)%capacity,h=col/(capacity*dim);
    int latest=-1;
    if(p.contiguous) {
        if(active[row*uint(p.aStride)]!=0.0f) {
            int candidate=int(s)-int(slots[row*uint(p.sStride)]);
            if(candidate>=0&&uint(candidate)<chunk&&active[row*uint(p.aStride)+uint(candidate)]!=0.0f)latest=candidate;
        }
    } else {
        for(uint t=0;t<chunk;++t)if(active[row*uint(p.aStride)+t]!=0.0f&&slots[row*uint(p.sStride)+t]==float(s))latest=int(t);
    }
    device const float4* source=(device const float4*)token;
    device const float4* previous=(device const float4*)cache;
    device float4* next=(device float4*)output;
    next[row*uint(p.oStride/4)+col]=latest<0?previous[row*uint(p.cStride/4)+col]:source[(row*chunk+uint(latest))*uint(p.tStride/4)+h*dim+d];
}
struct ChunkAttentionParams {ulong batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,kcStride,vcStride,sStride,aStride,gStride,oStride,gated;};
kernel void attention_chunk(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],device const float* keyCache [[buffer(3)]],device const float* valueCache [[buffer(4)]],
    device const float* slots [[buffer(5)]],device const float* active [[buffer(6)]],device const float* gate [[buffer(7)]],
    device float* output [[buffer(8)]],constant ChunkAttentionParams& p [[buffer(9)]],
    threadgroup float* scratch [[threadgroup(0)]],threadgroup int* selected [[threadgroup(1)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong flatRow=id/p.heads,head=id%p.heads,row=flatRow/p.chunk,token=flatRow%p.chunk,kv=head/(p.heads/p.kvHeads);
    if(flatRow>=p.batch*p.chunk)return;
    if(active[row*p.aStride+token]==0.0f){for(ulong d=tid;d<p.dim;d+=128)output[flatRow*p.oStride+head*p.dim+d]=0.0f;return;}
    const ulong visible=min(p.capacity,ulong(slots[row*p.sStride+token])+1);
    threadgroup float partial[4];
#ifdef GARNET_ATTENTION_PREFIX
    threadgroup int contiguousPrefix;
    if(tid==0) {
        float first=slots[row*p.sStride];
        int valid=isfinite(first)&&first>=0.0f&&first==floor(first);
        for(ulong t=0;t<=token;++t)if(active[row*p.aStride+t]==0.0f||slots[row*p.sStride+t]!=first+float(t))valid=0;
        contiguousPrefix=valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    for(ulong s=simd;s<visible;s+=4) {
#ifdef GARNET_ATTENTION_PREFIX
        int latest=-1;
        if(contiguousPrefix) {
            float first=slots[row*p.sStride];
            if(float(s)>=first)latest=int(float(s)-first);
        } else for(uint t=0;t<=token;++t)if(active[row*p.aStride+t]!=0.0f&&slots[row*p.sStride+t]==float(s))latest=int(t);
#else
        int latest=-1;for(uint t=0;t<=token;++t)if(active[row*p.aStride+t]!=0.0f&&slots[row*p.sStride+t]==float(s))latest=int(t);
#endif
        float score=-1e30f;
        if(float(s)<=slots[row*p.sStride+token]) {
            score=0.0f;
            for(ulong d=lane;d<p.dim;d+=32) {
                float k=latest<0?keyCache[row*p.kcStride+(kv*p.capacity+s)*p.dim+d]:key[(row*p.chunk+ulong(latest))*p.kStride+kv*p.dim+d];
                score+=query[flatRow*p.qStride+head*p.dim+d]*k;
            }
            score=simd_sum(score)*rsqrt(float(p.dim));
        }
        if(lane==0){scratch[s]=score;selected[s]=latest;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=0.0f;
    for(ulong s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
    denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denominator=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong d=tid;d<p.dim;d+=128) {
        float sum=0.0f;
        for(ulong s=0;s<visible;++s){int latest=selected[s];float v=latest<0?valueCache[row*p.vcStride+(kv*p.capacity+s)*p.dim+d]:value[(row*p.chunk+ulong(latest))*p.vStride+kv*p.dim+d];sum+=scratch[s]/denominator*v;}
        output[flatRow*p.oStride+head*p.dim+d]=p.gated?sum/(1.0f+exp(-gate[flatRow*p.gStride+head*p.dim+d])):sum;
    }
}
struct ScoresParams {ulong batch,heads,kvHeads,dim,capacity,qStride,kStride,sStride,aStride,oStride;};
kernel void attention_scores_serial(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
    constant ScoresParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.heads)return;
    ulong row=id/p.heads,head=id%p.heads,kv=head/(p.heads/p.kvHeads),base=row*p.oStride+head*p.capacity;
    if(active[row*p.aStride]==0.0f){for(ulong s=0;s<p.capacity;++s)output[base+s]=0.0f;return;}
    float maximum=-3.402823466e38f;
    for(ulong s=0;s<p.capacity;++s) {
        float score=-1e30f;
        if(float(s)<=slots[row*p.sStride]) {
            score=0.0f;
            for(ulong j=0;j<p.dim;++j)score+=query[row*p.qStride+head*p.dim+j]*key[row*p.kStride+(kv*p.capacity+s)*p.dim+j];
            score*=rsqrt(float(p.dim));
        }
        output[base+s]=score;maximum=max(maximum,score);
    }
    float denominator=0.0f;
    for(ulong s=0;s<p.capacity;++s){float probability=exp(output[base+s]-maximum);output[base+s]=probability;denominator+=probability;}
    for(ulong s=0;s<p.capacity;++s)output[base+s]/=denominator;
}
kernel void attention_scores(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* slots [[buffer(2)]],device const float* active [[buffer(3)]],device float* output [[buffer(4)]],
    constant ScoresParams& p [[buffer(5)]],threadgroup float* scratch [[threadgroup(0)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(id>=p.batch*p.heads)return;
    ulong row=id/p.heads,head=id%p.heads,kv=head/(p.heads/p.kvHeads),base=row*p.oStride+head*p.capacity;
    if(active[row*p.aStride]==0.0f){for(ulong s=tid;s<p.capacity;s+=128)output[base+s]=0.0f;return;}
    const ulong visible=min(p.capacity,ulong(slots[row*p.sStride])+1);
    threadgroup float partial[4];
    for(ulong s=simd;s<visible;s+=4) {
        float score=-1e30f;
        if(float(s)<=slots[row*p.sStride]) {
            score=0.0f;
            for(ulong j=lane;j<p.dim;j+=32)score+=query[row*p.qStride+head*p.dim+j]*key[row*p.kStride+(kv*p.capacity+s)*p.dim+j];
            score=simd_sum(score)*rsqrt(float(p.dim));
        }
        if(lane==0)scratch[s]=score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong s=tid;s<visible;s+=128)maximum=max(maximum,scratch[s]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    // All threads consume the max before partial storage is reused for sums.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator=0.0f;
    for(ulong s=tid;s<visible;s+=128){float probability=exp(scratch[s]-maximum);scratch[s]=probability;denominator+=probability;}
    denominator=simd_sum(denominator);if(lane==0)partial[simd]=denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    denominator=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong s=tid;s<p.capacity;s+=128)output[base+s]=s<visible?scratch[s]/denominator:0.0f;
}
struct AttentionParams {ulong batch,heads,kvHeads,dim,capacity,pStride,vStride,gStride,sStride,oStride,gated;};
kernel void attention_context(device const float* probabilities [[buffer(0)]],device const float* value [[buffer(1)]],
    device const float* gate [[buffer(2)]],device const float* slots [[buffer(3)]],device float* output [[buffer(4)]],constant AttentionParams& p [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.heads*p.dim)return;
    ulong row=id/(p.heads*p.dim),head=(id/p.dim)%p.heads,col=id%p.dim,kv=head/(p.heads/p.kvHeads);
    const float slot=slots[row*p.sStride];
    const ulong visible=isfinite(slot)&&slot>=0.0f&&slot<float(p.capacity)?ulong(slot)+1:p.capacity;
    float result=0.0f;
    for(ulong s=0;s<visible;++s)result+=probabilities[row*p.pStride+head*p.capacity+s]*value[row*p.vStride+(kv*p.capacity+s)*p.dim+col];
    output[row*p.oStride+head*p.dim+col]=p.gated?result/(1.0f+exp(-gate[row*p.gStride+head*p.dim+col])):result;
}
struct PrepareParams {ulong batch,keyHeads,valueHeads,keyDim,valueDim,kSize,cStride,aStride,bStride,oStride,width;};
kernel void linear_prepare(device const float* conv [[buffer(0)]],device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]],device const float* aLog [[buffer(3)]],device const float* dtBias [[buffer(4)]],
    device float* output [[buffer(5)]],constant PrepareParams& p [[buffer(6)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.width)return;
    ulong row=id/p.width,col=id%p.width,keyWidth=p.valueHeads*p.keyDim,valueWidth=p.valueHeads*p.valueDim;
    float result;
    if(col<2*keyWidth) {
        ulong part=col/keyWidth,within=col%keyWidth,head=within/p.keyDim,dim=within%p.keyDim;
        ulong channel=part*p.keyHeads*p.keyDim+(head/(p.valueHeads/p.keyHeads))*p.keyDim+dim;
        result=conv[row*p.cStride+channel*(p.kSize+1)];
    }else if(col<2*keyWidth+valueWidth) {
        ulong channel=2*p.keyHeads*p.keyDim+col-2*keyWidth;
        result=conv[row*p.cStride+channel*(p.kSize+1)];
    }else if(col<2*keyWidth+valueWidth+p.valueHeads) {
        ulong head=col-2*keyWidth-valueWidth;
        float x=a[row*p.aStride+head]+dtBias[head];
        float softplus=max(x,0.0f)+log(1.0f+exp(-abs(x)));
        result=-exp(aLog[head])*softplus;
    }else {
        ulong head=col-2*keyWidth-valueWidth-p.valueHeads;
        result=1.0f/(1.0f+exp(-b[row*p.bStride+head]));
    }
    output[row*p.oStride+col]=result;
}
struct GateParams {ulong batch,heads,valueDim,cStride,zStride,oStride,columnOffset;float epsilon;};
kernel void gated_norm(device const float* core [[buffer(0)]],device const float* z [[buffer(1)]],
    device const float* norm [[buffer(2)]],device float* output [[buffer(3)]],
    constant GateParams& p [[buffer(4)]],uint id [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    if(id>=p.batch*p.heads)return;
    ulong row=id/p.heads,head=id%p.heads;
    float sum=0.0f;
    for(ulong j=lane;j<p.valueDim;j+=32){float x=core[row*p.cStride+head*p.valueDim+j];sum+=x*x;}
    sum=simd_sum(sum);
    float inverse=rsqrt(sum/float(p.valueDim)+p.epsilon);
    for(ulong j=lane;j<p.valueDim;j+=32) {
        float gate=z[row*p.zStride+head*p.valueDim+j];
        output[row*p.oStride+p.columnOffset+head*p.valueDim+j]=core[row*p.cStride+head*p.valueDim+j]*inverse*norm[j]*gate/(1.0f+exp(-gate));
    }
}
struct ConcatParams {ulong batch,aWidth,bWidth,cWidth,aStride,bStride,cStride,oStride;};
kernel void concat_rows(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],device float* output [[buffer(3)]],
    constant ConcatParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    ulong width=p.aWidth+p.bWidth+p.cWidth;
    if(id>=p.batch*width)return;
    ulong row=id/width,col=id%width;
    output[row*p.oStride+col]=col<p.aWidth?a[row*p.aStride+col]:
        (col<p.aWidth+p.bWidth?b[row*p.bStride+col-p.aWidth]:c[row*p.cStride+col-p.aWidth-p.bWidth]);
}
struct SliceParams {ulong batch,groups,sourceWidth,start,count,iStride,oStride;};
kernel void slice_groups(device const float* input [[buffer(0)]],device float* output [[buffer(1)]],
    constant SliceParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.groups*p.count)return;
    ulong col=id%p.count,group=(id/p.count)%p.groups,row=id/(p.count*p.groups);
    output[row*p.oStride+group*p.count+col]=input[row*p.iStride+group*p.sourceWidth+p.start+col];
}
kernel void slice_groups_vector4(device const float4* input [[buffer(0)]],device float4* output [[buffer(1)]],
    constant SliceParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]) {
    uint count=uint(p.count/4),groups=uint(p.groups);
    if(id>=uint(p.batch)*groups*count)return;
    uint col=id%count,group=(id/count)%groups,row=id/(count*groups);
    output[row*uint(p.oStride/4)+group*count+col]=input[row*uint(p.iStride/4)+group*uint(p.sourceWidth/4)+uint(p.start/4)+col];
}
struct ConvParams {ulong batch,channels,kSize,tStride,wStride,sStride,aStride,oStride;};
kernel void causal_conv(device const float* token [[buffer(0)]],device const float* weight [[buffer(1)]],
    device const float* state [[buffer(2)]],device const float* active [[buffer(3)]],
    device float* packed [[buffer(4)]],constant ConvParams& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.channels)return;
    ulong row=id/p.channels,col=id%p.channels;
    bool enabled=active[row*p.aStride]!=0.0f;
    float sum=0.0f;
    for(ulong j=0;j<p.kSize;++j) {
        float next=j+1==p.kSize?token[row*p.tStride+col]:state[row*p.sStride+col*p.kSize+j+1];
        packed[row*p.oStride+col*(p.kSize+1)+1+j]=enabled?next:state[row*p.sStride+col*p.kSize+j];
        sum+=next*weight[col*p.wStride+j];
    }
    packed[row*p.oStride+col*(p.kSize+1)]=enabled?sum/(1.0f+exp(-sum)):0.0f;
}
struct ConvChunkParams {ulong batch,chunk,channels,kSize,tStride,wStride,sStride,aStride,oStride,nStride;};
kernel void causal_conv_chunk(device const float* tokens [[buffer(0)]],device const float* weight [[buffer(1)]],
    device const float* state [[buffer(2)]],device const float* active [[buffer(3)]],
    device float* output [[buffer(4)]],device float* nextState [[buffer(5)]],
    constant ConvChunkParams& p [[buffer(6)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.batch*p.channels)return;
    ulong row=id/p.channels,col=id%p.channels;
    float history[8];
    for(ulong j=0;j<p.kSize;++j)history[j]=state[row*p.sStride+col*p.kSize+j];
    for(ulong token=0;token<p.chunk;++token) {
        bool enabled=active[row*p.aStride+token]!=0.0f;
        float sum=0.0f;
        for(ulong j=0;j<p.kSize;++j) {
            float next=j+1==p.kSize?tokens[(row*p.chunk+token)*p.tStride+col]:history[j+1];
            if(enabled)history[j]=next;
            sum+=next*weight[col*p.wStride+j];
        }
        output[(row*p.chunk+token)*p.oStride+col]=enabled?sum/(1.0f+exp(-sum)):0.0f;
    }
    for(ulong j=0;j<p.kSize;++j)nextState[row*p.nStride+col*p.kSize+j]=history[j];
}
struct DeltaParams {ulong batch,heads,keyDim,valueDim,qStride,kStride,vStride,gStride,bStride,sStride,aStride,oStride;};
kernel void gated_delta(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],device const float* decay [[buffer(3)]],
    device const float* beta [[buffer(4)]],device const float* state [[buffer(5)]],
    device const float* active [[buffer(6)]],device float* packed [[buffer(7)]],
    constant DeltaParams& p [[buffer(8)]],uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(id>=p.batch*p.heads)return;
    ulong head=id%p.heads,row=id/p.heads;
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim;
    ulong outputBase=row*p.oStride+head*(p.keyDim+1)*p.valueDim;
    ulong outBase=outputBase+p.valueDim;
    if(active[row*p.aStride]==0.0f) {
        for(ulong column=tid;column<p.valueDim;column+=128) {
            packed[outputBase+column]=0.0f;
            for(ulong j=0;j<p.keyDim;++j)packed[outBase+j*p.valueDim+column]=state[stateBase+j*p.valueDim+column];
        }
        return;
    }
    threadgroup float qPartial[4],kPartial[4];
    float qs=0.0f,ks=0.0f;
    for(ulong j=tid;j<p.keyDim;j+=128) {
        float q=query[row*p.qStride+head*p.keyDim+j],k=key[row*p.kStride+head*p.keyDim+j];qs+=q*q;ks+=k*k;
    }
    qs=simd_sum(qs);ks=simd_sum(ks);if(lane==0){qPartial[simd]=qs;kPartial[simd]=ks;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qs=simd_sum(lane<4?qPartial[lane]:0.0f)+1e-6f;ks=simd_sum(lane<4?kPartial[lane]:0.0f)+1e-6f;
    float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks);
    for(ulong column=tid;column<p.valueDim;column+=128) {
    float factor=exp(decay[row*p.gStride+head]),memory=0.0f;
    for(ulong j=0;j<p.keyDim;++j)
        memory+=state[stateBase+j*p.valueDim+column]*factor*key[row*p.kStride+head*p.keyDim+j]*ki;
    float change=(value[row*p.vStride+head*p.valueDim+column]-memory)*beta[row*p.bStride+head];
    float result=0.0f;
    for(ulong j=0;j<p.keyDim;++j) {
        float next=state[stateBase+j*p.valueDim+column]*factor+key[row*p.kStride+head*p.keyDim+j]*ki*change;
        packed[outBase+j*p.valueDim+column]=next;
        result+=next*query[row*p.qStride+head*p.keyDim+j]*qi;
    }
    packed[outputBase+column]=result;
    }
}
struct DeltaSplitParams {ulong batch,heads,keyDim,valueDim,qStride,kStride,vStride,gStride,bStride,sStride,aStride,oStride,nextStride;};
kernel void gated_delta_split(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],device const float* decay [[buffer(3)]],
    device const float* beta [[buffer(4)]],device const float* state [[buffer(5)]],
    device const float* active [[buffer(6)]],device float* resultOutput [[buffer(7)]],device float* nextState [[buffer(8)]],
    constant DeltaSplitParams& p [[buffer(9)]],uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(id>=p.batch*p.heads)return;
    ulong head=id%p.heads,row=id/p.heads;
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim;
    ulong outputBase=row*p.oStride+head*p.valueDim;
    ulong outBase=row*p.nextStride+head*p.keyDim*p.valueDim;
    if(active[row*p.aStride]==0.0f) {
        for(ulong column=tid;column<p.valueDim;column+=128) {
            resultOutput[outputBase+column]=0.0f;
            for(ulong j=0;j<p.keyDim;++j)nextState[outBase+j*p.valueDim+column]=state[stateBase+j*p.valueDim+column];
        }
        return;
    }
    threadgroup float qPartial[4],kPartial[4];
    float qs=0.0f,ks=0.0f;
    for(ulong j=tid;j<p.keyDim;j+=128) {
        float q=query[row*p.qStride+head*p.keyDim+j],k=key[row*p.kStride+head*p.keyDim+j];qs+=q*q;ks+=k*k;
    }
    qs=simd_sum(qs);ks=simd_sum(ks);if(lane==0){qPartial[simd]=qs;kPartial[simd]=ks;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qs=simd_sum(lane<4?qPartial[lane]:0.0f)+1e-6f;ks=simd_sum(lane<4?kPartial[lane]:0.0f)+1e-6f;
    float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks);
    for(ulong column=tid;column<p.valueDim;column+=128) {
    float factor=exp(decay[row*p.gStride+head]),memory=0.0f;
    for(ulong j=0;j<p.keyDim;++j)
        memory+=state[stateBase+j*p.valueDim+column]*factor*key[row*p.kStride+head*p.keyDim+j]*ki;
    float change=(value[row*p.vStride+head*p.valueDim+column]-memory)*beta[row*p.bStride+head];
    float result=0.0f;
    for(ulong j=0;j<p.keyDim;++j) {
        float next=state[stateBase+j*p.valueDim+column]*factor+key[row*p.kStride+head*p.keyDim+j]*ki*change;
        nextState[outBase+j*p.valueDim+column]=next;
        result+=next*query[row*p.qStride+head*p.keyDim+j]*qi;
    }
    resultOutput[outputBase+column]=result;
    }
}
kernel void gated_delta_split_parallel(device const float* query [[buffer(0)]],device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],device const float* decay [[buffer(3)]],
    device const float* beta [[buffer(4)]],device const float* state [[buffer(5)]],
    device const float* active [[buffer(6)]],device float* resultOutput [[buffer(7)]],device float* nextState [[buffer(8)]],
    constant DeltaSplitParams& p [[buffer(9)]],uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong tiles=p.valueDim/16,headId=ulong(id)/tiles;
    if(headId>=p.batch*p.heads)return;
    ulong head=headId%p.heads,row=headId/p.heads,column=(ulong(id)%tiles)*16+simd*4;
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim;
    ulong outputBase=row*p.oStride+head*p.valueDim;
    ulong outBase=row*p.nextStride+head*p.keyDim*p.valueDim;
    if(active[row*p.aStride]==0.0f) {
        for(ulong j=lane;j<p.keyDim;j+=32)
            *(device float4*)(nextState+outBase+j*p.valueDim+column)=*(device const float4*)(state+stateBase+j*p.valueDim+column);
        if(lane==0)*(device float4*)(resultOutput+outputBase+column)=float4(0.0f);
        return;
    }
    threadgroup float qPartial[4],kPartial[4];
    float qs=0.0f,ks=0.0f;
    for(ulong j=tid;j<p.keyDim;j+=128) {
        float q=query[row*p.qStride+head*p.keyDim+j],k=key[row*p.kStride+head*p.keyDim+j];qs+=q*q;ks+=k*k;
    }
    qs=simd_sum(qs);ks=simd_sum(ks);if(lane==0){qPartial[simd]=qs;kPartial[simd]=ks;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qs=simd_sum(lane<4?qPartial[lane]:0.0f)+1e-6f;ks=simd_sum(lane<4?kPartial[lane]:0.0f)+1e-6f;
    float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks);
    float factor=exp(decay[row*p.gStride+head]);float4 memory=0.0f;
    for(ulong j=lane;j<p.keyDim;j+=32)
        memory+=*(device const float4*)(state+stateBase+j*p.valueDim+column)*factor*key[row*p.kStride+head*p.keyDim+j]*ki;
    memory=float4(simd_sum(memory.x),simd_sum(memory.y),simd_sum(memory.z),simd_sum(memory.w));
    float4 change=(*(device const float4*)(value+row*p.vStride+head*p.valueDim+column)-memory)*beta[row*p.bStride+head];
    float4 result=0.0f;
    for(ulong j=lane;j<p.keyDim;j+=32) {
        float4 next=*(device const float4*)(state+stateBase+j*p.valueDim+column)*factor+key[row*p.kStride+head*p.keyDim+j]*ki*change;
        *(device float4*)(nextState+outBase+j*p.valueDim+column)=next;
        result+=next*query[row*p.qStride+head*p.keyDim+j]*qi;
    }
    result=float4(simd_sum(result.x),simd_sum(result.y),simd_sum(result.z),simd_sum(result.w));
    if(lane==0)*(device float4*)(resultOutput+outputBase+column)=result;
}
struct DeltaChunkParams {ulong batch,chunk,heads,keyDim,valueDim,pStride,sStride,aStride,oStride,nextStride;};
kernel void gated_delta_chunk(device const float* prepared [[buffer(0)]],device const float* state [[buffer(1)]],
    device const float* active [[buffer(2)]],device float* output [[buffer(3)]],device float* nextState [[buffer(4)]],
    constant DeltaChunkParams& p [[buffer(5)]],uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong tiles=p.valueDim/16,headId=ulong(id)/tiles;
    if(headId>=p.batch*p.heads)return;
    ulong head=headId%p.heads,row=headId/p.heads,column=(ulong(id)%tiles)*16+simd*4;
    ulong stateBase=row*p.sStride+head*p.keyDim*p.valueDim,nextBase=row*p.nextStride+head*p.keyDim*p.valueDim;
    float4 memories[4];
    for(uint j=0;j<4;++j)memories[j]=*(device const float4*)(state+stateBase+(ulong(lane)+32*j)*p.valueDim+column);
    threadgroup float qPartial[4],kPartial[4];
    ulong qw=p.heads*p.keyDim,vw=p.heads*p.valueDim;
    for(ulong token=0;token<p.chunk;++token) {
        ulong base=(row*p.chunk+token)*p.pStride,outputBase=(row*p.chunk+token)*p.oStride+head*p.valueDim;
        if(active[row*p.aStride+token]!=0.0f) {
            float q=prepared[base+head*p.keyDim+tid],k=prepared[base+qw+head*p.keyDim+tid];
            float qs=simd_sum(q*q),ks=simd_sum(k*k);
            if(lane==0){qPartial[simd]=qs;kPartial[simd]=ks;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            qs=simd_sum(lane<4?qPartial[lane]:0.0f)+1e-6f;ks=simd_sum(lane<4?kPartial[lane]:0.0f)+1e-6f;
            float qi=rsqrt(qs)*rsqrt(float(p.keyDim)),ki=rsqrt(ks),factor=exp(prepared[base+2*qw+vw+head]);
            float4 memory=0.0f;
            for(uint j=0;j<4;++j)memory+=memories[j]*factor*prepared[base+qw+head*p.keyDim+ulong(lane)+32*j]*ki;
            memory=float4(simd_sum(memory.x),simd_sum(memory.y),simd_sum(memory.z),simd_sum(memory.w));
            float4 change=(*(device const float4*)(prepared+base+2*qw+head*p.valueDim+column)-memory)*prepared[base+2*qw+vw+p.heads+head];
            float4 result=0.0f;
            for(uint j=0;j<4;++j) {
                ulong keyIndex=ulong(lane)+32*j;
                memories[j]=memories[j]*factor+prepared[base+qw+head*p.keyDim+keyIndex]*ki*change;
                result+=memories[j]*prepared[base+head*p.keyDim+keyIndex]*qi;
            }
            result=float4(simd_sum(result.x),simd_sum(result.y),simd_sum(result.z),simd_sum(result.w));
            if(lane==0)*(device float4*)(output+outputBase+column)=result;
        }else if(lane==0)*(device float4*)(output+outputBase+column)=float4(0.0f);
        // Every SIMD group must finish reading the partial norms before any
        // group can replace them for the next token.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<4;++j)*(device float4*)(nextState+nextBase+(ulong(lane)+32*j)*p.valueDim+column)=memories[j];
}
struct LayoutParams {ulong aRows,aColumns,aStride,bStride,oRows,oColumns,oStride,tokenWidth;};
kernel void chunk_layout(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
 device float* out [[buffer(2)]],constant LayoutParams& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.oRows*p.oColumns)return;
 ulong sourceRow=id/p.aColumns,sourceColumn=id%p.aColumns,row=id/p.oColumns,col=id%p.oColumns;
 float value=a[sourceRow*p.aStride+sourceColumn];
 if(p.tokenWidth&&b[sourceRow*p.bStride+sourceColumn/p.tokenWidth]==0.0f)value=0;
 out[row*p.oStride+col]=value;
}
struct VisionParams {ulong rows,columns,outRows,outColumns,aStride,bStride,cStride,oStride,tokens,hidden,heads,unit,mode;};
kernel void vision_norm(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    if(row>=p.outRows)return;threadgroup float partial[4];
    float mean=0.0f;for(ulong j=tid;j<p.columns;j+=128)mean+=a[row*p.aStride+j];
    mean=simd_sum(mean);if(lane==0)partial[simd]=mean;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean=simd_sum(lane<4?partial[lane]:0.0f)/float(p.columns);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float variance=0.0f;for(ulong j=tid;j<p.columns;j+=128){float x=a[row*p.aStride+j]-mean;variance+=x*x;}
    variance=simd_sum(variance);if(lane==0)partial[simd]=variance;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    variance=simd_sum(lane<4?partial[lane]:0.0f);
    float inverse=rsqrt(variance/float(p.columns)+1e-6f);
    for(ulong j=tid;j<p.columns;j+=128)out[row*p.oStride+j]=(a[row*p.aStride+j]-mean)*inverse*b[j]+c[j];
}
kernel void vision_scores(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],threadgroup float* scratch [[threadgroup(0)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong row=id/p.heads,head=id%p.heads,dim=p.hidden/p.heads,batchBase=(row/p.tokens)*p.tokens;
    if(row>=p.outRows)return;threadgroup float partial[4];
    for(ulong key=simd;key<p.tokens;key+=4) {
        float score=0;for(ulong d=lane;d<dim;d+=32)score+=a[row*p.aStride+head*dim+d]*a[(batchBase+key)*p.aStride+p.hidden+head*dim+d];
        score=simd_sum(score)*rsqrt(float(dim));if(lane==0)scratch[key]=score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong key=tid;key<p.tokens;key+=128)maximum=max(maximum,scratch[key]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;for(ulong key=tid;key<p.tokens;key+=128){float v=exp(scratch[key]-maximum);scratch[key]=v;total+=v;}
    total=simd_sum(total);if(lane==0)partial[simd]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong key=tid;key<p.tokens;key+=128)out[row*p.oStride+head*p.tokens+key]=scratch[key]/total;
}
kernel void vision_pack_qk(device const float* input [[buffer(0)]],device float* query [[buffer(1)]],
    device float* key [[buffer(2)]],constant VisionParams& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    ulong dim=p.hidden/p.heads,packedRow=ulong(id)/dim,d=ulong(id)%dim;
    if(packedRow>=p.rows*p.heads)return;
    ulong row=(packedRow/(p.heads*p.tokens))*p.tokens+packedRow%p.tokens,head=(packedRow/p.tokens)%p.heads;
    query[packedRow*p.bStride+d]=input[row*p.aStride+head*dim+d];
    key[packedRow*p.cStride+d]=input[row*p.aStride+p.hidden+head*dim+d];
}
kernel void vision_softmax(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],constant VisionParams& p [[buffer(4)]],threadgroup float* scratch [[threadgroup(0)]],
    uint id [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong row=id/p.heads,head=id%p.heads,dim=p.hidden/p.heads,batchBase=(row/p.tokens)*p.tokens;
    if(row>=p.outRows)return;threadgroup float partial[4];
    ulong packedRow=((row/p.tokens)*p.heads+head)*p.tokens+row%p.tokens;
    for(ulong key=tid;key<p.tokens;key+=128)scratch[key]=b[packedRow*p.bStride+key]*rsqrt(float(dim));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum=-INFINITY;for(ulong key=tid;key<p.tokens;key+=128)maximum=max(maximum,scratch[key]);
    maximum=simd_max(maximum);if(lane==0)partial[simd]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum=simd_max(lane<4?partial[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;for(ulong key=tid;key<p.tokens;key+=128){float v=exp(scratch[key]-maximum);scratch[key]=v;total+=v;}
    total=simd_sum(total);if(lane==0)partial[simd]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total=simd_sum(lane<4?partial[lane]:0.0f);
    for(ulong key=tid;key<p.tokens;key+=128)out[row*p.oStride+head*p.tokens+key]=scratch[key]/total;
}
// Input/output matrices contain flattened batch/token rows. The merger copies
// across padded MPS row boundaries explicitly rather than aliasing row strides.
kernel void vision_op(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],device float* out [[buffer(3)]],
    constant VisionParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.outRows*p.outColumns)return;
    ulong row=id/p.outColumns,col=id%p.outColumns;
    if(p.mode==0){out[row*p.oStride+col]=a[row*p.aStride+col]+b[col];return;}
    if(p.mode==1){
        // Compute row statistics once rather than repeating both sums per column.
        if(col)return;
        float mean=0;for(ulong j=0;j<p.columns;++j)mean+=a[row*p.aStride+j];mean/=float(p.columns);
        float variance=0;for(ulong j=0;j<p.columns;++j){float x=a[row*p.aStride+j]-mean;variance+=x*x;}
        float inverse=rsqrt(variance/float(p.columns)+1e-6f);
        for(ulong j=0;j<p.columns;++j)out[row*p.oStride+j]=(a[row*p.aStride+j]-mean)*inverse*b[j]+c[j];return;
    }
    if(p.mode==2||p.mode==3){
        float x=a[row*p.aStride+col],activation;
        // Avoid native tanh overflow; at |argument| >= 10 its FP32 value saturates.
        if(p.mode==2)activation=tanh(clamp(0.7978845608028654f*(x+0.044715f*x*x*x),-10.0f,10.0f));
        else {
            // Abramowitz-Stegun erf approximation (absolute error < 1.5e-7).
            float z=abs(x)*0.7071067811865475f,t=1.0f/(1.0f+0.3275911f*z);
            float v=1.0f-(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t*exp(-z*z);
            activation=x<0?-v:v;
        }
        out[row*p.oStride+col]=0.5f*x*(1.0f+activation);return;
    }
    if(p.mode==4){out[row*p.oStride+col]=a[(row*p.unit+col/p.columns)*p.aStride+col%p.columns];return;}
    if(p.mode==9){out[row*p.oStride+col]=a[(row/p.tokens)*p.aStride+(row%p.tokens)*p.outColumns+col];return;}
    if(p.mode==5){
        ulong token=row%p.tokens;float position=0;
        for(ulong i=0;i<4;++i)position+=b[ulong(c[token*p.cStride+i])*p.bStride+col]*c[token*p.cStride+4+i];
        out[row*p.oStride+col]=a[row*p.aStride+col]+position;return;
    }
    ulong dim=p.hidden/p.heads;
    if(p.mode==6){
        if(col>=2*p.hidden){out[row*p.oStride+col]=a[row*p.aStride+col];return;}
        ulong d=col%dim,base=col-d,other=(d+dim/2)%dim;
        float angle=b[(row%p.tokens)*p.bStride+d];
        out[row*p.oStride+col]=a[row*p.aStride+col]*cos(angle)+(d<dim/2?-1.0f:1.0f)*a[row*p.aStride+base+other]*sin(angle);return;
    }
    if(p.mode==7){
        // One thread handles a whole softmax row; dispatch grid includes
        // columns, but only its first column performs the reduction.
        if(col%p.tokens)return;
        ulong head=col/p.tokens,batchBase=(row/p.tokens)*p.tokens;float maximum=-INFINITY;
        for(ulong key=0;key<p.tokens;++key){
            float score=0;for(ulong d=0;d<dim;++d)score+=a[row*p.aStride+head*dim+d]*a[(batchBase+key)*p.aStride+p.hidden+head*dim+d];
            score*=rsqrt(float(dim));out[row*p.oStride+head*p.tokens+key]=score;maximum=max(maximum,score);
        }
        float total=0;for(ulong key=0;key<p.tokens;++key){float v=exp(out[row*p.oStride+head*p.tokens+key]-maximum);out[row*p.oStride+head*p.tokens+key]=v;total+=v;}
        for(ulong key=0;key<p.tokens;++key)out[row*p.oStride+head*p.tokens+key]/=total;return;
    }
    if(p.mode==8){
        ulong head=col/dim,d=col%dim,batchBase=(row/p.tokens)*p.tokens;float result=0;
        for(ulong key=0;key<p.tokens;++key)result+=a[row*p.aStride+head*p.tokens+key]*b[(batchBase+key)*p.bStride+2*p.hidden+head*dim+d];
        out[row*p.oStride+col]=result;
    }
}

constant uint fixed_rows [[function_constant(0)]];
constant uint fixed_inner [[function_constant(1)]];
constant uint fixed_columns [[function_constant(2)]];
constant uint fixed_aStride [[function_constant(3)]];
constant uint fixed_oStride [[function_constant(4)]];
constant bool fixed_no_tail [[function_constant(5)]];
kernel void specialized_bf16_decode(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(!fixed_no_tail&&column>=ulong(fixed_columns))return;
    float4 sum0=0.0f,sum1=0.0f;
    if(ulong(fixed_inner)%4==0) {
        for(ulong k=ulong(lane)*4;k<ulong(fixed_inner);k+=128) {
            float4 x=*(device const float4*)(a+k),y=ulong(fixed_rows)==2?*(device const float4*)(a+ulong(fixed_aStride)+k):float4(0.0f);
            ushort4 b0=*(device const ushort4*)(weight+column*ulong(fixed_inner)+k),b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
            if(fixed_no_tail||column+1<ulong(fixed_columns))b1=*(device const ushort4*)(weight+(column+1)*ulong(fixed_inner)+k);
            if(fixed_no_tail||column+2<ulong(fixed_columns))b2=*(device const ushort4*)(weight+(column+2)*ulong(fixed_inner)+k);
            if(fixed_no_tail||column+3<ulong(fixed_columns))b3=*(device const ushort4*)(weight+(column+3)*ulong(fixed_inner)+k);
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
                sum0+=x[j]*w;if(ulong(fixed_rows)==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<ulong(fixed_inner);k+=32) {
        float x=a[k],y=ulong(fixed_rows)==2?a[ulong(fixed_aStride)+k]:0.0f;
        float4 w=0.0f;
        w.x=as_type<float>(uint(weight[column*ulong(fixed_inner)+k])<<16);
        if(fixed_no_tail||column+1<ulong(fixed_columns))w.y=as_type<float>(uint(weight[(column+1)*ulong(fixed_inner)+k])<<16);
        if(fixed_no_tail||column+2<ulong(fixed_columns))w.z=as_type<float>(uint(weight[(column+2)*ulong(fixed_inner)+k])<<16);
        if(fixed_no_tail||column+3<ulong(fixed_columns))w.w=as_type<float>(uint(weight[(column+3)*ulong(fixed_inner)+k])<<16);
        sum0+=x*w;if(ulong(fixed_rows)==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&(fixed_no_tail||column+j<ulong(fixed_columns));++j){out[column+j]=sum0[j];if(ulong(fixed_rows)==2)out[ulong(fixed_oStride)+column+j]=sum1[j];}
}
kernel void specialized_bf16_decode_small(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*4+simd*2;
    if(!fixed_no_tail&&column>=ulong(fixed_columns))return;
    float2 sum0=0.0f,sum1=0.0f;
    if(ulong(fixed_inner)%4==0) {
        for(ulong k=ulong(lane)*4;k<ulong(fixed_inner);k+=128) {
            float4 x=*(device const float4*)(a+k),y=ulong(fixed_rows)==2?*(device const float4*)(a+ulong(fixed_aStride)+k):float4(0.0f);
            ushort4 b0=*(device const ushort4*)(weight+column*ulong(fixed_inner)+k),b1=ushort4(0);
            if(fixed_no_tail||column+1<ulong(fixed_columns))b1=*(device const ushort4*)(weight+(column+1)*ulong(fixed_inner)+k);
            for(uint j=0;j<4;++j) {
                float2 w=as_type<float2>(uint2(b0[j],b1[j])<<16);
                sum0+=x[j]*w;if(ulong(fixed_rows)==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<ulong(fixed_inner);k+=32) {
        float x=a[k],y=ulong(fixed_rows)==2?a[ulong(fixed_aStride)+k]:0.0f;
        float2 w=0.0f;
        w.x=as_type<float>(uint(weight[column*ulong(fixed_inner)+k])<<16);
        if(fixed_no_tail||column+1<ulong(fixed_columns))w.y=as_type<float>(uint(weight[(column+1)*ulong(fixed_inner)+k])<<16);
        sum0+=x*w;if(ulong(fixed_rows)==2)sum1+=y*w;
    }
    }
    sum0=float2(simd_sum(sum0.x),simd_sum(sum0.y));
    sum1=float2(simd_sum(sum1.x),simd_sum(sum1.y));
    if(lane==0)for(uint j=0;j<2&&(fixed_no_tail||column+j<ulong(fixed_columns));++j){out[column+j]=sum0[j];if(ulong(fixed_rows)==2)out[ulong(fixed_oStride)+column+j]=sum1[j];}
}
// BF16 weights are expanded only inside a small shared-memory tile. FP32
// SIMD-group matrix multiply accumulates two 8x8 output tiles per group.

inline ushort primary_word(device const uchar* payload,device const uint* metadata,uint row,uint k,uint inner) {
    uint block=(row/4)*(inner/128)+k/128,meta=metadata[block],at=(meta>>9)*256,index=(k%128)*4+row%4;
    if(meta&256u){uint sm=payload[at+index],tag=(payload[at+512+index/2]>>((index%2)*4))&15u;
        return ushort((sm&127u)|((sm&128u)<<8)|(((uint(meta)&255u)+tag)<<7));}
    return ushort(payload[at+index*2])|(ushort(payload[at+index*2+1])<<8);
}
inline ushort4 primary_row4(device const uchar* payload,device const uint* metadata,uint row,uint k,uint inner) {
    return ushort4(primary_word(payload,metadata,row,k,inner),primary_word(payload,metadata,row,k+1,inner),primary_word(payload,metadata,row,k+2,inner),primary_word(payload,metadata,row,k+3,inner));
}

// Packed admission proves row%4==0, k%4==0, inner%128==0 and four valid rows.
// At every decode iteration k..k+3 stays in a single 128-K block.
struct PrimaryQuad { ushort4 k0,k1,k2,k3; };
inline ushort4 primary_quad_kword(uint sm,ushort tags,uint base) {
    uint4 signMant=uint4(as_type<uchar4>(sm));
    uint4 exponents=uint4(tags,tags>>4,tags>>8,tags>>12)&15u;
    return ushort4((signMant&127u)|((signMant&128u)<<8)|((exponents+base)<<7));
}
inline PrimaryQuad primary_quad(device const uchar* payload,device const uint* metadata,uint row,uint k,uint inner) {
    uint meta=metadata[(row/4)*(inner/128)+k/128],at=(meta>>9)*256,index=(k%128)*4;
    PrimaryQuad q;
    if(meta&256u) {
        uint4 sm=*(device const uint4*)(payload+at+index);
        ushort4 tags=as_type<ushort4>(*(device const uint2*)(payload+at+512+index/2));
        uint base=uint(meta)&255u;
        q.k0=primary_quad_kword(sm.x,tags.x,base);q.k1=primary_quad_kword(sm.y,tags.y,base);
        q.k2=primary_quad_kword(sm.z,tags.z,base);q.k3=primary_quad_kword(sm.w,tags.w,base);
    } else {
        uint4 lo=*(device const uint4*)(payload+at+index*2),hi=*(device const uint4*)(payload+at+index*2+16);
        q.k0=as_type<ushort4>(lo.xy);q.k1=as_type<ushort4>(lo.zw);
        q.k2=as_type<ushort4>(hi.xy);q.k3=as_type<ushort4>(hi.zw);
    }
    return q;
}
// Correctness-only bounded readback exercises the exact tile helper used by decode.
kernel void primary_expand_vector_words(device const uchar* payload [[buffer(0)]],device const uint* metadata [[buffer(1)]],
 device uint* output [[buffer(2)]],constant ulong& inner [[buffer(3)]],constant ulong& start [[buffer(4)]],constant ulong& count [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    if(ulong(id)>=count)return;
    ulong i=start+id,row=i/inner,k=i%inner;
    PrimaryQuad q=primary_quad(payload,metadata,row-row%4,k-k%4,inner);
    ushort4 values=k%4==0?q.k0:k%4==1?q.k1:k%4==2?q.k2:q.k3;
    output[id]=uint(values[row%4])<<16;
}

kernel void primary_expand_words(device const uchar* payload [[buffer(0)]],device const uint* metadata [[buffer(1)]],
 device uint* output [[buffer(2)]],constant ulong& inner [[buffer(3)]],constant ulong& start [[buffer(4)]],constant ulong& count [[buffer(5)]],uint id [[thread_position_in_grid]]) {
 if(ulong(id)>=count)return;ulong i=start+id;output[id]=uint(primary_word(payload,metadata,i/inner,i%inner,inner))<<16;
}
kernel void primary_bf16_decode(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const uint* metadata [[buffer(4)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            PrimaryQuad quad=primary_quad(payload,metadata,column,k,p.inner);
            ushort4 weights[4]={quad.k0,quad.k1,quad.k2,quad.k3};
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(weights[j])<<16);
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
kernel void primary_bf16_gemm(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const uint* metadata [[buffer(4)]],
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
kernel void primary_gemm_16_64_64(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const uint* metadata [[buffer(4)]],
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
kernel void primary_gemm_32_64_32(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const uint* metadata [[buffer(4)]],
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
kernel void primary_specialized_bf16_decode(device const float* a [[buffer(0)]],device const uchar* payload [[buffer(1)]],device const uint* metadata [[buffer(4)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(!fixed_no_tail&&column>=ulong(fixed_columns))return;
    float4 sum0=0.0f,sum1=0.0f;
    if(ulong(fixed_inner)%4==0) {
        for(ulong k=ulong(lane)*4;k<ulong(fixed_inner);k+=128) {
            float4 x=*(device const float4*)(a+k),y=ulong(fixed_rows)==2?*(device const float4*)(a+ulong(fixed_aStride)+k):float4(0.0f);
            PrimaryQuad quad=primary_quad(payload,metadata,column,k,ulong(fixed_inner));
            ushort4 weights[4]={quad.k0,quad.k1,quad.k2,quad.k3};
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(weights[j])<<16);
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

// Exact compile-time geometries. Aligned float4 bit stores write the
// same contiguous FP32 right tile; all original barriers/math stay outside.
template<uint tileColumns,uint tileK,uint threads>
inline void primary_prefill_tile_fill(device const uchar* payload,device const uint* metadata,
 threadgroup float* right,uint colBase,uint kBase,uint columns,uint inner,uint tid) {
 for(uint index=tid;index<(tileColumns/4)*(tileK/4);index+=threads) {
  uint localK=(index/(tileColumns/4))*4,localColumn=(index%(tileColumns/4))*4;
  uint row=colBase+localColumn,k=kBase+localK;
  PrimaryQuad quad;
  if(row<columns&&k<inner)quad=primary_quad(payload,metadata,row,k,inner);
  else {quad.k0=ushort4(0);quad.k1=ushort4(0);quad.k2=ushort4(0);quad.k3=ushort4(0);}
  ushort4 words[4]={quad.k0,quad.k1,quad.k2,quad.k3};
  for(uint j=0;j<4;++j)
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
