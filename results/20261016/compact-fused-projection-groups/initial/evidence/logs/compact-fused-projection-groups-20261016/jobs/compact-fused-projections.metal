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

kernel void bf16_body_coalesced_block(device const float* a [[buffer(0)]],device const ushort* fallback [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],device const uchar* payload [[buffer(4)]],
    device const uint* metadata [[buffer(5)]],uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint threads [[threads_per_threadgroup]]) {
    ulong column=ulong(group)*(threads/32)*4+simd*4;if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
        ulong block=(column/4)*(p.inner/128)+k/128;uint meta=metadata[block];float4 weights[4];
        if(meta&256u) {
            ulong at=block*768;
            uint4 sm=*(device const uint4*)(payload+at+ulong(lane)*16);
            ushort4 tags=as_type<ushort4>(*(device const uint2*)(payload+at+512+ulong(lane)*8));
            for(uint j=0;j<4;++j) {
                uint4 signMant=uint4(as_type<uchar4>(sm[j]));
                uint4 exponents=uint4(tags[j],tags[j]>>4,tags[j]>>8,tags[j]>>12)&15u;
                uint4 bits=(signMant&127u)|((signMant&128u)<<8)|((exponents+(meta&255u))<<7);
                weights[j]=as_type<float4>(bits<<16);
            }
        } else {
            ulong at=(column/4)*p.inner*4+k*4;
            uint4 p0=*(device const uint4*)(fallback+at),p1=*(device const uint4*)(fallback+at+8);
            ushort4 words[4]={as_type<ushort4>(p0.xy),as_type<ushort4>(p0.zw),as_type<ushort4>(p1.xy),as_type<ushort4>(p1.zw)};
            for(uint j=0;j<4;++j)weights[j]=as_type<float4>(uint4(words[j])<<16);
        }
        for(uint j=0;j<4;++j){sum0+=x[j]*weights[j];if(p.rows==2)sum1+=y[j]*weights[j];}
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
inline void grouped_raw64(device const float* a,device const ushort* weight,
    device float* out,constant BF16Params& p,
    uint group,uint simd,uint lane) {
    ulong column=ulong(group)*8+simd*4;
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
inline void grouped_small64(device const float* a,device const ushort* weight,
    device float* out,constant BF16Params& p,
    uint group,uint simd,uint lane) {
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

inline void grouped_compact64(device const float* a,device const ushort* fallback,
    device float* out,constant BF16Params& p,device const uchar* payload,
    device const uint* metadata,uint group,
    uint simd,uint lane,uint threads) {
    ulong column=ulong(group)*(threads/32)*4+simd*4;if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
        ulong block=(column/4)*(p.inner/128)+k/128;uint meta=metadata[block];float4 weights[4];
        if(meta&256u) {
            ulong at=block*768;
            uint4 sm=*(device const uint4*)(payload+at+ulong(lane)*16);
            ushort4 tags=as_type<ushort4>(*(device const uint2*)(payload+at+512+ulong(lane)*8));
            for(uint j=0;j<4;++j) {
                uint4 signMant=uint4(as_type<uchar4>(sm[j]));
                uint4 exponents=uint4(tags[j],tags[j]>>4,tags[j]>>8,tags[j]>>12)&15u;
                uint4 bits=(signMant&127u)|((signMant&128u)<<8)|((exponents+(meta&255u))<<7);
                weights[j]=as_type<float4>(bits<<16);
            }
        } else {
            ulong at=(column/4)*p.inner*4+k*4;
            uint4 p0=*(device const uint4*)(fallback+at),p1=*(device const uint4*)(fallback+at+8);
            ushort4 words[4]={as_type<ushort4>(p0.xy),as_type<ushort4>(p0.zw),as_type<ushort4>(p1.xy),as_type<ushort4>(p1.zw)};
            for(uint j=0;j<4;++j)weights[j]=as_type<float4>(uint4(words[j])<<16);
        }
        for(uint j=0;j<4;++j){sum0+=x[j]*weights[j];if(p.rows==2)sum1+=y[j]*weights[j];}
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}

kernel void compact_fused_projection_group(device const float* a [[buffer(0)]],
    device const ushort* w0 [[buffer(1)]],device float* o0 [[buffer(2)]],device const uchar* b0 [[buffer(3)]],device const uint* m0 [[buffer(4)]],
    device const ushort* w1 [[buffer(5)]],device float* o1 [[buffer(6)]],device const uchar* b1 [[buffer(7)]],device const uint* m1 [[buffer(8)]],
    device const ushort* w2 [[buffer(9)]],device float* o2 [[buffer(10)]],device const uchar* b2 [[buffer(11)]],device const uint* m2 [[buffer(12)]],
    constant BF16Params* params [[buffer(13)]],constant uint* cuts [[buffer(14)]],constant uint* kinds [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    uint part=group<cuts[1]?0:(group<cuts[2]?1:2),local=group-cuts[part];
    device const ushort* w=part==0?w0:(part==1?w1:w2);
    device float* o=part==0?o0:(part==1?o1:o2);
    device const uchar* b=part==0?b0:(part==1?b1:b2);
    device const uint* m=part==0?m0:(part==1?m1:m2);
    if(kinds[part]==2)grouped_compact64(a,w,o,params[part],b,m,local,simd,lane,64);
    else if(kinds[part]==1)grouped_raw64(a,w,o,params[part],local,simd,lane);
    else grouped_small64(a,w,o,params[part],local,simd,lane);
}
