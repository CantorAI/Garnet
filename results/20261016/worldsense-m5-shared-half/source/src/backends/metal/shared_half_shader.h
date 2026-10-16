// Derived from the original strict FP32 BF16 dot/reduction kernels.
#pragma once
namespace Garnet {
inline constexpr const char* kSharedHalfShader=R"SHAREDHALF(
struct SharedHalfParams { ulong rows,inner,columns,aStride,oStride,wStride; };
constant uint fixed_wStride [[function_constant(6)]];
inline float shared_half_value(ushort h,int scale) {
    uint sign=uint(h&0x8000)<<16, fraction=uint(h&1023), exponent=uint(h>>10)&31;
    // Normal half conversion is exact. Adjust the FP32 exponent as integer
    // bits; only subnormals need normalization below (never a half float load).
    if(exponent) return as_type<float>(as_type<uint>(float(as_type<half>(h))) - (uint(scale)<<23));
    if(!fraction)return as_type<float>(sign);
    uint top=31-clz(fraction);
    return as_type<float>(sign | (uint(int(top)+103-scale)<<23) | ((fraction-(1u<<top))<<(23-top)));
}
kernel void shared_half_decode(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant SharedHalfParams& p [[buffer(3)]], device const int* scales [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(column>=p.columns)return;
    float4 sum0=0.0f,sum1=0.0f;
    if(p.inner%4==0) {
        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
            float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
            ushort4 b0=*(device const ushort4*)(weight+column*p.wStride+k),b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
            if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.wStride+k);
            if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.wStride+k);
            if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.wStride+k);
            for(uint j=0;j<4;++j) {
                float4 w=float4(shared_half_value(b0[j],scales[column]), column+1<p.columns?shared_half_value(b1[j],scales[column+1]):0.0f, column+2<p.columns?shared_half_value(b2[j],scales[column+2]):0.0f, column+3<p.columns?shared_half_value(b3[j],scales[column+3]):0.0f);
                sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<p.inner;k+=32) {
        float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
        float4 w=0.0f;
        w.x=shared_half_value(weight[column*p.wStride+k],scales[column]);
        if(column+1<p.columns)w.y=shared_half_value(weight[(column+1)*p.wStride+k],scales[column+1]);
        if(column+2<p.columns)w.z=shared_half_value(weight[(column+2)*p.wStride+k],scales[column+2]);
        if(column+3<p.columns)w.w=shared_half_value(weight[(column+3)*p.wStride+k],scales[column+3]);
        sum0+=x*w;if(p.rows==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
kernel void specialized_shared_half_decode(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant SharedHalfParams& p [[buffer(3)]], device const int* scales [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*16+simd*4;
    if(!fixed_no_tail&&column>=ulong(fixed_columns))return;
    float4 sum0=0.0f,sum1=0.0f;
    if(ulong(fixed_inner)%4==0) {
        for(ulong k=ulong(lane)*4;k<ulong(fixed_inner);k+=128) {
            float4 x=*(device const float4*)(a+k),y=ulong(fixed_rows)==2?*(device const float4*)(a+ulong(fixed_aStride)+k):float4(0.0f);
            ushort4 b0=*(device const ushort4*)(weight+column*ulong(fixed_wStride)+k),b1=ushort4(0),b2=ushort4(0),b3=ushort4(0);
            if(fixed_no_tail||column+1<ulong(fixed_columns))b1=*(device const ushort4*)(weight+(column+1)*ulong(fixed_wStride)+k);
            if(fixed_no_tail||column+2<ulong(fixed_columns))b2=*(device const ushort4*)(weight+(column+2)*ulong(fixed_wStride)+k);
            if(fixed_no_tail||column+3<ulong(fixed_columns))b3=*(device const ushort4*)(weight+(column+3)*ulong(fixed_wStride)+k);
            for(uint j=0;j<4;++j) {
                float4 w=float4(shared_half_value(b0[j],scales[column]), column+1<p.columns?shared_half_value(b1[j],scales[column+1]):0.0f, column+2<p.columns?shared_half_value(b2[j],scales[column+2]):0.0f, column+3<p.columns?shared_half_value(b3[j],scales[column+3]):0.0f);
                sum0+=x[j]*w;if(ulong(fixed_rows)==2)sum1+=y[j]*w;
            }
        }
    } else {
    for(ulong k=lane;k<ulong(fixed_inner);k+=32) {
        float x=a[k],y=ulong(fixed_rows)==2?a[ulong(fixed_aStride)+k]:0.0f;
        float4 w=0.0f;
        w.x=shared_half_value(weight[column*ulong(fixed_wStride)+k],scales[column]);
        if(fixed_no_tail||column+1<ulong(fixed_columns))w.y=shared_half_value(weight[(column+1)*ulong(fixed_wStride)+k],scales[column+1]);
        if(fixed_no_tail||column+2<ulong(fixed_columns))w.z=shared_half_value(weight[(column+2)*ulong(fixed_wStride)+k],scales[column+2]);
        if(fixed_no_tail||column+3<ulong(fixed_columns))w.w=shared_half_value(weight[(column+3)*ulong(fixed_wStride)+k],scales[column+3]);
        sum0+=x*w;if(ulong(fixed_rows)==2)sum1+=y*w;
    }
    }
    sum0=float4(simd_sum(sum0.x),simd_sum(sum0.y),simd_sum(sum0.z),simd_sum(sum0.w));
    sum1=float4(simd_sum(sum1.x),simd_sum(sum1.y),simd_sum(sum1.z),simd_sum(sum1.w));
    if(lane==0)for(uint j=0;j<4&&(fixed_no_tail||column+j<ulong(fixed_columns));++j){out[column+j]=sum0[j];if(ulong(fixed_rows)==2)out[ulong(fixed_oStride)+column+j]=sum1[j];}
}
kernel void shared_half_inverse_test(device float* out [[buffer(0)]],constant int& scale [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<65536)out[i]=((i>>10)&31)==31?0.0f:shared_half_value(ushort(i),scale);
}
)SHAREDHALF";
}
