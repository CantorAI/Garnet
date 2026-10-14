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
kernel void bf16_decode_s16_t64(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
 device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
 uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 uint local=lane%16;ulong column=ulong(group)*16+simd*8+(lane/16)*4;
 if(column>=p.columns)return;
 float4 sum00=0.0f,sum01=0.0f,sum10=0.0f,sum11=0.0f;
 if(p.inner%4==0) {
  for(ulong k=ulong(local)*4;k<p.inner;k+=128) {
   if(k<p.inner) {float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
    ushort4 b0=ushort4(0);if(column+0<p.columns)b0=*(device const ushort4*)(weight+(column+0)*p.inner+k);
    ushort4 b1=ushort4(0);if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
    ushort4 b2=ushort4(0);if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
    ushort4 b3=ushort4(0);if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
    for(uint j=0;j<4;++j) {float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);sum00+=x[j]*w;if(p.rows==2)sum10+=y[j]*w;}
   }
   if(k+64<p.inner) {float4 x=*(device const float4*)(a+k+64),y=p.rows==2?*(device const float4*)(a+p.aStride+k+64):float4(0.0f);
    ushort4 b0=ushort4(0);if(column+0<p.columns)b0=*(device const ushort4*)(weight+(column+0)*p.inner+k+64);
    ushort4 b1=ushort4(0);if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k+64);
    ushort4 b2=ushort4(0);if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k+64);
    ushort4 b3=ushort4(0);if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k+64);
    for(uint j=0;j<4;++j) {float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);sum01+=x[j]*w;if(p.rows==2)sum11+=y[j]*w;}
   }
  }
 } else {
  for(ulong k=local;k<p.inner;k+=32) {
   if(k<p.inner) {float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;float4 w=0.0f;
    if(column+0<p.columns)w.x=as_type<float>(uint(weight[(column+0)*p.inner+k])<<16);
    if(column+1<p.columns)w.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
    if(column+2<p.columns)w.z=as_type<float>(uint(weight[(column+2)*p.inner+k])<<16);
    if(column+3<p.columns)w.w=as_type<float>(uint(weight[(column+3)*p.inner+k])<<16);
    sum00+=x*w;if(p.rows==2)sum10+=y*w;
   }
   if(k+16<p.inner) {float x=a[k+16],y=p.rows==2?a[p.aStride+k+16]:0.0f;float4 w=0.0f;
    if(column+0<p.columns)w.x=as_type<float>(uint(weight[(column+0)*p.inner+k+16])<<16);
    if(column+1<p.columns)w.y=as_type<float>(uint(weight[(column+1)*p.inner+k+16])<<16);
    if(column+2<p.columns)w.z=as_type<float>(uint(weight[(column+2)*p.inner+k+16])<<16);
    if(column+3<p.columns)w.w=as_type<float>(uint(weight[(column+3)*p.inner+k+16])<<16);
    sum01+=x*w;if(p.rows==2)sum11+=y*w;
   }
  }
 }
 float4 sum0=sum00+sum01,sum1=sum10+sum11;
 sum0+=simd_shuffle_xor(sum0,8);sum1+=simd_shuffle_xor(sum1,8);
 sum0+=simd_shuffle_xor(sum0,4);sum1+=simd_shuffle_xor(sum1,4);
 sum0+=simd_shuffle_xor(sum0,2);sum1+=simd_shuffle_xor(sum1,2);
 sum0+=simd_shuffle_xor(sum0,1);sum1+=simd_shuffle_xor(sum1,1);
 if(local==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
kernel void bf16_decode_s16_t128(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
 device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
 uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 uint local=lane%16;ulong column=ulong(group)*32+simd*8+(lane/16)*4;
 if(column>=p.columns)return;
 float4 sum00=0.0f,sum01=0.0f,sum10=0.0f,sum11=0.0f;
 if(p.inner%4==0) {
  for(ulong k=ulong(local)*4;k<p.inner;k+=128) {
   if(k<p.inner) {float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
    ushort4 b0=ushort4(0);if(column+0<p.columns)b0=*(device const ushort4*)(weight+(column+0)*p.inner+k);
    ushort4 b1=ushort4(0);if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
    ushort4 b2=ushort4(0);if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
    ushort4 b3=ushort4(0);if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
    for(uint j=0;j<4;++j) {float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);sum00+=x[j]*w;if(p.rows==2)sum10+=y[j]*w;}
   }
   if(k+64<p.inner) {float4 x=*(device const float4*)(a+k+64),y=p.rows==2?*(device const float4*)(a+p.aStride+k+64):float4(0.0f);
    ushort4 b0=ushort4(0);if(column+0<p.columns)b0=*(device const ushort4*)(weight+(column+0)*p.inner+k+64);
    ushort4 b1=ushort4(0);if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k+64);
    ushort4 b2=ushort4(0);if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k+64);
    ushort4 b3=ushort4(0);if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k+64);
    for(uint j=0;j<4;++j) {float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);sum01+=x[j]*w;if(p.rows==2)sum11+=y[j]*w;}
   }
  }
 } else {
  for(ulong k=local;k<p.inner;k+=32) {
   if(k<p.inner) {float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;float4 w=0.0f;
    if(column+0<p.columns)w.x=as_type<float>(uint(weight[(column+0)*p.inner+k])<<16);
    if(column+1<p.columns)w.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
    if(column+2<p.columns)w.z=as_type<float>(uint(weight[(column+2)*p.inner+k])<<16);
    if(column+3<p.columns)w.w=as_type<float>(uint(weight[(column+3)*p.inner+k])<<16);
    sum00+=x*w;if(p.rows==2)sum10+=y*w;
   }
   if(k+16<p.inner) {float x=a[k+16],y=p.rows==2?a[p.aStride+k+16]:0.0f;float4 w=0.0f;
    if(column+0<p.columns)w.x=as_type<float>(uint(weight[(column+0)*p.inner+k+16])<<16);
    if(column+1<p.columns)w.y=as_type<float>(uint(weight[(column+1)*p.inner+k+16])<<16);
    if(column+2<p.columns)w.z=as_type<float>(uint(weight[(column+2)*p.inner+k+16])<<16);
    if(column+3<p.columns)w.w=as_type<float>(uint(weight[(column+3)*p.inner+k+16])<<16);
    sum01+=x*w;if(p.rows==2)sum11+=y*w;
   }
  }
 }
 float4 sum0=sum00+sum01,sum1=sum10+sum11;
 sum0+=simd_shuffle_xor(sum0,8);sum1+=simd_shuffle_xor(sum1,8);
 sum0+=simd_shuffle_xor(sum0,4);sum1+=simd_shuffle_xor(sum1,4);
 sum0+=simd_shuffle_xor(sum0,2);sum1+=simd_shuffle_xor(sum1,2);
 sum0+=simd_shuffle_xor(sum0,1);sum1+=simd_shuffle_xor(sum1,1);
 if(local==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}
}
