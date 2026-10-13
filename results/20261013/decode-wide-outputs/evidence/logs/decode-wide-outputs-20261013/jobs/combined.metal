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
kernel void bf16_decode_c8_t64(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
 device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
 uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 ulong column=ulong(group)*16+simd*8;
 if(column>=p.columns)return;
 float4 sum0_0=0.0f,sum1_0=0.0f;
 float4 sum0_1=0.0f,sum1_1=0.0f;
 if(p.inner%4==0) {
  for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
   float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
   ushort4 b0=ushort4(0);if(column+0<p.columns)b0=*(device const ushort4*)(weight+(column+0)*p.inner+k);
   ushort4 b1=ushort4(0);if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
   ushort4 b2=ushort4(0);if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
   ushort4 b3=ushort4(0);if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
   ushort4 b4=ushort4(0);if(column+4<p.columns)b4=*(device const ushort4*)(weight+(column+4)*p.inner+k);
   ushort4 b5=ushort4(0);if(column+5<p.columns)b5=*(device const ushort4*)(weight+(column+5)*p.inner+k);
   ushort4 b6=ushort4(0);if(column+6<p.columns)b6=*(device const ushort4*)(weight+(column+6)*p.inner+k);
   ushort4 b7=ushort4(0);if(column+7<p.columns)b7=*(device const ushort4*)(weight+(column+7)*p.inner+k);
   for(uint j=0;j<4;++j) {
    float4 w0=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
    sum0_0+=x[j]*w0;if(p.rows==2)sum1_0+=y[j]*w0;
    float4 w1=as_type<float4>(uint4(b4[j],b5[j],b6[j],b7[j])<<16);
    sum0_1+=x[j]*w1;if(p.rows==2)sum1_1+=y[j]*w1;
   }
  }
 } else {
  for(ulong k=lane;k<p.inner;k+=32) {
   float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
   float4 w0=0.0f;
   if(column+0<p.columns)w0.x=as_type<float>(uint(weight[(column+0)*p.inner+k])<<16);
   if(column+1<p.columns)w0.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
   if(column+2<p.columns)w0.z=as_type<float>(uint(weight[(column+2)*p.inner+k])<<16);
   if(column+3<p.columns)w0.w=as_type<float>(uint(weight[(column+3)*p.inner+k])<<16);
   sum0_0+=x*w0;if(p.rows==2)sum1_0+=y*w0;
   float4 w1=0.0f;
   if(column+4<p.columns)w1.x=as_type<float>(uint(weight[(column+4)*p.inner+k])<<16);
   if(column+5<p.columns)w1.y=as_type<float>(uint(weight[(column+5)*p.inner+k])<<16);
   if(column+6<p.columns)w1.z=as_type<float>(uint(weight[(column+6)*p.inner+k])<<16);
   if(column+7<p.columns)w1.w=as_type<float>(uint(weight[(column+7)*p.inner+k])<<16);
   sum0_1+=x*w1;if(p.rows==2)sum1_1+=y*w1;
  }
 }
 sum0_0=float4(simd_sum(sum0_0.x),simd_sum(sum0_0.y),simd_sum(sum0_0.z),simd_sum(sum0_0.w));
 sum1_0=float4(simd_sum(sum1_0.x),simd_sum(sum1_0.y),simd_sum(sum1_0.z),simd_sum(sum1_0.w));
 if(lane==0)for(uint j=0;j<4&&column+0+j<p.columns;++j){out[column+0+j]=sum0_0[j];if(p.rows==2)out[p.oStride+column+0+j]=sum1_0[j];}
 sum0_1=float4(simd_sum(sum0_1.x),simd_sum(sum0_1.y),simd_sum(sum0_1.z),simd_sum(sum0_1.w));
 sum1_1=float4(simd_sum(sum1_1.x),simd_sum(sum1_1.y),simd_sum(sum1_1.z),simd_sum(sum1_1.w));
 if(lane==0)for(uint j=0;j<4&&column+4+j<p.columns;++j){out[column+4+j]=sum0_1[j];if(p.rows==2)out[p.oStride+column+4+j]=sum1_1[j];}
}
kernel void bf16_decode_c8_t128(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
 device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
 uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 ulong column=ulong(group)*32+simd*8;
 if(column>=p.columns)return;
 float4 sum0_0=0.0f,sum1_0=0.0f;
 float4 sum0_1=0.0f,sum1_1=0.0f;
 if(p.inner%4==0) {
  for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
   float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
   ushort4 b0=ushort4(0);if(column+0<p.columns)b0=*(device const ushort4*)(weight+(column+0)*p.inner+k);
   ushort4 b1=ushort4(0);if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
   ushort4 b2=ushort4(0);if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
   ushort4 b3=ushort4(0);if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
   ushort4 b4=ushort4(0);if(column+4<p.columns)b4=*(device const ushort4*)(weight+(column+4)*p.inner+k);
   ushort4 b5=ushort4(0);if(column+5<p.columns)b5=*(device const ushort4*)(weight+(column+5)*p.inner+k);
   ushort4 b6=ushort4(0);if(column+6<p.columns)b6=*(device const ushort4*)(weight+(column+6)*p.inner+k);
   ushort4 b7=ushort4(0);if(column+7<p.columns)b7=*(device const ushort4*)(weight+(column+7)*p.inner+k);
   for(uint j=0;j<4;++j) {
    float4 w0=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
    sum0_0+=x[j]*w0;if(p.rows==2)sum1_0+=y[j]*w0;
    float4 w1=as_type<float4>(uint4(b4[j],b5[j],b6[j],b7[j])<<16);
    sum0_1+=x[j]*w1;if(p.rows==2)sum1_1+=y[j]*w1;
   }
  }
 } else {
  for(ulong k=lane;k<p.inner;k+=32) {
   float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
   float4 w0=0.0f;
   if(column+0<p.columns)w0.x=as_type<float>(uint(weight[(column+0)*p.inner+k])<<16);
   if(column+1<p.columns)w0.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
   if(column+2<p.columns)w0.z=as_type<float>(uint(weight[(column+2)*p.inner+k])<<16);
   if(column+3<p.columns)w0.w=as_type<float>(uint(weight[(column+3)*p.inner+k])<<16);
   sum0_0+=x*w0;if(p.rows==2)sum1_0+=y*w0;
   float4 w1=0.0f;
   if(column+4<p.columns)w1.x=as_type<float>(uint(weight[(column+4)*p.inner+k])<<16);
   if(column+5<p.columns)w1.y=as_type<float>(uint(weight[(column+5)*p.inner+k])<<16);
   if(column+6<p.columns)w1.z=as_type<float>(uint(weight[(column+6)*p.inner+k])<<16);
   if(column+7<p.columns)w1.w=as_type<float>(uint(weight[(column+7)*p.inner+k])<<16);
   sum0_1+=x*w1;if(p.rows==2)sum1_1+=y*w1;
  }
 }
 sum0_0=float4(simd_sum(sum0_0.x),simd_sum(sum0_0.y),simd_sum(sum0_0.z),simd_sum(sum0_0.w));
 sum1_0=float4(simd_sum(sum1_0.x),simd_sum(sum1_0.y),simd_sum(sum1_0.z),simd_sum(sum1_0.w));
 if(lane==0)for(uint j=0;j<4&&column+0+j<p.columns;++j){out[column+0+j]=sum0_0[j];if(p.rows==2)out[p.oStride+column+0+j]=sum1_0[j];}
 sum0_1=float4(simd_sum(sum0_1.x),simd_sum(sum0_1.y),simd_sum(sum0_1.z),simd_sum(sum0_1.w));
 sum1_1=float4(simd_sum(sum1_1.x),simd_sum(sum1_1.y),simd_sum(sum1_1.z),simd_sum(sum1_1.w));
 if(lane==0)for(uint j=0;j<4&&column+4+j<p.columns;++j){out[column+4+j]=sum0_1[j];if(p.rows==2)out[p.oStride+column+4+j]=sum1_1[j];}
}
kernel void bf16_decode_c16_t64(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
 device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
 uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 ulong column=ulong(group)*32+simd*16;
 if(column>=p.columns)return;
 float4 sum0_0=0.0f,sum1_0=0.0f;
 float4 sum0_1=0.0f,sum1_1=0.0f;
 float4 sum0_2=0.0f,sum1_2=0.0f;
 float4 sum0_3=0.0f,sum1_3=0.0f;
 if(p.inner%4==0) {
  for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
   float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
   ushort4 b0=ushort4(0);if(column+0<p.columns)b0=*(device const ushort4*)(weight+(column+0)*p.inner+k);
   ushort4 b1=ushort4(0);if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
   ushort4 b2=ushort4(0);if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
   ushort4 b3=ushort4(0);if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
   ushort4 b4=ushort4(0);if(column+4<p.columns)b4=*(device const ushort4*)(weight+(column+4)*p.inner+k);
   ushort4 b5=ushort4(0);if(column+5<p.columns)b5=*(device const ushort4*)(weight+(column+5)*p.inner+k);
   ushort4 b6=ushort4(0);if(column+6<p.columns)b6=*(device const ushort4*)(weight+(column+6)*p.inner+k);
   ushort4 b7=ushort4(0);if(column+7<p.columns)b7=*(device const ushort4*)(weight+(column+7)*p.inner+k);
   ushort4 b8=ushort4(0);if(column+8<p.columns)b8=*(device const ushort4*)(weight+(column+8)*p.inner+k);
   ushort4 b9=ushort4(0);if(column+9<p.columns)b9=*(device const ushort4*)(weight+(column+9)*p.inner+k);
   ushort4 b10=ushort4(0);if(column+10<p.columns)b10=*(device const ushort4*)(weight+(column+10)*p.inner+k);
   ushort4 b11=ushort4(0);if(column+11<p.columns)b11=*(device const ushort4*)(weight+(column+11)*p.inner+k);
   ushort4 b12=ushort4(0);if(column+12<p.columns)b12=*(device const ushort4*)(weight+(column+12)*p.inner+k);
   ushort4 b13=ushort4(0);if(column+13<p.columns)b13=*(device const ushort4*)(weight+(column+13)*p.inner+k);
   ushort4 b14=ushort4(0);if(column+14<p.columns)b14=*(device const ushort4*)(weight+(column+14)*p.inner+k);
   ushort4 b15=ushort4(0);if(column+15<p.columns)b15=*(device const ushort4*)(weight+(column+15)*p.inner+k);
   for(uint j=0;j<4;++j) {
    float4 w0=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
    sum0_0+=x[j]*w0;if(p.rows==2)sum1_0+=y[j]*w0;
    float4 w1=as_type<float4>(uint4(b4[j],b5[j],b6[j],b7[j])<<16);
    sum0_1+=x[j]*w1;if(p.rows==2)sum1_1+=y[j]*w1;
    float4 w2=as_type<float4>(uint4(b8[j],b9[j],b10[j],b11[j])<<16);
    sum0_2+=x[j]*w2;if(p.rows==2)sum1_2+=y[j]*w2;
    float4 w3=as_type<float4>(uint4(b12[j],b13[j],b14[j],b15[j])<<16);
    sum0_3+=x[j]*w3;if(p.rows==2)sum1_3+=y[j]*w3;
   }
  }
 } else {
  for(ulong k=lane;k<p.inner;k+=32) {
   float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
   float4 w0=0.0f;
   if(column+0<p.columns)w0.x=as_type<float>(uint(weight[(column+0)*p.inner+k])<<16);
   if(column+1<p.columns)w0.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
   if(column+2<p.columns)w0.z=as_type<float>(uint(weight[(column+2)*p.inner+k])<<16);
   if(column+3<p.columns)w0.w=as_type<float>(uint(weight[(column+3)*p.inner+k])<<16);
   sum0_0+=x*w0;if(p.rows==2)sum1_0+=y*w0;
   float4 w1=0.0f;
   if(column+4<p.columns)w1.x=as_type<float>(uint(weight[(column+4)*p.inner+k])<<16);
   if(column+5<p.columns)w1.y=as_type<float>(uint(weight[(column+5)*p.inner+k])<<16);
   if(column+6<p.columns)w1.z=as_type<float>(uint(weight[(column+6)*p.inner+k])<<16);
   if(column+7<p.columns)w1.w=as_type<float>(uint(weight[(column+7)*p.inner+k])<<16);
   sum0_1+=x*w1;if(p.rows==2)sum1_1+=y*w1;
   float4 w2=0.0f;
   if(column+8<p.columns)w2.x=as_type<float>(uint(weight[(column+8)*p.inner+k])<<16);
   if(column+9<p.columns)w2.y=as_type<float>(uint(weight[(column+9)*p.inner+k])<<16);
   if(column+10<p.columns)w2.z=as_type<float>(uint(weight[(column+10)*p.inner+k])<<16);
   if(column+11<p.columns)w2.w=as_type<float>(uint(weight[(column+11)*p.inner+k])<<16);
   sum0_2+=x*w2;if(p.rows==2)sum1_2+=y*w2;
   float4 w3=0.0f;
   if(column+12<p.columns)w3.x=as_type<float>(uint(weight[(column+12)*p.inner+k])<<16);
   if(column+13<p.columns)w3.y=as_type<float>(uint(weight[(column+13)*p.inner+k])<<16);
   if(column+14<p.columns)w3.z=as_type<float>(uint(weight[(column+14)*p.inner+k])<<16);
   if(column+15<p.columns)w3.w=as_type<float>(uint(weight[(column+15)*p.inner+k])<<16);
   sum0_3+=x*w3;if(p.rows==2)sum1_3+=y*w3;
  }
 }
 sum0_0=float4(simd_sum(sum0_0.x),simd_sum(sum0_0.y),simd_sum(sum0_0.z),simd_sum(sum0_0.w));
 sum1_0=float4(simd_sum(sum1_0.x),simd_sum(sum1_0.y),simd_sum(sum1_0.z),simd_sum(sum1_0.w));
 if(lane==0)for(uint j=0;j<4&&column+0+j<p.columns;++j){out[column+0+j]=sum0_0[j];if(p.rows==2)out[p.oStride+column+0+j]=sum1_0[j];}
 sum0_1=float4(simd_sum(sum0_1.x),simd_sum(sum0_1.y),simd_sum(sum0_1.z),simd_sum(sum0_1.w));
 sum1_1=float4(simd_sum(sum1_1.x),simd_sum(sum1_1.y),simd_sum(sum1_1.z),simd_sum(sum1_1.w));
 if(lane==0)for(uint j=0;j<4&&column+4+j<p.columns;++j){out[column+4+j]=sum0_1[j];if(p.rows==2)out[p.oStride+column+4+j]=sum1_1[j];}
 sum0_2=float4(simd_sum(sum0_2.x),simd_sum(sum0_2.y),simd_sum(sum0_2.z),simd_sum(sum0_2.w));
 sum1_2=float4(simd_sum(sum1_2.x),simd_sum(sum1_2.y),simd_sum(sum1_2.z),simd_sum(sum1_2.w));
 if(lane==0)for(uint j=0;j<4&&column+8+j<p.columns;++j){out[column+8+j]=sum0_2[j];if(p.rows==2)out[p.oStride+column+8+j]=sum1_2[j];}
 sum0_3=float4(simd_sum(sum0_3.x),simd_sum(sum0_3.y),simd_sum(sum0_3.z),simd_sum(sum0_3.w));
 sum1_3=float4(simd_sum(sum1_3.x),simd_sum(sum1_3.y),simd_sum(sum1_3.z),simd_sum(sum1_3.w));
 if(lane==0)for(uint j=0;j<4&&column+12+j<p.columns;++j){out[column+12+j]=sum0_3[j];if(p.rows==2)out[p.oStride+column+12+j]=sum1_3[j];}
}
kernel void bf16_decode_c16_t128(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
 device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
 uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 ulong column=ulong(group)*64+simd*16;
 if(column>=p.columns)return;
 float4 sum0_0=0.0f,sum1_0=0.0f;
 float4 sum0_1=0.0f,sum1_1=0.0f;
 float4 sum0_2=0.0f,sum1_2=0.0f;
 float4 sum0_3=0.0f,sum1_3=0.0f;
 if(p.inner%4==0) {
  for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
   float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
   ushort4 b0=ushort4(0);if(column+0<p.columns)b0=*(device const ushort4*)(weight+(column+0)*p.inner+k);
   ushort4 b1=ushort4(0);if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
   ushort4 b2=ushort4(0);if(column+2<p.columns)b2=*(device const ushort4*)(weight+(column+2)*p.inner+k);
   ushort4 b3=ushort4(0);if(column+3<p.columns)b3=*(device const ushort4*)(weight+(column+3)*p.inner+k);
   ushort4 b4=ushort4(0);if(column+4<p.columns)b4=*(device const ushort4*)(weight+(column+4)*p.inner+k);
   ushort4 b5=ushort4(0);if(column+5<p.columns)b5=*(device const ushort4*)(weight+(column+5)*p.inner+k);
   ushort4 b6=ushort4(0);if(column+6<p.columns)b6=*(device const ushort4*)(weight+(column+6)*p.inner+k);
   ushort4 b7=ushort4(0);if(column+7<p.columns)b7=*(device const ushort4*)(weight+(column+7)*p.inner+k);
   ushort4 b8=ushort4(0);if(column+8<p.columns)b8=*(device const ushort4*)(weight+(column+8)*p.inner+k);
   ushort4 b9=ushort4(0);if(column+9<p.columns)b9=*(device const ushort4*)(weight+(column+9)*p.inner+k);
   ushort4 b10=ushort4(0);if(column+10<p.columns)b10=*(device const ushort4*)(weight+(column+10)*p.inner+k);
   ushort4 b11=ushort4(0);if(column+11<p.columns)b11=*(device const ushort4*)(weight+(column+11)*p.inner+k);
   ushort4 b12=ushort4(0);if(column+12<p.columns)b12=*(device const ushort4*)(weight+(column+12)*p.inner+k);
   ushort4 b13=ushort4(0);if(column+13<p.columns)b13=*(device const ushort4*)(weight+(column+13)*p.inner+k);
   ushort4 b14=ushort4(0);if(column+14<p.columns)b14=*(device const ushort4*)(weight+(column+14)*p.inner+k);
   ushort4 b15=ushort4(0);if(column+15<p.columns)b15=*(device const ushort4*)(weight+(column+15)*p.inner+k);
   for(uint j=0;j<4;++j) {
    float4 w0=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);
    sum0_0+=x[j]*w0;if(p.rows==2)sum1_0+=y[j]*w0;
    float4 w1=as_type<float4>(uint4(b4[j],b5[j],b6[j],b7[j])<<16);
    sum0_1+=x[j]*w1;if(p.rows==2)sum1_1+=y[j]*w1;
    float4 w2=as_type<float4>(uint4(b8[j],b9[j],b10[j],b11[j])<<16);
    sum0_2+=x[j]*w2;if(p.rows==2)sum1_2+=y[j]*w2;
    float4 w3=as_type<float4>(uint4(b12[j],b13[j],b14[j],b15[j])<<16);
    sum0_3+=x[j]*w3;if(p.rows==2)sum1_3+=y[j]*w3;
   }
  }
 } else {
  for(ulong k=lane;k<p.inner;k+=32) {
   float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
   float4 w0=0.0f;
   if(column+0<p.columns)w0.x=as_type<float>(uint(weight[(column+0)*p.inner+k])<<16);
   if(column+1<p.columns)w0.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
   if(column+2<p.columns)w0.z=as_type<float>(uint(weight[(column+2)*p.inner+k])<<16);
   if(column+3<p.columns)w0.w=as_type<float>(uint(weight[(column+3)*p.inner+k])<<16);
   sum0_0+=x*w0;if(p.rows==2)sum1_0+=y*w0;
   float4 w1=0.0f;
   if(column+4<p.columns)w1.x=as_type<float>(uint(weight[(column+4)*p.inner+k])<<16);
   if(column+5<p.columns)w1.y=as_type<float>(uint(weight[(column+5)*p.inner+k])<<16);
   if(column+6<p.columns)w1.z=as_type<float>(uint(weight[(column+6)*p.inner+k])<<16);
   if(column+7<p.columns)w1.w=as_type<float>(uint(weight[(column+7)*p.inner+k])<<16);
   sum0_1+=x*w1;if(p.rows==2)sum1_1+=y*w1;
   float4 w2=0.0f;
   if(column+8<p.columns)w2.x=as_type<float>(uint(weight[(column+8)*p.inner+k])<<16);
   if(column+9<p.columns)w2.y=as_type<float>(uint(weight[(column+9)*p.inner+k])<<16);
   if(column+10<p.columns)w2.z=as_type<float>(uint(weight[(column+10)*p.inner+k])<<16);
   if(column+11<p.columns)w2.w=as_type<float>(uint(weight[(column+11)*p.inner+k])<<16);
   sum0_2+=x*w2;if(p.rows==2)sum1_2+=y*w2;
   float4 w3=0.0f;
   if(column+12<p.columns)w3.x=as_type<float>(uint(weight[(column+12)*p.inner+k])<<16);
   if(column+13<p.columns)w3.y=as_type<float>(uint(weight[(column+13)*p.inner+k])<<16);
   if(column+14<p.columns)w3.z=as_type<float>(uint(weight[(column+14)*p.inner+k])<<16);
   if(column+15<p.columns)w3.w=as_type<float>(uint(weight[(column+15)*p.inner+k])<<16);
   sum0_3+=x*w3;if(p.rows==2)sum1_3+=y*w3;
  }
 }
 sum0_0=float4(simd_sum(sum0_0.x),simd_sum(sum0_0.y),simd_sum(sum0_0.z),simd_sum(sum0_0.w));
 sum1_0=float4(simd_sum(sum1_0.x),simd_sum(sum1_0.y),simd_sum(sum1_0.z),simd_sum(sum1_0.w));
 if(lane==0)for(uint j=0;j<4&&column+0+j<p.columns;++j){out[column+0+j]=sum0_0[j];if(p.rows==2)out[p.oStride+column+0+j]=sum1_0[j];}
 sum0_1=float4(simd_sum(sum0_1.x),simd_sum(sum0_1.y),simd_sum(sum0_1.z),simd_sum(sum0_1.w));
 sum1_1=float4(simd_sum(sum1_1.x),simd_sum(sum1_1.y),simd_sum(sum1_1.z),simd_sum(sum1_1.w));
 if(lane==0)for(uint j=0;j<4&&column+4+j<p.columns;++j){out[column+4+j]=sum0_1[j];if(p.rows==2)out[p.oStride+column+4+j]=sum1_1[j];}
 sum0_2=float4(simd_sum(sum0_2.x),simd_sum(sum0_2.y),simd_sum(sum0_2.z),simd_sum(sum0_2.w));
 sum1_2=float4(simd_sum(sum1_2.x),simd_sum(sum1_2.y),simd_sum(sum1_2.z),simd_sum(sum1_2.w));
 if(lane==0)for(uint j=0;j<4&&column+8+j<p.columns;++j){out[column+8+j]=sum0_2[j];if(p.rows==2)out[p.oStride+column+8+j]=sum1_2[j];}
 sum0_3=float4(simd_sum(sum0_3.x),simd_sum(sum0_3.y),simd_sum(sum0_3.z),simd_sum(sum0_3.w));
 sum1_3=float4(simd_sum(sum1_3.x),simd_sum(sum1_3.y),simd_sum(sum1_3.z),simd_sum(sum1_3.w));
 if(lane==0)for(uint j=0;j<4&&column+12+j<p.columns;++j){out[column+12+j]=sum0_3[j];if(p.rows==2)out[p.oStride+column+12+j]=sum1_3[j];}
}
