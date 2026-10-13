#include <metal_stdlib>
using namespace metal;
struct P {ulong rows,inner,columns,aStride,hStride,oStride;};
kernel void split_activation(device const float* input [[buffer(0)]],device half* hi [[buffer(1)]],
 device half* lo [[buffer(2)]],device half* tail [[buffer(3)]],device int* scales [[buffer(4)]],
 device atomic_uint* failures [[buffer(5)]],constant P& p [[buffer(6)]],uint row [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 threadgroup float partial[4];float largest=0;
 for(ulong k=tid;k<p.inner;k+=128)largest=max(largest,abs(input[row*p.aStride+k]));
 largest=simd_max(largest);if(lane==0)partial[simd]=largest;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 largest=simd_max(lane<4?partial[lane]:0.0f);int exponent=0;
 if(largest!=0)frexp(largest,exponent);int scale=largest==0?0:9-exponent;
 if(tid==0)scales[row]=scale;
 for(ulong k=tid;k<p.inner;k+=128){float original=input[row*p.aStride+k],x=ldexp(original,scale);
  half a=half(x),b=half((x-float(a))*4096.0f),c=half((x-float(a)-float(b)/4096.0f)*16777216.0f);
  if(original==0.0f){a=half(original);b=a;c=a;}
  float recovered=ldexp(float(a)+float(b)/4096.0f+float(c)/16777216.0f,-scale);
  if(!isfinite(original)||as_type<uint>(recovered)!=as_type<uint>(original))atomic_fetch_add_explicit(failures,1u,memory_order_relaxed);
  hi[row*p.hStride+k]=a;lo[row*p.hStride+k]=b;tail[row*p.hStride+k]=c;
 }
}
kernel void combine_split(device const float* hi [[buffer(0)]],device const float* lo [[buffer(1)]],
 device const float* tail [[buffer(2)]],device const int* rowScales [[buffer(3)]],
 device const int* colScales [[buffer(4)]],device float* output [[buffer(5)]],constant P& p [[buffer(6)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.rows*p.columns)return;ulong row=id/p.columns,col=id%p.columns,offset=row*p.oStride+col;
 output[offset]=ldexp(hi[offset]+lo[offset]/4096.0f+tail[offset]/16777216.0f,-rowScales[row]-colScales[col]);
}
kernel void combine_control(device const float* input [[buffer(0)]],device const int* scales [[buffer(1)]],
 device float* output [[buffer(2)]],constant P& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.rows*p.columns)return;ulong row=id/p.columns,col=id%p.columns;
 output[row*p.oStride+col]=ldexp(input[row*p.oStride+col],-scales[col]);
}
