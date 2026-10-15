// Component-only exact input decomposition; compile with fast math disabled.
#include <metal_stdlib>
using namespace metal;
struct P{ulong rows,inner,columns,aStride,hStride,oStride;};
kernel void joined_split(device const float* input [[buffer(0)]],device half* joined [[buffer(1)]],
 device int* scales [[buffer(2)]],device atomic_uint* failures [[buffer(3)]],constant P& p [[buffer(4)]],
 uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]){
 threadgroup float partial[4];float largest=0;
 for(ulong k=tid;k<p.inner;k+=128)largest=max(largest,abs(input[row*p.aStride+k]));
 largest=simd_max(largest);if(lane==0)partial[simd]=largest;threadgroup_barrier(mem_flags::mem_threadgroup);
 largest=simd_max(lane<4?partial[lane]:0.0f);int exponent=0;if(largest!=0)frexp(largest,exponent);int scale=largest==0?0:9-exponent;
 if(tid==0)scales[row]=scale;
 for(ulong k=tid;k<p.inner;k+=128){float original=input[row*p.aStride+k],x=ldexp(original,scale);
  half a=half(x),b=half((x-float(a))*4096.0f),c=half((x-float(a)-float(b)/4096.0f)*16777216.0f);
  if(original==0.0f){a=half(original);b=a;c=a;}
  float recovered=ldexp(float(a)+float(b)/4096.0f+float(c)/16777216.0f,-scale);
  if(!isfinite(original)||as_type<uint>(recovered)!=as_type<uint>(original))atomic_fetch_add_explicit(failures,1u,memory_order_relaxed);
  joined[(row*3)*p.hStride+k]=a;joined[(row*3+1)*p.hStride+k]=b;joined[(row*3+2)*p.hStride+k]=c;
 }
}
kernel void joined_combine(device const float* parts [[buffer(0)]],device const int* rowScales [[buffer(1)]],
 device const int* colScales [[buffer(2)]],device float* output [[buffer(3)]],constant P& p [[buffer(4)]],uint id [[thread_position_in_grid]]){
 if(ulong(id)>=p.rows*p.columns)return;ulong row=ulong(id)/p.columns,col=ulong(id)%p.columns;
 output[row*p.oStride+col]=ldexp(parts[(row*3)*p.oStride+col]+parts[(row*3+1)*p.oStride+col]/4096.0f+parts[(row*3+2)*p.oStride+col]/16777216.0f,-rowScales[row]-colScales[col]);
}
