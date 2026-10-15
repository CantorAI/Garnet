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

// Lossless half components/weights, with explicit float SIMD-matrix accumulation.
struct MixedP {ulong rows,inner,columns,leftStride,weightStride,outputStride;};
kernel void coalesced_product(device const half* input [[buffer(0)]],device const half* weight [[buffer(1)]],
 device float* output [[buffer(2)]],constant MixedP& p [[buffer(3)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 threadgroup half left[8*32],right[32*36];threadgroup float result[8*32];
 ulong rowBase=ulong(group.y)*8,columnBase=ulong(group.x)*32;
 simdgroup_float8x8 accum=make_filled_simdgroup_matrix<float,8>(0.0f);
 for(ulong base=0;base<p.inner;base+=32) {
  for(uint i=tid;i<8*32;i+=128){ulong row=rowBase+i/32,k=base+i%32;left[i]=(row<p.rows&&k<p.inner)?input[row*p.leftStride+k]:half(0.0f);}
  // Each SIMD loads adjacent K values from one checkpoint row. Padding avoids
  // the 32-column transpose's worst shared-memory bank conflict pattern.
  for(uint i=tid;i<32*32;i+=128){ulong column=columnBase+i/32,k=base+i%32;right[(i%32)*36+i/32]=(column<p.columns&&k<p.inner)?weight[column*p.weightStride+k]:half(0.0f);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint offset=0;offset<32;offset+=8){simdgroup_half8x8 a,b;simdgroup_load(a,left+offset,32);simdgroup_load(b,right+offset*36+simd*8,36);simdgroup_multiply_accumulate(accum,a,b,accum);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 simdgroup_store(accum,result+simd*8,32);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<8*32;i+=128){ulong row=rowBase+i/32,column=columnBase+i%32;if(row<p.rows&&column<p.columns)output[row*p.outputStride+column]=result[i];}
}
