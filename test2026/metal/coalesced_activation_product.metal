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
  // Each SIMD loads adjacent K values from one checkpoint row. Padding was a
  // bank-conflict hypothesis; no hardware-counter attribution was established.
  for(uint i=tid;i<32*32;i+=128){ulong column=columnBase+i/32,k=base+i%32;right[(i%32)*36+i/32]=(column<p.columns&&k<p.inner)?weight[column*p.weightStride+k]:half(0.0f);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint offset=0;offset<32;offset+=8){simdgroup_half8x8 a,b;simdgroup_load(a,left+offset,32);simdgroup_load(b,right+offset*36+simd*8,36);simdgroup_multiply_accumulate(accum,a,b,accum);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 simdgroup_store(accum,result+simd*8,32);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<8*32;i+=128){ulong row=rowBase+i/32,column=columnBase+i%32;if(row<p.rows&&column<p.columns)output[row*p.outputStride+column]=result[i];}
}
