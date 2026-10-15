// Batched FP32 matrix product for the isolated parallel-delta probe.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
struct G {ulong rows,inner,columns,aRows,aCols,bRows,bCols,transposeA,transposeB;};
kernel void delta_tiled_gemm(device const float* a [[buffer(0)]],device const float* b [[buffer(1)]],
 device float* out [[buffer(2)]],constant G& p [[buffer(3)]],uint3 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 threadgroup float left[16*32],right[32*32],result[16*32];
 ulong rowBase=ulong(group.y)*16,colBase=ulong(group.x)*32;
 ulong ab=ulong(group.z)*p.aRows*p.aCols,bb=ulong(group.z)*p.bRows*p.bCols;
 simdgroup_float8x8 accum0=make_filled_simdgroup_matrix<float,8>(0.0f);
 simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);
 for(ulong base=0;base<p.inner;base+=32) {
  for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,k=base+i%32;
   left[i]=(row<p.rows&&k<p.inner)?a[ab+(p.transposeA?k*p.aCols+row:row*p.aCols+k)]:0.0f;}
  for(uint i=tid;i<32*32;i+=128){ulong k=base+i/32,col=colBase+i%32;
   right[i]=(k<p.inner&&col<p.columns)?b[bb+(p.transposeB?col*p.bCols+k:k*p.bCols+col)]:0.0f;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint offset=0;offset<32;offset+=8) {
   simdgroup_float8x8 l0,l1,r;
   simdgroup_load(l0,left+offset,32);simdgroup_load(l1,left+8*32+offset,32);
   simdgroup_load(r,right+offset*32+simd*8,32);
   simdgroup_multiply_accumulate(accum0,l0,r,accum0);simdgroup_multiply_accumulate(accum1,l1,r,accum1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 simdgroup_store(accum0,result+simd*8,32);simdgroup_store(accum1,result+8*32+simd*8,32);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<16*32;i+=128){ulong row=rowBase+i/32,col=colBase+i%32;
  if(row<p.rows&&col<p.columns)out[(ulong(group.z)*p.rows+row)*p.columns+col]=result[i];}
}
