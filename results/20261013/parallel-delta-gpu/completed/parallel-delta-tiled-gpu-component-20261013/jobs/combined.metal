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
#include <metal_stdlib>
using namespace metal;
struct P {ulong batch,chunk,heads,keyDim,valueDim,pStride,sStride,aStride,oStride,nextStride;};
kernel void pack_delta(device const float* src [[buffer(0)]],device const float* active [[buffer(1)]],
 device float* q [[buffer(2)]],device float* k [[buffer(3)]],device float* v [[buffer(4)]],
 device float* factors [[buffer(5)]],device float* betas [[buffer(6)]],constant P& p [[buffer(7)]],
 uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 ulong token=group%p.chunk,headId=group/p.chunk,head=headId%p.heads,row=headId/p.heads;
 if(headId>=p.batch*p.heads)return;
 ulong base=(row*p.chunk+token)*p.pStride,qw=p.heads*p.keyDim,vw=p.heads*p.valueDim;
 ulong qi=(headId*p.chunk+token)*p.keyDim+tid,vi=(headId*p.chunk+token)*p.valueDim+tid;
 if(active[row*p.aStride+token]==0.0f){q[qi]=0;k[qi]=0;v[vi]=0;if(tid==0){factors[group]=1;betas[group]=0;}return;}
 float x=src[base+head*p.keyDim+tid],y=src[base+qw+head*p.keyDim+tid];
 float xs=simd_sum(x*x),ys=simd_sum(y*y);threadgroup float xp[4],yp[4];
 if(lane==0){xp[simd]=xs;yp[simd]=ys;}threadgroup_barrier(mem_flags::mem_threadgroup);
 xs=simd_sum(lane<4?xp[lane]:0.0f)+1e-6f;ys=simd_sum(lane<4?yp[lane]:0.0f)+1e-6f;
 q[qi]=x*(rsqrt(xs)*rsqrt(float(p.keyDim)));k[qi]=y*rsqrt(ys);
 v[vi]=src[base+2*qw+head*p.valueDim+tid];
 if(tid==0){factors[group]=exp(src[base+2*qw+vw+head]);betas[group]=src[base+2*qw+vw+p.heads+head];}
}
kernel void decay_products(device const float* f [[buffer(0)]],device float* dr [[buffer(1)]],
 device float* cumulative [[buffer(2)]],constant P& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.batch*p.heads*p.chunk)return;
 ulong h=id/p.chunk,j=id%p.chunk;float factor=1;
 for(ulong i=0;i<p.chunk;++i){
  ulong dst=(h*p.chunk+i)*p.chunk+j;
  if(i<j)dr[dst]=0;else if(i==j)dr[dst]=1;else{factor*=f[h*p.chunk+i];dr[dst]=factor;}
 }
 if(j==0){factor=1;for(ulong i=0;i<p.chunk;++i){factor*=f[h*p.chunk+i];cumulative[h*p.chunk+i]=factor;}}
}
kernel void coefficients(device const float* gram [[buffer(0)]],device const float* qk [[buffer(1)]],
 device const float* dr [[buffer(2)]],device const float* beta [[buffer(3)]],
 device float* lower [[buffer(4)]],device float* output_coeff [[buffer(5)]],constant P& p [[buffer(6)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.batch*p.heads*p.chunk*p.chunk)return;
 ulong j=id%p.chunk,i=(id/p.chunk)%p.chunk,h=id/(p.chunk*p.chunk);
 lower[id]=j<i?beta[h*p.chunk+i]*dr[id]*gram[id]:0;
 output_coeff[id]=j<=i?dr[id]*qk[id]:0;
}
kernel void rhs_delta(device const float* value [[buffer(0)]],device const float* ks [[buffer(1)]],
 device const float* cum [[buffer(2)]],device const float* beta [[buffer(3)]],device float* rhs [[buffer(4)]],
 constant P& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.batch*p.heads*p.chunk*p.valueDim)return;
 ulong row=id/p.valueDim;rhs[id]=beta[row]*(value[id]-cum[row]*ks[id]);
}
kernel void solve_delta(device const float* lower [[buffer(0)]],device const float* rhs [[buffer(1)]],
 device float* u [[buffer(2)]],constant P& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.batch*p.heads*p.valueDim)return;
 ulong h=id/p.valueDim,col=id%p.valueDim;
 for(ulong i=0;i<p.chunk;++i){
  float sum=0;for(ulong j=0;j<i;++j)sum+=lower[(h*p.chunk+i)*p.chunk+j]*u[(h*p.chunk+j)*p.valueDim+col];
  u[(h*p.chunk+i)*p.valueDim+col]=rhs[(h*p.chunk+i)*p.valueDim+col]-sum;
 }
}
kernel void scale_updates(device const float* u [[buffer(0)]],device const float* dr [[buffer(1)]],
 device float* scaled [[buffer(2)]],constant P& p [[buffer(3)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.batch*p.heads*p.chunk*p.valueDim)return;
 ulong h=id/(p.chunk*p.valueDim),token=(id/p.valueDim)%p.chunk;
 scaled[id]=u[id]*dr[(h*p.chunk+p.chunk-1)*p.chunk+token];
}
kernel void finish_output(device const float* qs [[buffer(0)]],device const float* weighted [[buffer(1)]],
 device const float* cum [[buffer(2)]],device const float* active [[buffer(3)]],device float* out [[buffer(4)]],
 constant P& p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.batch*p.heads*p.chunk*p.valueDim)return;
 ulong h=id/(p.chunk*p.valueDim),head=h%p.heads,row=h/p.heads,token=(id/p.valueDim)%p.chunk,col=id%p.valueDim;
 ulong dst=(row*p.chunk+token)*p.oStride+head*p.valueDim+col;
 out[dst]=active[row*p.aStride+token]!=0?cum[h*p.chunk+token]*qs[id]+weighted[id]:0;
}
kernel void finish_state(device const float* prior [[buffer(0)]],device const float* updates [[buffer(1)]],
 device const float* cum [[buffer(2)]],device float* out [[buffer(3)]],constant P& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
 if(id>=p.batch*p.heads*p.keyDim*p.valueDim)return;
 ulong h=id/(p.keyDim*p.valueDim),row=h/p.heads,head=h%p.heads,inner=id%(p.keyDim*p.valueDim);
 out[row*p.nextStride+head*p.keyDim*p.valueDim+inner]=cum[h*p.chunk+p.chunk-1]*prior[row*p.sStride+head*p.keyDim*p.valueDim+inner]+updates[id];
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
