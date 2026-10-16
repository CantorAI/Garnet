"""Four independent eight-lane, two-output groups per SIMD; FP32 throughout."""
from pathlib import Path
import sys
native=Path(sys.argv[1]).read_text()
start=native.index('kernel void bf16_decode(')
end=native.index('// BF16 weights are expanded',start)
code='#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+native[start:end]
for threads in (64,128):
    code+=f'''kernel void bf16_decode_s8_t{threads}(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
 device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
 uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {{
 uint local=lane%8;ulong column=ulong(group)*{threads//32*8}+simd*8+(lane/8)*2;
 if(column>=p.columns)return;
 float2 sum0=0.0f,sum1=0.0f;
 if(p.inner%4==0) {{
  for(ulong k=ulong(local)*4;k<p.inner;k+=32) {{
   float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
   ushort4 b0=*(device const ushort4*)(weight+column*p.inner+k),b1=ushort4(0);
   if(column+1<p.columns)b1=*(device const ushort4*)(weight+(column+1)*p.inner+k);
   for(uint j=0;j<4;++j) {{float2 w=as_type<float2>(uint2(b0[j],b1[j])<<16);sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;}}
  }}
 }} else {{
  for(ulong k=local;k<p.inner;k+=8) {{
   float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;
   float2 w=float2(as_type<float>(uint(weight[column*p.inner+k])<<16),0.0f);
   if(column+1<p.columns)w.y=as_type<float>(uint(weight[(column+1)*p.inner+k])<<16);
   sum0+=x*w;if(p.rows==2)sum1+=y*w;
  }}
 }}
'''
    for mask in (4,2,1):
        code+=f' sum0+=simd_shuffle_xor(sum0,{mask});sum1+=simd_shuffle_xor(sum1,{mask});\n'
    code+=' if(local==0)for(uint j=0;j<2&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}\n}\n'
Path(sys.argv[2]).write_bytes(code.encode())
