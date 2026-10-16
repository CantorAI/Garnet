"""Keep FP32 dot order; test one output per SIMD group on exact BF16 weights."""
from pathlib import Path
import sys
native=Path(sys.argv[1]).read_text()
begin=native.index('struct BF16Params')
end=native.index('// BF16 weights are expanded',begin)
base=native[begin:end]
# The original declarations preceding decode include helper kernels; preserve them.
source='#include <metal_stdlib>\nusing namespace metal;\n'+base
for threads in (32,64,128):
 source+=f'''
kernel void bf16_decode_single_t{threads}(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
 device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
 uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {{
 ulong column=ulong(group)*{threads//32}+simd;if(column>=p.columns)return;
 float sum0=0.0f,sum1=0.0f;
 if(p.inner%4==0) {{
  for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {{
   float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
   ushort4 bits=*(device const ushort4*)(weight+column*p.inner+k);
   for(uint j=0;j<4;++j){{float w=as_type<float>(uint(bits[j])<<16);sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;}}
  }}
 }} else {{
  for(ulong k=lane;k<p.inner;k+=32) {{float w=as_type<float>(uint(weight[column*p.inner+k])<<16);
   sum0+=a[k]*w;if(p.rows==2)sum1+=a[p.aStride+k]*w;}}
 }}
 sum0=simd_sum(sum0);sum1=simd_sum(sum1);
 if(lane==0){{out[column]=sum0;if(p.rows==2)out[p.oStride+column]=sum1;}}
}}
'''
Path(sys.argv[2]).write_bytes(source.encode())
