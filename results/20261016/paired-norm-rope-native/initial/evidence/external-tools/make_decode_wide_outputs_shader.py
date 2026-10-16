"""Test eight/sixteen outputs per SIMD group without changing per-output math."""
from pathlib import Path
import sys
native=Path(sys.argv[1]).read_text()
start=native.index('kernel void bf16_decode(')
end=native.index('kernel void bf16_decode_small(',start)
control=native[start:end]
codes=['#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n',control]
for outputs,threads in ((8,64),(8,128),(16,64),(16,128)):
 blocks=outputs//4
 code=f'''kernel void bf16_decode_c{outputs}_t{threads}(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
 device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
 uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {{
 ulong column=ulong(group)*{outputs*threads//32}+simd*{outputs};
 if(column>=p.columns)return;
'''
 for b in range(blocks):code+=f' float4 sum0_{b}=0.0f,sum1_{b}=0.0f;\n'
 code+=' if(p.inner%4==0) {\n  for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {\n   float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);\n'
 for j in range(outputs):code+=f'   ushort4 b{j}=ushort4(0);if(column+{j}<p.columns)b{j}=*(device const ushort4*)(weight+(column+{j})*p.inner+k);\n'
 code+='   for(uint j=0;j<4;++j) {\n'
 for b in range(blocks):
  expr=','.join(f'b{b*4+j}[j]' for j in range(4))
  code+=f'    float4 w{b}=as_type<float4>(uint4({expr})<<16);\n    sum0_{b}+=x[j]*w{b};if(p.rows==2)sum1_{b}+=y[j]*w{b};\n'
 code+='   }\n  }\n } else {\n  for(ulong k=lane;k<p.inner;k+=32) {\n   float x=a[k],y=p.rows==2?a[p.aStride+k]:0.0f;\n'
 for b in range(blocks):
  code+=f'   float4 w{b}=0.0f;\n'
  for j,axis in enumerate('xyzw'):code+=f'   if(column+{b*4+j}<p.columns)w{b}.{axis}=as_type<float>(uint(weight[(column+{b*4+j})*p.inner+k])<<16);\n'
  code+=f'   sum0_{b}+=x*w{b};if(p.rows==2)sum1_{b}+=y*w{b};\n'
 code+='  }\n }\n'
 for b in range(blocks):
  for row in (0,1):
   variable=f'sum{row}_{b}';expr=','.join(f'simd_sum({variable}.{axis})' for axis in 'xyzw')
   code+=f' {variable}=float4({expr});\n'
  code+=f' if(lane==0)for(uint j=0;j<4&&column+{b*4}+j<p.columns;++j){{out[column+{b*4}+j]=sum0_{b}[j];if(p.rows==2)out[p.oStride+column+{b*4}+j]=sum1_{b}[j];}}\n'
 code+='}\n';codes.append(code)
Path(sys.argv[2]).write_bytes(''.join(codes).encode())
