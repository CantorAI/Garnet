"""Combine original gate/up dot products and original SiLU expression."""
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text();start=s.index('kernel void bf16_decode(');end=s.index('kernel void bf16_decode_small(',start);control=s[start:end]
start=s.index('kernel void silu_mul(');end=s.index('kernel void add_matrix(',start);silu=s[start:end]
kernel=control.replace('kernel void bf16_decode(','kernel void fused_mlp_t128(',1)
kernel=kernel.replace('device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]]','device const ushort* upweight [[buffer(2)]],device float* out [[buffer(3)]],constant BF16Params& p [[buffer(4)]]')
kernel=kernel.replace('float4 sum0=0.0f,sum1=0.0f;','float4 sum0=0.0f,sum1=0.0f,up0=0.0f,up1=0.0f;')
start=kernel.index('            ushort4 b0=');end=kernel.index('            for(uint j=0;',start)
loads=kernel[start:end]
up_loads=loads.replace('weight','upweight')
for i in range(4):up_loads=up_loads.replace('b'+str(i),'u'+str(i))
kernel=kernel[:end]+up_loads+kernel[end:]
needle='                sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;'
assert needle in kernel
kernel=kernel.replace(needle,needle+'\n                float4 uw=as_type<float4>(uint4(u0[j],u1[j],u2[j],u3[j])<<16);\n                up0+=x[j]*uw;if(p.rows==2)up1+=y[j]*uw;')
start=kernel.index('        float4 w=0.0f;');end=kernel.index('        sum0+=x*w;',start)
up_loads=kernel[start:end].replace('weight','upweight').replace('float4 w=','float4 uw=').replace('w.','uw.')
kernel=kernel[:end]+up_loads+kernel[end:]
kernel=kernel.replace('        sum0+=x*w;if(p.rows==2)sum1+=y*w;','        sum0+=x*w;if(p.rows==2)sum1+=y*w;\n        up0+=x*uw;if(p.rows==2)up1+=y*uw;')
start=kernel.index('    sum0=float4(simd_sum');end=kernel.index('    if(lane==0)',start)
reductions=kernel[start:end].replace('sum0','up0').replace('sum1','up1')
kernel=kernel[:end]+reductions+kernel[end:]
kernel=kernel.replace('out[column+j]=sum0[j];','out[column+j]=(sum0[j]/(1.0f+exp(-sum0[j])))*up0[j];')
kernel=kernel.replace('out[p.oStride+column+j]=sum1[j];','out[p.oStride+column+j]=(sum1[j]/(1.0f+exp(-sum1[j])))*up1[j];')
other=kernel.replace('fused_mlp_t128','fused_mlp_t64').replace('ulong column=ulong(group)*16+simd*4;','ulong column=ulong(group)*8+simd*4;')
preamble='#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\nstruct Params {ulong rows,columns,aStride,bStride,cStride;float epsilon,offset;};\n'
Path(sys.argv[2]).write_bytes((preamble+control+silu+kernel+other).encode())
