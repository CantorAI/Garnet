"""Two 16-lane output groups per SIMD, with original 32-way partial sums."""
from pathlib import Path
import sys
native=Path(sys.argv[1]).read_text();start=native.index('kernel void bf16_decode(');end=native.index('kernel void bf16_decode_small(',start)
codes=['#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n',native[start:end]]
for threads in (64,128):
    code=f'''kernel void bf16_decode_s16_t{threads}(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
 device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
 uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {{
 uint local=lane%16;ulong column=ulong(group)*{threads//32*8}+simd*8+(lane/16)*4;
 if(column>=p.columns)return;
 float4 sum00=0.0f,sum01=0.0f,sum10=0.0f,sum11=0.0f;
 if(p.inner%4==0) {{
  for(ulong k=ulong(local)*4;k<p.inner;k+=128) {{
'''
    for part in (0,1):
        offset='k' if part==0 else 'k+64'
        code+=f'   if({offset}<p.inner) {{float4 x=*(device const float4*)(a+{offset}),y=p.rows==2?*(device const float4*)(a+p.aStride+{offset}):float4(0.0f);\n'
        for col in range(4):code+=f'    ushort4 b{col}=ushort4(0);if(column+{col}<p.columns)b{col}=*(device const ushort4*)(weight+(column+{col})*p.inner+{offset});\n'
        code+=f'    for(uint j=0;j<4;++j) {{float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);sum0{part}+=x[j]*w;if(p.rows==2)sum1{part}+=y[j]*w;}}\n   }}\n'
    code+='  }\n } else {\n  for(ulong k=local;k<p.inner;k+=32) {\n'
    for part in (0,1):
        offset='k' if part==0 else 'k+16'
        code+=f'   if({offset}<p.inner) {{float x=a[{offset}],y=p.rows==2?a[p.aStride+{offset}]:0.0f;float4 w=0.0f;\n'
        for col,axis in enumerate('xyzw'):code+=f'    if(column+{col}<p.columns)w.{axis}=as_type<float>(uint(weight[(column+{col})*p.inner+{offset}])<<16);\n'
        code+=f'    sum0{part}+=x*w;if(p.rows==2)sum1{part}+=y*w;\n   }}\n'
    code+='  }\n }\n float4 sum0=sum00+sum01,sum1=sum10+sum11;\n'
    for mask in (8,4,2,1):code+=f' sum0+=simd_shuffle_xor(sum0,{mask});sum1+=simd_shuffle_xor(sum1,{mask});\n'
    code+=' if(local==0)for(uint j=0;j<4&&column+j<p.columns;++j){out[column+j]=sum0[j];if(p.rows==2)out[p.oStride+column+j]=sum1[j];}\n}\n'
    codes.append(code)
Path(sys.argv[2]).write_bytes(''.join(codes).encode())
