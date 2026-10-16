"""Separate FP32 partial dot products and FP32 reduction, with exact BF16 weights."""
from pathlib import Path
import sys
native=Path(sys.argv[1]).read_text();start=native.index('kernel void bf16_decode(');small=native.index('kernel void bf16_decode_small(',start);end=native.index('// BF16 weights are expanded',small)
body=native[start:small]
code='#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+native[start:end]
for parts in (2,4):
    partial=body.replace('bf16_decode(',f'bf16_decode_sk{parts}(')
    partial=partial.replace('ulong column=ulong(group)*16+simd*4;',f'uint part=group%{parts};ulong column=ulong(group/{parts})*16+simd*4;')
    partial=partial.replace('k=ulong(lane)*4;k<p.inner;k+=128',f'k=ulong(lane)*4+ulong(part)*128;k<p.inner;k+={128*parts}')
    partial=partial.replace('k=lane;k<p.inner;k+=32',f'k=lane+ulong(part)*32;k<p.inner;k+={32*parts}')
    partial=partial.replace('out[column+j]', 'out[ulong(part)*p.rows*p.oStride+column+j]').replace('out[p.oStride+column+j]','out[(ulong(part)*p.rows+1)*p.oStride+column+j]')
    assert 'group)*16' not in partial and f'k+={128*parts}' in partial
    code+=partial+f'''kernel void bf16_decode_sk{parts}_reduce(device const float* partial [[buffer(0)]],device float* out [[buffer(1)]],
 constant BF16Params& p [[buffer(2)]],uint id [[thread_position_in_grid]]) {{
 if(ulong(id)>=p.rows*p.columns)return;
 ulong row=ulong(id)/p.columns,column=ulong(id)%p.columns;
 float sum=0.0f;
 for(uint part=0;part<{parts};++part)sum+=partial[(ulong(part)*p.rows+row)*p.oStride+column];
 out[row*p.oStride+column]=sum;
}}
'''
Path(sys.argv[2]).write_bytes(code.encode())
