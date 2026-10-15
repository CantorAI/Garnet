"""Native Metal BF16 conversion and vector dot; same stored weight bits."""
from pathlib import Path
import sys
source=Path(sys.argv[1]).read_text();start=source.index('kernel void bf16_decode(');end=source.index('kernel void bf16_decode_small(',start);control=source[start:end]
codes=['#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n',control]
for dot,threads in ((False,64),(False,128),(True,64),(True,128)):
    name=f'bf16_native_{"dot" if dot else "serial"}_t{threads}'
    changed=control.replace('kernel void bf16_decode(',f'kernel void {name}(',1)
    changed=changed.replace('device const ushort* weight','device const bfloat* weight').replace('ushort4','bfloat4')
    changed=changed.replace('ulong column=ulong(group)*16+simd*4;',f'ulong column=ulong(group)*{threads//32*4}+simd*4;')
    changed=changed.replace('as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16)','float4(float(b0[j]),float(b1[j]),float(b2[j]),float(b3[j]))')
    for offset,component in ((0,'x'),(1,'y'),(2,'z'),(3,'w')):
        index='column' if not offset else f'(column+{offset})'
        changed=changed.replace(f'as_type<float>(uint(weight[{index}*p.inner+k])<<16)',f'float(weight[{index}*p.inner+k])')
    if dot:
        old='''            for(uint j=0;j<4;++j) {
                float4 w=float4(float(b0[j]),float(b1[j]),float(b2[j]),float(b3[j]));
                sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
            }'''
        new='''            float4 w0=float4(b0),w1=float4(b1),w2=float4(b2),w3=float4(b3);
            sum0+=float4(dot(x,w0),dot(x,w1),dot(x,w2),dot(x,w3));
            if(p.rows==2)sum1+=float4(dot(y,w0),dot(y,w1),dot(y,w2),dot(y,w3));'''
        assert old in changed;changed=changed.replace(old,new)
    assert 'uint(weight[' not in changed and 'ushort4' not in changed
    codes.append(changed)
Path(sys.argv[2]).write_bytes(''.join(codes).encode())
