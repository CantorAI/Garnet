"""Independent FP32 dot accumulators; changes summation order, not precision."""
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text();start=s.index('kernel void bf16_decode(');end=s.index('kernel void bf16_decode_small(',start);control=s[start:end]
loop=control.index('        for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {')
opening=control.index('{',loop);depth=1;closing=opening+1
while depth:
    if control[closing]=='{':depth+=1
    elif control[closing]=='}':depth-=1
    closing+=1
body=control[opening+1:closing-1]
codes=['#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n',control]
for factor,threads in ((2,64),(2,128),(4,64),(4,128)):
    name=f'bf16_decode_a{factor}_t{threads}'
    changed=''.join(f'        float4 part0_{i}=0.0f,part1_{i}=0.0f;\n' for i in range(factor))
    changed+=f'        for(ulong k=ulong(lane)*4;k<p.inner;k+={128*factor}) {{\n'
    for phase in range(factor):
        part=re.sub(r'\bk\b','kk',body).replace('sum0+=',f'part0_{phase}+=').replace('sum1+=',f'part1_{phase}+=')
        changed+=f'            {{ulong kk=k+{phase*128};if(kk<p.inner) {{'+part+'\n            }}\n'
    changed+='        }\n'
    for row in (0,1):
        expression=(f'part{row}_0+part{row}_1' if factor==2 else f'(part{row}_0+part{row}_1)+(part{row}_2+part{row}_3)')
        changed+=f'        sum{row}={expression};\n'
    code=control[:loop]+changed+control[closing:]
    code=code.replace('kernel void bf16_decode(',f'kernel void {name}(',1)
    code=code.replace('ulong column=ulong(group)*16+simd*4;',f'ulong column=ulong(group)*{threads//32*4}+simd*4;',1)
    codes.append(code)
Path(sys.argv[2]).write_bytes(''.join(codes).encode())
