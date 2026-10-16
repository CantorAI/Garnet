"""Generate isolated BF16 decode input-tile variants from the native control."""
from pathlib import Path
import sys
native=Path(sys.argv[1]).read_text();start=native.index('struct BF16Params');end=native.index('// BF16 weights are expanded',start)
control=native[start:end];kernel_start=control.index('kernel void bf16_decode(');kernel=control[kernel_start:]
variants=[]
for tile in (8,16):
 code=kernel.replace('kernel void bf16_decode(','kernel void bf16_decode'+str(tile)+'(',1)
 code=code.replace('if(p.inner%4==0)','if(p.inner%'+str(tile)+'==0)',1)
 code=code.replace('for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {','for(ulong k=ulong(lane)*'+str(tile)+';k<p.inner;k+='+str(32*tile)+') {\n            for(uint block=0;block<'+str(tile//4)+';++block){ulong kk=k+block*4;',1)
 begin=code.index('            float4 x=');finish=code.index('\n        }\n    } else',begin)
 body=code[begin:finish].replace('a+k','a+kk').replace('p.aStride+k','p.aStride+kk').replace('p.inner+k','p.inner+kk')
 code=code[:begin]+body+'\n            }'+code[finish:];variants.append(code)
Path(sys.argv[2]).write_bytes(('#include <metal_stdlib>\nusing namespace metal;\n'+control+''.join(variants)).encode())
