"""Generate isolated work-distribution variants with unchanged per-output math."""
from pathlib import Path
import sys

native=Path(sys.argv[1]).read_text()
start=native.index('struct BF16Params')
end=native.index('// BF16 weights are expanded',start)
control=native[start:end]
kernel=control[control.index('kernel void bf16_decode('):]
variants=[]
for outputs,threads in ((2,64),(2,128),(4,64),(4,256)):
 name='bf16_decode_c'+str(outputs)+'_t'+str(threads)
 code=kernel.replace('kernel void bf16_decode(','kernel void '+name+'(',1)
 code=code.replace('ulong column=ulong(group)*16+simd*4;',
                   'ulong column=ulong(group)*'+str(outputs*threads//32)+'+simd*'+str(outputs)+';',1)
 if outputs==2:
  code=code.replace('float4 sum0=0.0f,sum1=0.0f;','float2 sum0=0.0f,sum1=0.0f;')
  code=code.replace(',b2=ushort4(0),b3=ushort4(0)','')
  for j in (2,3):
   line='            if(column+'+str(j)+'<p.columns)b'+str(j)+'=*(device const ushort4*)(weight+(column+'+str(j)+')*p.inner+k);\n'
   assert line in code;code=code.replace(line,'')
  code=code.replace('float4 w=as_type<float4>(uint4(b0[j],b1[j],b2[j],b3[j])<<16);',
                    'float2 w=as_type<float2>(uint2(b0[j],b1[j])<<16);')
  code=code.replace('float4 w=0.0f;','float2 w=0.0f;')
  for j,axis in ((2,'z'),(3,'w')):
   line='        if(column+'+str(j)+'<p.columns)w.'+axis+'=as_type<float>(uint(weight[(column+'+str(j)+')*p.inner+k])<<16);\n'
   assert line in code;code=code.replace(line,'')
  for row in (0,1):
   code=code.replace('sum'+str(row)+'=float4(simd_sum(sum'+str(row)+'.x),simd_sum(sum'+str(row)+'.y),simd_sum(sum'+str(row)+'.z),simd_sum(sum'+str(row)+'.w));',
                     'sum'+str(row)+'=float2(simd_sum(sum'+str(row)+'.x),simd_sum(sum'+str(row)+'.y));')
  code=code.replace('for(uint j=0;j<4&&column+j<p.columns;',
                    'for(uint j=0;j<2&&column+j<p.columns;')
 variants.append(code)
Path(sys.argv[2]).write_bytes(('#include <metal_stdlib>\nusing namespace metal;\n'+control+''.join(variants)).encode())
