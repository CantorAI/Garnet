"""Generate wider value tiles from the original recurrence without changing per-value FP32 expressions."""
import re
def original_kernel(source):
 start=source.index('kernel void gated_delta_chunk(');end=source.index('\nstruct LayoutParams ',start);return source[start:end]
def arithmetic_block(source):
 original=original_kernel(source);start=original.index('            float4 memory=0.0f;');end=original.index('\n        }else if(lane==0)',start);return original[start:end]
def band(text,index):
 for word in ('memories','memory','change','result'):text=re.sub(r'\b'+word+r'\b',word+'_b'+str(index),text)
 return re.sub(r'\bcolumn\b','(column+'+str(index*16)+')',text)
def kernel(source,width):
 assert width in (16,32,64);original=original_kernel(source)
 if width==16:return original.replace('kernel void gated_delta_chunk(','kernel void gated_delta_chunk_value16(')
 def replace(old,new,count=1):
  nonlocal original
  assert original.count(old)==count,(old,original.count(old),count);original=original.replace(old,new)
 replace('kernel void gated_delta_chunk(','kernel void gated_delta_chunk_value'+str(width)+'(')
 replace('ulong tiles=p.valueDim/16,headId=ulong(id)/tiles;','ulong tiles=p.valueDim/'+str(width)+',headId=ulong(id)/tiles;')
 replace('column=(ulong(id)%tiles)*16+simd*4;','column=(ulong(id)%tiles)*'+str(width)+'+simd*4;')
 replace('    float4 memories[4];','\n'.join('    float4 memories_b'+str(i)+'[4];' for i in range(width//16)))
 load='    for(uint j=0;j<4;++j)memories[j]=*(device const float4*)(state+stateBase+(ulong(lane)+32*j)*p.valueDim+column);'
 replace(load,'\n'.join(band(load,i) for i in range(width//16)))
 block=arithmetic_block(source);replace(block,'\n'.join(band(block,i) for i in range(width//16)))
 zero='        }else if(lane==0)*(device float4*)(output+outputBase+column)=float4(0.0f);'
 replace(zero,'        }else if(lane==0){\n'+'\n'.join(band('            *(device float4*)(output+outputBase+column)=float4(0.0f);',i) for i in range(width//16))+'\n        }')
 store='    for(uint j=0;j<4;++j)*(device float4*)(nextState+nextBase+(ulong(lane)+32*j)*p.valueDim+column)=memories[j];'
 replace(store,'\n'.join(band(store,i) for i in range(width//16)))
 return original
def shader(source):
 start=source.index('struct DeltaChunkParams ');end=source.index('\nkernel void gated_delta_chunk(',start);params=source[start:end]
 return '#include <metal_stdlib>\nusing namespace metal;\n'+params+'\n'+'\n'.join(kernel(source,width) for width in (16,32,64))
