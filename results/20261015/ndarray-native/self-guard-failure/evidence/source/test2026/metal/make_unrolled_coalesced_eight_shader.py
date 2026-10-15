"""Duplicate the original four-output block body with separate named accumulators.

Hypothesis only: indexed accumulator arrays in the first eight-output shader
changed compiler arithmetic. This revision retains the original lexical
arithmetic for each block. Exact bits remain a mandatory executed gate.
"""
from pathlib import Path
import re,sys

def generate(source):
    start=source.index('        ulong block=')
    end=source.index('\n    }\n    sum0=',start)
    body=source[start:end]
    assert body.count('sum0+=x[j]*weights[j]')==1
    reductions=source[end+len('\n    }\n'):source.index('\n}\n)METAL',end)]
    pieces=[]
    for pair in range(2):
        block=re.sub(r'\bsum([01])\b',lambda m:f'acc{pair}{m[1]}',body)
        pieces.append('        {\n            ulong column=first+'+str(pair*4)+';\n            if(column<p.columns) {\n'+block+'\n            }\n        }')
    output=[]
    for pair in range(2):
        block=re.sub(r'\bsum([01])\b',lambda m:f'acc{pair}{m[1]}',reductions)
        output.append('    {\n        ulong column=first+'+str(pair*4)+';\n        if(column<p.columns) {\n'+block+'\n        }\n    }')
    prefix=source[:source.index('    ulong column=')]
    prefix=prefix.replace('kCoalescedBodyBlockShader','kCoalescedEightBodyShader').replace('bf16_body_coalesced_block(','bf16_body_coalesced_eight(')
    prefix=prefix.replace('// Private exact BF16 block decode: four outputs by128 K, coalesced byte planes.',
        '// Private eight-output scheduling: two lexical original four-output blocks.')
    result=prefix+'''    ulong first=ulong(group)*(threads/32)*8+simd*8;if(first>=p.columns)return;
    float4 acc00=0.0f,acc01=0.0f,acc10=0.0f,acc11=0.0f;
    for(ulong k=ulong(lane)*4;k<p.inner;k+=128) {
        float4 x=*(device const float4*)(a+k),y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0.0f);
'''+ '\n'.join(pieces)+'\n    }\n'+'\n'.join(output)+source[source.index('\n}\n)METAL',end):]
    assert set(re.findall(r'\bacc\d+\b',result))=={'acc00','acc01','acc10','acc11'}
    assert not re.search(r'\bsum\d+\b',result)
    return result

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    src=root/'src/backends/metal/coalesced_body_block_shader.h'
    target=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('coalesced_eight_body_shader.h')
    target.write_bytes(generate(src.read_text()).encode())
