"""Extract actual FP32 chunk attention and add a GPU-verified prefix lookup."""
from pathlib import Path
import sys
source=Path(sys.argv[1]).read_text()
start=source.index('struct ChunkAttentionParams {')
control=source[start:source.index('struct ScoresParams {',start)]
# Resolve the production experiment's MSL conditional to its control branch.
lines=[];inside=False;keep=True
for line in control.splitlines(keepends=True):
    if line.strip()=='#ifdef GARNET_ATTENTION_PREFIX':
        assert not inside;inside=True;keep=False
    elif inside and line.strip()=='#else':keep=True
    elif inside and line.strip()=='#endif':inside=False;keep=True
    elif keep:lines.append(line)
assert not inside
control=''.join(lines)
candidate=control[control.index('kernel void attention_chunk('):].replace('kernel void attention_chunk(', 'kernel void attention_chunk_prefix(',1)
old='    threadgroup float partial[4];'
assert candidate.count(old)==1
candidate=candidate.replace(old,'''    threadgroup float partial[4];
    threadgroup int contiguousPrefix;
    if(tid==0) {
        float first=slots[row*p.sStride];
        int valid=isfinite(first)&&first>=0.0f&&first==floor(first);
        for(ulong t=0;t<=token;++t)if(active[row*p.aStride+t]==0.0f||slots[row*p.sStride+t]!=first+float(t))valid=0;
        contiguousPrefix=valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);''')
old='        int latest=-1;for(uint t=0;t<=token;++t)if(active[row*p.aStride+t]!=0.0f&&slots[row*p.sStride+t]==float(s))latest=int(t);'
assert candidate.count(old)==1
candidate=candidate.replace(old,'''        int latest=-1;
        if(contiguousPrefix) {
            float first=slots[row*p.sStride];
            if(float(s)>=first)latest=int(float(s)-first);
        } else for(uint t=0;t<=token;++t)if(active[row*p.aStride+t]!=0.0f&&slots[row*p.sStride+t]==float(s))latest=int(t);''')
Path(sys.argv[2]).write_bytes(('#include <metal_stdlib>\nusing namespace metal;\n'+control+candidate).encode())
