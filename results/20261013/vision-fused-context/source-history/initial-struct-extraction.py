"""Extract the selected vision kernels and fuse their softmax/value contraction."""
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
start = source.index('struct VisionParams {')
params = source[start:source.index(';', start) + 1]
start = source.index('kernel void vision_softmax(')
softmax = source[start:source.index('// Input/output matrices', start)]
start = source.index('kernel void vision_op(')
context = source[start:source.index(')METAL"]', start)]
fused = softmax.replace('kernel void vision_softmax(', 'kernel void vision_softmax_context(', 1)
old = '    for(ulong key=tid;key<p.tokens;key+=128)out[row*p.oStride+head*p.tokens+key]=scratch[key]/total;'
assert old in fused
fused = fused.replace(old, '''    for(ulong key=tid;key<p.tokens;key+=128)scratch[key]/=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(ulong d=tid;d<dim;d+=128) {
        float result=0;
        for(ulong key=0;key<p.tokens;++key)result+=scratch[key]*a[(batchBase+key)*p.aStride+2*p.hidden+head*dim+d];
        out[row*p.oStride+head*dim+d]=result;
    }''')
Path(sys.argv[2]).write_bytes(('#include <metal_stdlib>\nusing namespace metal;\n' + params + '\n' + softmax + context + fused).encode())
