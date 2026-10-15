"""Packed derived INT4 index; original seed/refinement and private-greedy semantics."""
from pathlib import Path
import runpy
def generate_shader(source):
    source=source.replace('device const char* approximate','device const uchar* approximate')
    begin=source.index('        char4 b0=');end=source.index('\n    }',begin)
    source=source[:begin]+'''        ushort b0=*(device const ushort*)(approximate+(column*p.inner+k)/2),b1=0,b2=0,b3=0;
        if(valid.y)b1=*(device const ushort*)(approximate+((column+1)*p.inner+k)/2);
        if(valid.z)b2=*(device const ushort*)(approximate+((column+2)*p.inner+k)/2);
        if(valid.w)b3=*(device const ushort*)(approximate+((column+3)*p.inner+k)/2);
        for(uint j=0;j<4;++j){
            float4 w=float4(int((b0>>(j*4))&15)-8,int((b1>>(j*4))&15)-8,int((b2>>(j*4))&15)-8,int((b3>>(j*4))&15)-8);
            sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
        }'''+source[end:]
    return source
def generate_harness(source):
    generate=runpy.run_path(str(Path(__file__).with_name('make_screened_head_harness.py')))['generate']
    source=generate(source)
    assert source.count('approximate.length!=columns*inner)')==1
    return source.replace('approximate.length!=columns*inner)','approximate.length!=columns*inner/2)').replace('double((columns-seedRows)*inner+2*reads)','double((columns-seedRows)*inner/2+2*reads)')
