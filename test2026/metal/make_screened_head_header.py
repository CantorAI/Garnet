"""Embed exactly the guarded component's strict-math screen functions."""
from pathlib import Path
from make_guarded_screened_head import generate_shader
folder=Path(__file__).resolve().parent;root=folder.parents[1]
shader='#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+generate_shader((folder/'screened_head.metal').read_text())
shader=shader.replace('using namespace metal;','using namespace metal;\n#ifndef GARNET_SCREENED_HEAD_SEED_ROWS\n#define GARNET_SCREENED_HEAD_SEED_ROWS 8192\n#endif').replace('min(ulong(8192),p.columns)','min(ulong(GARNET_SCREENED_HEAD_SEED_ROWS),p.columns)')
shader=shader.replace('struct BF16Params {','#ifndef GARNET_SCREENED_HEAD_INTERLEAVED\n#define GARNET_SCREENED_HEAD_INTERLEAVED 0\n#endif\nstruct BF16Params {')
start=shader.index('        char4 b0=');end=shader.index('\n    }',start);ordinary=shader[start:end]
interleaved='''        uint4 packed=*(device const uint4*)(approximate+column*p.inner+k*4);
        for(uint j=0;j<4;++j){float4 w=float4(as_type<char4>(packed[j]));sum0=fma(x[j],w,sum0);if(p.rows==2)sum1=fma(y[j],w,sum1);}
'''
shader=shader[:start]+'#if GARNET_SCREENED_HEAD_INTERLEAVED\n'+interleaved+'#else\n'+ordinary+'\n#endif'+shader[end:]
(root/'src/backends/metal/screened_head_shader.h').write_bytes(('#pragma once\nstatic constexpr const char* kScreenedHeadShader=R"SCREENED(\n'+shader+')SCREENED";\n').encode())
