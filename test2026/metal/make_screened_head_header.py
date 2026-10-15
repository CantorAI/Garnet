"""Embed exactly the guarded component's strict-math screen functions."""
from pathlib import Path
from make_guarded_screened_head import generate_shader
folder=Path(__file__).resolve().parent;root=folder.parents[1]
shader='#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+generate_shader((folder/'screened_head.metal').read_text())
(root/'src/backends/metal/screened_head_shader.h').write_bytes(('#pragma once\nstatic constexpr const char* kScreenedHeadShader=R"SCREENED(\n'+shader+')SCREENED";\n').encode())
