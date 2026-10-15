"""Embed the independently measured candidate without changing its shader body."""
from pathlib import Path
from make_parallel_vision_context import generate
folder=Path(__file__).resolve().parent;root=folder.parents[1]
shader,_=generate((root/'src/backends/metal/metal_context.mm').read_text(),(folder/'vision_fused_context_gpu.mm').read_text())
candidate=shader[shader.index('kernel void vision_parallel_context('):]
(root/'src/backends/metal/parallel_vision_context_shader.h').write_bytes(('#pragma once\nnamespace Garnet {\ninline constexpr const char* kParallelVisionContextShader=R"PARVISION(\n'+candidate+')PARVISION";\n}\n').encode())
