"""Embed the byte-identical admitted component kernel with its own strict library."""
from pathlib import Path
from make_serial_vision_context import generate
root=Path(__file__).resolve().parents[2];base=root/'test2026/metal'
shader,_=generate((root/'src/backends/metal/metal_context.mm').read_text(),(base/'vision_fused_context_gpu.mm').read_text())
start=shader.index('struct VisionParams {');params=shader[start:shader.index('};',start)+2]
kernel=shader[shader.index('kernel void vision_prefetch8('):]
source='#include <metal_stdlib>\nusing namespace metal;\n'+params+'\n'+kernel
header='// Generated from the audited original-order component.\n#pragma once\ninline constexpr const char* kSerialVisionPrefetchShader=R"GARNET(\n'+source+')GARNET";\n'
(root/'src/backends/metal/serial_vision_prefetch_shader.h').write_bytes(header.encode())
print('Generated original-order strict vision prefetch header')
