import os,json,subprocess,numpy as np
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');data=root/'logs/vision-debug-input.f32'
np.random.default_rng(768).normal(size=(1,4,1536)).astype('<f4').tofile(data)
for stage in (141,142,143,144,145,146,147,148):
 env=os.environ.copy();env['GARNET_METAL_VISION_DEBUG_OP']=str(stage)
 subprocess.run(['/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3',str(root/'source/test2026/metal/vision_stage_debug.py'),str(root/'build'),str(root/'source'),str(root/'models/Qwen3.5-0.8B'),str(data),str(stage)],env=env,check=True)
