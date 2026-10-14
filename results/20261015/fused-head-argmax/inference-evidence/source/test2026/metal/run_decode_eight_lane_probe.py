"""Exclusive eight-lane FP32 decode component; no full-model speed claim."""
from pathlib import Path
source=Path(__file__).with_name('run_decode_subsimd_probe.py').read_text()
source=source.replace('make_decode_subsimd_probe.py','make_decode_eight_lane_probe.py').replace('make_decode_subsimd_shader.py','make_decode_eight_lane_shader.py').replace('decode_subsimd_gpu','decode_eight_lane_gpu')
source=source.replace('Sub-SIMD','Eight-lane').replace('sub-SIMD','eight-lane')
source=source.replace("paths=[native,", "paths=[s/'make_decode_subsimd_probe.py',s/'run_decode_subsimd_probe.py',native,")
exec(compile(source,__file__,'exec'))
