"""Exclusive two/four-part FP32 decode; includes both kernels in wall/GPU span."""
from pathlib import Path
source=Path(__file__).with_name('run_decode_eight_lane_probe.py').read_text().replace('make_decode_eight_lane_probe.py','make_decode_splitk_probe.py').replace('make_decode_eight_lane_shader.py','make_decode_splitk_shader.py').replace('decode_eight_lane_gpu','decode_splitk_gpu')
source=source.replace('Eight-lane','Split-K').replace('eight-lane','split-K')
source=source.replace("paths=[s/'make_decode_subsimd_probe.py',", "paths=[s/'make_decode_eight_lane_probe.py',s/'run_decode_eight_lane_probe.py',s/'make_decode_subsimd_probe.py',")
exec(compile(source,__file__,'exec'))
