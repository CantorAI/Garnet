"""Archive terminal two-kernel component evidence, preserving rejected methods."""
from pathlib import Path
source=Path(__file__).with_name('archive_decode_subsimd.py').read_text().replace("glob('decode-subsimd-probe*')","glob('decode-splitk-*')")
needle="for name in ('make_decode_subsimd_shader.py'"
assert source.count(needle)==1
source=source.replace(needle,"for name in ('make_decode_splitk_shader.py','make_decode_splitk_probe.py','run_decode_splitk_probe.py','launch_decode_splitk_probe.py','archive_decode_splitk.py','make_decode_eight_lane_probe.py','run_decode_eight_lane_probe.py','make_decode_subsimd_shader.py'")
exec(compile(source,__file__,'exec'))
