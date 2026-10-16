"""Retain terminal eight-lane evidence, including numerical rejection if any."""
from pathlib import Path
source=Path(__file__).with_name('archive_decode_subsimd.py').read_text().replace("glob('decode-subsimd-probe*')","glob('decode-eight-lane-*')")
needle="for name in ('make_decode_subsimd_shader.py'"
assert source.count(needle)==1
source=source.replace(needle,"for name in ('make_decode_eight_lane_shader.py','make_decode_eight_lane_probe.py','run_decode_eight_lane_probe.py','queue_decode_eight_lane_probe.py','archive_decode_eight_lane.py','make_decode_subsimd_shader.py'")
exec(compile(source,__file__,'exec'))
