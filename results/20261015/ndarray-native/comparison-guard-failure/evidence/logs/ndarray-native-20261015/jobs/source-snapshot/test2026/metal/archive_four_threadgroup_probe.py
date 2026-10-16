"""Freeze terminal threadgroup probes and original CPU packing admission."""
from pathlib import Path
p=Path(__file__).with_name('archive_two_matrix_probe.py');wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns);source=ns['source']
source=source.replace("['two-matrix-probe-20261015']","['four-threadgroup-probe-20261015']")
exec(compile(source,__file__,'exec'))
