"""Independent-source complete RGB/patch parity, including eligible large images."""
from pathlib import Path
from frame_preprocessing_striped import StripedFramePreparer
preparer=StripedFramePreparer()
source=Path(__file__).with_name('test_compact_frame_preprocessing.py').read_text()
source=source.replace('from frame_preprocessing_compact import prepare_frame as candidate','candidate=preparer.prepare')
source=source.replace('((1,1),(31,47),(64,96),(513,257))','((1,1),(31,47),(64,96),(513,257),(1372,1818))')
source=source.replace('checks.append(dict(size=size,','checks.append(dict(route=preparer.last_route,size=size,')
source=source.replace('report=dict(scope=__doc__,','assert preparer.parallel_calls>0 and preparer.fallback_calls>0\nreport=dict(parallel_calls=preparer.parallel_calls,fallback_calls=preparer.fallback_calls,scope=__doc__,')
try:exec(compile(source,__file__,'exec'))
finally:preparer.close()
