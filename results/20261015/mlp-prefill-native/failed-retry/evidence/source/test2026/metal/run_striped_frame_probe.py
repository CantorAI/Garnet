"""CPU-only exact-coordinate resize and frozen fresh-preparation ABBA."""
from pathlib import Path
source=Path(__file__).with_name('run_compact_frame_probe.py').read_text()
source=source.replace('from frame_preprocessing_compact import prepare_frame as candidate','from frame_preprocessing_striped import StripedFramePreparer\npreparer=StripedFramePreparer()\ncandidate=preparer.prepare')
source=source.replace('test_compact_frame_preprocessing.py','test_striped_frame_preprocessing.py').replace('frame_preprocessing_compact.py','frame_preprocessing_striped.py')
source=source.replace('return dict(batch=batch,candidate=policy,','assert not policy or preparer.last_route==\'parallel\'\n  return dict(route=preparer.last_route if policy else \'control\',batch=batch,candidate=policy,')
source=source.replace("paths=[Path(__file__),", "paths=[s/'run_compact_frame_probe.py',s/'test_compact_frame_preprocessing.py',Path(__file__),")
source=source.replace("(out/'summary.json').write_bytes", "preparer.close()\n(out/'summary.json').write_bytes")
exec(compile(source,__file__,'exec'))
