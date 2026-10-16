"""Freeze terminal native NDArray checks and all generated/original source identities."""
from pathlib import Path
p=Path(__file__).with_name('archive_shape_selected_two_checks.py');source=p.read_text()
source=source.replace('shape-selected-two-checks-20261015','ndarray-native-checks-20261015').replace('build-shape-selected-two-body','build-ndarray-prefill-shape80')
source=source.replace('Fresh28 validation and selected-route primitive','Fresh original28 validation plus two actual-library NDArray primitive stages')
exec(compile(source,__file__,'exec'))
