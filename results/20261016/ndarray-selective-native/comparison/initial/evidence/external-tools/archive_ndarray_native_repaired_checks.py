"""Freeze repaired fresh30 validation, including the exact external driver sources."""
from pathlib import Path
p=Path(__file__).with_name('archive_shape_selected_two_checks.py');source=p.read_text()
source=source.replace('shape-selected-two-checks-20261015','ndarray-native-checks-repaired-20261015').replace('build-shape-selected-two-body','build-ndarray-prefill-shape80-guardrepair')
source=source.replace('Fresh28 validation and selected-route primitive','Fresh original28 validation plus exact owned NDArray primitive stages')
needle='proof=dict(files={},';assert source.count(needle)==1
source=source.replace(needle,"for name in ('archive_ndarray_native_repaired_checks.py','archive_shape_selected_two_checks.py','verify_ndarray_native_checks.py'):\n    p=Path(__file__).with_name(name);files['evidence/external-archive-tools/'+name]=p\n"+needle)
exec(compile(source,__file__,'exec'))
