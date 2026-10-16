"""Archive terminal selective full30, original/transformed source, exact external drivers and assets."""
from pathlib import Path
p=Path(__file__).with_name('archive_shape_selected_two_checks.py');source=p.read_text()
source=source.replace('shape-selected-two-checks-20261015','ndarray-selective-native-packed-checks-20261016')
source=source.replace('build-shape-selected-two-body','build-ndarray-selective-prefill-packedguard')
source=source.replace('Fresh28 validation and selected-route primitive','Fresh original28 validation plus selective cache/route/abort primitive stages')
needle='proof=dict(files={},';assert source.count(needle)==1
source=source.replace(needle,"for name in ('archive_ndarray_selective_native_checks.py','archive_shape_selected_two_checks.py','verify_ndarray_selective_native_checks.py','freeze_ndarray_selective_native_auditor.py'):\n    p=Path(__file__).with_name(name);files['evidence/external-archive-tools/'+name]=p\n"+needle)
exec(compile(source,__file__,'exec'))
