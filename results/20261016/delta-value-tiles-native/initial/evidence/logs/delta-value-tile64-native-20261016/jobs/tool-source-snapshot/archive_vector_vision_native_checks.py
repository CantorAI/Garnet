"""Archive terminal native vector checks, original/transformed source and selection."""
from pathlib import Path
p=Path(__file__).with_name('archive_shape_selected_two_checks.py');source=p.read_text()
source=source.replace('shape-selected-two-checks-20261015','vector-vision-native-checks-20261016')
source=source.replace('build-shape-selected-two-body','build-vector-vision-width4')
source=source.replace('Fresh28 validation and selected-route primitive','Fresh original28 plus ordered vector native primitive')
needle='proof=dict(files={},';assert source.count(needle)==1
source=source.replace(needle,"""files['evidence/selected-component/vector-vision.metal']=parent/'logs/vector-vision-calibrated-confirm-20261016/jobs/vector-vision.metal'
for name in ('archive_vector_vision_native_checks.py','archive_shape_selected_two_checks.py','verify_vector_vision_native_checks.py'):
    p=Path(__file__).with_name(name);files['evidence/external-archive-tools/'+name]=p
"""+needle)
exec(compile(source,__file__,'exec'))
