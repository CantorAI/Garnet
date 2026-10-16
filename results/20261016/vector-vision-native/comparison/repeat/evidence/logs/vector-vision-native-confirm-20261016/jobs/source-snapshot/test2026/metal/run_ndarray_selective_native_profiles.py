"""All six original native complete-output ABBA profiles after fresh selective full30 audit."""
from pathlib import Path
p=Path(__file__).with_name('run_shape_selected_two_profiles.py');source=p.read_text()
source=source.replace('shape-selected-two-checks-verification.json','ndarray-selective-native-checks-archive-verification.json')
source=source.replace("audit['validation_stages']==28 and audit['ctests']==7 and audit['coalesced_primitive_cases']==104",
    "audit['validation_stages']==30 and audit['ctests']==7 and audit['coalesced_primitive_cases']==80 and audit['ndarray_cache_limit']==256")
source=source.replace('compare_shape_selected_two_native.py','compare_ndarray_selective_native.py')
exec(compile(source,__file__,'exec'))
