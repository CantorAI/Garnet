"""All six native complete-output ABBA profiles after fresh30 independent audit."""
from pathlib import Path
p=Path(__file__).with_name('run_shape_selected_two_profiles.py');source=p.read_text()
source=source.replace('shape-selected-two-checks-verification.json','vector-vision-native-checks-archive-verification.json')
source=source.replace("audit['validation_stages']==28 and audit['ctests']==7 and audit['coalesced_primitive_cases']==104","audit['validation_stages']==30 and audit['ctests']==7 and audit['coalesced_primitive_cases']==80 and audit['vector_primitive_checks']==96")
source=source.replace('compare_shape_selected_two_native.py','compare_vector_vision_native.py')
exec(compile(source,__file__,'exec'))
