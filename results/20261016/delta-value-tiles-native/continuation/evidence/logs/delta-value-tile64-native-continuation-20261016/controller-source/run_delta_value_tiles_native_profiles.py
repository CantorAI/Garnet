"""All six complete-output native profiles after fresh32 archive/audit."""
from pathlib import Path
p=Path(__file__).with_name('run_shape_selected_two_profiles.py');source=p.read_text()
source=source.replace('shape-selected-two-checks-verification.json','delta-value-tile64-native-checks-archive-verification.json')
source=source.replace("audit['validation_stages']==28 and audit['ctests']==7 and audit['coalesced_primitive_cases']==104","audit['validation_stages']==32 and audit['ctests']==7 and audit['coalesced_primitive_cases']==80 and audit['vector_primitive_checks']==96 and audit['delta_value_tile64_primitive_cases']==162 and audit['delta_value_tile64_trained_steps']==48")
source=source.replace('compare_shape_selected_two_native.py','compare_delta_value_tiles_native.py');exec(compile(source,__file__,'exec'))
