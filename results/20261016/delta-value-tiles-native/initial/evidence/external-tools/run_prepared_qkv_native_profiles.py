"""All six native complete-output profiles after full33 terminal archive and audit."""
from pathlib import Path
p=Path(__file__).with_name('run_shape_selected_two_profiles.py');source=p.read_text()
source=source.replace('shape-selected-two-checks-verification.json','prepared-qkv-native-checks-archive-verification.json')
source=source.replace("audit['validation_stages']==28 and audit['ctests']==7 and audit['coalesced_primitive_cases']==104","audit['validation_stages']==33 and audit['ctests']==7 and audit['coalesced_primitive_cases']==80 and audit['vector_primitive_checks']==96 and audit['prepared_qkv_primitive_cases']==24 and audit['prepared_qkv_trained_steps']==24")
source=source.replace('compare_shape_selected_two_native.py','compare_prepared_qkv_native.py')
exec(compile(source,__file__,'exec'))
