"""Exclusive native comparison only after the frozen fresh30 audit passes."""
from pathlib import Path
source=Path(__file__).with_name('launch_shape_selected_two_profiles.py').read_text()
source=source.replace('shape-selected-two-checks-verification.json','ndarray-native-checks-verification.json').replace('shape-selected-two-checks-20261015.tar.gz','ndarray-native-checks-repaired-20261015.tar.gz')
source=source.replace("audit['validation_stages']==28 and audit['ctests']==7 and audit['coalesced_primitive_cases']==104","audit['validation_stages']==30 and audit['ctests']==7 and audit['coalesced_primitive_cases']==80 and audit['ndarray_cache_limit']==256")
source=source.replace('run_shape_selected_two_profiles.py','run_ndarray_native_profiles.py')
exec(compile(source,__file__,'exec'))
