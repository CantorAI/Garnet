"""Wait on exact PID84121, archive/audit terminal validation, then dispatch native ABBA."""
from pathlib import Path
source=Path(__file__).with_name('finalize_shape_selected_two_checks.py').read_text()
source=source.replace('shape-selected-two-checks-20261015','ndarray-native-checks-repaired-20261015')
source=source.replace('archive_shape_selected_two_checks.py','archive_ndarray_native_repaired_checks.py').replace('verify_shape_selected_two_checks.py','verify_ndarray_native_checks.py')
source=source.replace('shape-selected-two-checks-verification.json','ndarray-native-checks-verification.json').replace('launch_shape_selected_two_profiles.py','launch_ndarray_native_profiles.py').replace('shape-selected-two-native-20261015','ndarray-native-20261015')
source=source.replace("assert handle['pid']==42435","assert handle['pid']==84121")
exec(compile(source,__file__,'exec'))
