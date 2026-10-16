"""Wait exact validation/audit controller, then native cohorts, archive, audit, decision."""
from pathlib import Path
p=Path(__file__).with_name('continue_ndarray_selective_native.py');source=p.read_text()
source=source.replace('ndarray-selective-native-packed-investigation-20261016','vector-vision-native-investigation-20261016')
source=source.replace('ndarray-selective-native','vector-vision-native')
for old,new in [('continue_ndarray_selective_native.py','continue_vector_vision_native.py'),
 ('launch_ndarray_selective_native_profiles.py','launch_vector_vision_native_profiles.py'),
 ('run_ndarray_selective_native_profiles.py','run_vector_vision_native_profiles.py'),
 ('compare_ndarray_selective_native.py','compare_vector_vision_native.py'),
 ('archive_ndarray_selective_native_profiles.py','archive_vector_vision_native_profiles.py'),
 ('verify_ndarray_selective_native_profiles.py','verify_vector_vision_native_profiles.py'),
 ('analyze_ndarray_selective_native_results.py','analyze_vector_vision_native_results.py'),
 ('finalize_ndarray_selective_native_checks.py','finalize_vector_vision_native_checks.py')]:source=source.replace(old,new)
exec(compile(source,__file__,'exec'))
