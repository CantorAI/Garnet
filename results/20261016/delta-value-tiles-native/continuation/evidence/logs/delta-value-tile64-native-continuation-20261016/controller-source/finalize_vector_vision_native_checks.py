"""Watch exact native check PID; freeze terminal evidence and independently audit."""
from pathlib import Path
import sys
p=Path(__file__).with_name('finalize_ndarray_selective_native_checks.py');source=p.read_text()
source=source.replace('ndarray-selective-native-packed-investigation-20261016','vector-vision-native-investigation-20261016')
source=source.replace('ndarray-selective-native-packed-checks-20261016','vector-vision-native-checks-20261016')
source=source.replace('archive_ndarray_selective_native_checks.py','archive_vector_vision_native_checks.py')
source=source.replace('verify_ndarray_selective_native_checks.py','verify_vector_vision_native_checks.py')
source=source.replace('launch_ndarray_selective_native_checks.py','launch_vector_vision_native_checks.py')
source=source.replace('ndarray-selective-native-checks-verification.json','vector-vision-native-checks-verification.json')
exec(compile(source,__file__,'exec'))
