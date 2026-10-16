"""Freeze original full30 auditor with only selective source/primitive/driver identities changed."""
from pathlib import Path
import ast,hashlib,sys
p=Path(__file__).with_name('verify_ndarray_native_checks.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='9ec11306d6dadab5197c521f3f6a8b8ab6a02c0d3348cd35df251ce290059404'
source=p.read_text()
source=source.replace('ndarray-native-checks-repaired-20261015','ndarray-selective-native-packed-checks-20261016')
source=source.replace('build-ndarray-prefill-shape80-guardrepair','build-ndarray-selective-prefill-packedguard')
source=source.replace('make_ndarray_native.py','make_ndarray_selective_native.py')
old="{'run_ndarray_native_checks_repaired.py','run_ndarray_native_checks.py','launch_ndarray_native_checks_repaired.py'}"
assert source.count(old)==1
source=source.replace(old,"{'run_ndarray_selective_native_checks.py','run_ndarray_native_checks.py','run_ndarray_native_checks_repaired.py','launch_ndarray_selective_native_checks.py'}")
source=source.replace("tools/'run_ndarray_native_checks_repaired.py'", "tools/'run_ndarray_selective_native_checks.py'")
source=source.replace('generated.complete30.repaired.executed.py','generated.selective.complete30.executed.py')
source=source.replace('jobs.complete30.repaired.executed.py','jobs.selective.complete30.executed.py')
source=source.replace('NDArray native exact/update/cache256/fallback14/cached-after-cap/abort-recovery/shape-fallback PASS',
    'Selective NDArray native exact/update/cache256/fallback14/cached-after-cap/abort-recovery/shape80-160/native-half PASS')
source=source.replace('Audit fresh30 gates and regenerate native transforms/exact-owned-primitive guard.',
    'Audit original fresh30 gates and regenerate selective transforms/exact-owned-primitive guard.')
ast.parse(source)
Path(sys.argv[1]).write_bytes(source.encode())
