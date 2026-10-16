"""Extend frozen34 evidence with complete native outputs and every declared source snapshot."""
from pathlib import Path
p=Path(__file__).with_name('archive_ndarray_native_profiles.py');source=p.read_text()
source=source.replace("'ndarray-native-20261015','ndarray-native-repaired-20261015','ndarray-native-confirm-20261015'","'paired-norm-rope-native-20261016','paired-norm-rope-native-confirm-20261016'").replace("else 'ndarray-native-20261015'","else 'paired-norm-rope-native-20261016'")
source=source.replace("validation=root/'ndarray-native-checks-repaired-20261015.tar.gz'","validation=Path(json.loads((root/'paired-norm-rope-native-checks-gate.json').read_text())['checks_archive'])")
needle="hardware={name:subprocess.check_output"
assert source.count(needle)==1
source=source.replace(needle,"""# The new full source gate includes textual docs and small binary fixtures.
# Preserve every declared source and snapshot byte instead of dropping suffixes.
for name,sha in identities.items():
    original=root/'source'/name;snapshot=jobs/'source-snapshot'/name
    assert digest(original)==digest(snapshot)==sha
    files['evidence/source/'+name]=original
    files['evidence/'+snapshot.relative_to(root).as_posix()]=snapshot
"""+needle)
source=source.replace('Fresh30 and native full-model row80 NDArray ABBA','Fresh34 and native row1/2 paired text/coder norm-RoPE versus vector d52 ABBA')
source=source.replace("checks_archive_sha256=audit['archive_sha256']","audit_dependency_sha256=proof['audit_dependency_sha256'],checks_archive_sha256=audit['archive_sha256']")
exec(compile(source,__file__,'exec'))
