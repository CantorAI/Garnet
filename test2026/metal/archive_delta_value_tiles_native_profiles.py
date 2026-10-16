"""Extend fresh32 evidence with complete native outputs and every source/snapshot byte."""
from pathlib import Path
p=Path(__file__).with_name('archive_ndarray_native_profiles.py');source=p.read_text()
source=source.replace("'ndarray-native-20261015','ndarray-native-repaired-20261015','ndarray-native-confirm-20261015'","'delta-value-tile64-native-20261016','delta-value-tile64-native-confirm-20261016'").replace("else 'ndarray-native-20261015'","else 'delta-value-tile64-native-20261016'")
source=source.replace("validation=root/'ndarray-native-checks-repaired-20261015.tar.gz'","validation=Path(json.loads((root/'delta-value-tile64-native-checks-gate.json').read_text())['checks_archive'])")
needle='hardware={name:subprocess.check_output';assert source.count(needle)==1
source=source.replace(needle,"""for name,sha in identities.items():
    original=root/'source'/name;snapshot=jobs/'source-snapshot'/name
    assert digest(original)==digest(snapshot)==sha
    files['evidence/source/'+name]=original
    files['evidence/'+snapshot.relative_to(root).as_posix()]=snapshot
"""+needle)
source=source.replace('Fresh30 and native full-model row80 NDArray ABBA','Fresh32 and native width64 chunk80 recurrence versus accepted d52 ABBA')
source=source.replace("checks_archive_sha256=audit['archive_sha256']","audit_dependency_sha256=proof['audit_dependency_sha256'],checks_archive_sha256=audit['archive_sha256']")
exec(compile(source,__file__,'exec'))
