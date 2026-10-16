"""Extend frozen33 evidence with complete native outputs and every declared source snapshot."""
from pathlib import Path
p=Path(__file__).with_name('archive_ndarray_native_profiles.py');source=p.read_text()
source=source.replace("'ndarray-native-20261015','ndarray-native-repaired-20261015','ndarray-native-confirm-20261015'","'prepared-qkv-native-20261016','prepared-qkv-native-confirm-20261016'").replace("else 'ndarray-native-20261015'","else 'prepared-qkv-native-20261016'")
source=source.replace("validation=root/'ndarray-native-checks-repaired-20261015.tar.gz'","validation=Path(json.loads((root/'preparation.json').read_text())['parent'])/'prepared-qkv-native-checks-20261016.tar.gz'")
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
source=source.replace('Fresh30 and native full-model row80 NDArray ABBA','Fresh33 and native prepared text QKV versus vector d52 ABBA')
exec(compile(source,__file__,'exec'))
