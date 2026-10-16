"""Freeze complete native profiles with full30 provenance and selective admission report."""
from pathlib import Path
p=Path(__file__).with_name('archive_ndarray_native_profiles.py');source=p.read_text()
source=source.replace("'ndarray-native-20261015','ndarray-native-repaired-20261015','ndarray-native-confirm-20261015'",
    "'ndarray-selective-native-20261016','ndarray-selective-native-confirm-20261016'")
source=source.replace("else 'ndarray-native-20261015'", "else 'ndarray-selective-native-20261016'")
source=source.replace("validation=root/'ndarray-native-checks-repaired-20261015.tar.gz'",
    "validation=Path(json.loads((root/'preparation.json').read_text())['parent'])/'ndarray-selective-native-packed-checks-20261016.tar.gz'")
source=source.replace('native full-model row80 NDArray ABBA','native full-model selective packed row80/160 NDArray ABBA')
exec(compile(source,__file__,'exec'))
