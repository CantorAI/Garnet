"""Archive and audit completion of25 reused plus8 fresh checks; no model-performance launch."""
from pathlib import Path
import hashlib,json
p=Path(__file__).with_name('finalize_prepared_qkv_native_checks.py');parent=p.parents[3]
old=json.loads((parent/'logs/prepared-qkv-native-finalize-20261016/metadata.json').read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest()==old['source_sha256'][p.as_posix()]
source=p.read_text().replace('prepared-qkv-native-checks-20261016','prepared-qkv-native-checks-countfix-20261016')
source=source.replace('archive_prepared_qkv_native_checks.py','archive_prepared_qkv_native_checks_countfix.py').replace('verify_prepared_qkv_native_checks.py','verify_prepared_qkv_native_checks_countfix.py')
exec(compile(source,__file__,'exec'))
