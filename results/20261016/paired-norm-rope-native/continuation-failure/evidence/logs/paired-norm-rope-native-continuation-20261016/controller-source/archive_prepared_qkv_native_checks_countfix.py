"""Preserve original failed archive plus unchanged passing25 and fresh8 completion evidence."""
from pathlib import Path
p=Path(__file__).with_name('archive_prepared_qkv_native_checks.py');source=p.read_text()
source=source.replace("trial='prepared-qkv-native-checks-20261016'","trial='prepared-qkv-native-checks-countfix-20261016'")
needle='proof=dict(files={},omitted=omitted,jobs={trial:status},';assert source.count(needle)==1
source=source.replace(needle,"""prior_archive=Path(prep['parent'])/'prepared-qkv-native-checks-20261016.tar.gz'
assert digest(prior_archive)=='f1c7ba015ebaf86c6982bd0284c03d2f6116fb2d6e12fd05a3cdcc31d5c735eb'
prior_payloads={}
with tarfile.open(prior_archive) as previous:
    prior_proof=json.load(previous.extractfile('evidence/manifest.json'))
    for name,info in prior_proof['files'].items():
        raw=previous.extractfile(name).read();assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256'];prior_payloads[name]=raw
    for name,info in prior_proof['omitted'].items():
        if name in omitted:assert omitted[name]==info
        omitted[name]=info
for name in ('archive_prepared_qkv_native_checks_countfix.py','verify_prepared_qkv_native_checks_countfix.py','finalize_prepared_qkv_native_checks_countfix.py','resume_prepared_qkv_native_checks.py','launch_resume_prepared_qkv_native_checks.py'):
    add(Path(__file__).with_name(name),'external-archive-tools/'+name)
"""+"proof=dict(files={},omitted=omitted,jobs={**prior_proof['jobs'],trial:status},prior_archive_sha256=digest(prior_archive),")
source=source.replace('Fresh original30, actual linked QKV primitive and original-bound trained graph/session reference; terminal failure or success preserved, no performance claim.','Original failed-count archive retained; unchanged passing25 reused plus8 fresh original-bound checks, no model performance claim.')
old="""for name,p in sorted(files.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))""";assert source.count(old)==1
source=source.replace(old,"""for name in sorted(set(files)|set(prior_payloads)):
  raw=files[name].read_bytes() if name in files else prior_payloads[name]
  if name in files and name in prior_payloads:assert raw==prior_payloads[name]
  proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))""")
exec(compile(source,__file__,'exec'))
