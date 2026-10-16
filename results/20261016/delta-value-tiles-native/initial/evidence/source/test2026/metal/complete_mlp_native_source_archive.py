"""Preserve the original terminal archive; add omitted frozen Metal source bytes."""
from pathlib import Path
import hashlib,io,json,sys,tarfile

root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
original=root/'mlp-prefill-native-20261015.tar.gz'
expected='4c54a571ab781c9635943839c7fb341db27bd18632d1b7d52f329467fb63f852'
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert digest(original)==expected and not target.exists()
native=root/'logs/mlp-prefill-native-20261015'
controller=root/'logs/mlp-prefill-native-and-eight-completed-20261015'
assert json.loads((native/'exit.json').read_text())['exit_code']==0
assert json.loads((controller/'exit.json').read_text())['exit_code']==1
with tarfile.open(original) as tar:
    members=tar.getmembers();assert all(m.isfile() for m in members)
    data={m.name:tar.extractfile(m).read() for m in members}
assert len(data)==len(members)
original_manifest=data.pop('evidence/manifest.json')
proof=json.loads(original_manifest)
for name,info in proof['files'].items():
    assert len(data[name])==info['bytes'] and hashlib.sha256(data[name]).hexdigest()==info['sha256']
identities=json.loads(data['evidence/logs/mlp-prefill-native-20261015/source-identities.json'])
added=[]
for name,sha in identities.items():
    source='evidence/source/'+name
    snapshot='evidence/logs/mlp-prefill-native-20261015/source-snapshot/'+name
    if source in data:
        assert snapshot in data and hashlib.sha256(data[source]).hexdigest()==sha
        assert hashlib.sha256(data[snapshot]).hexdigest()==sha
        continue
    assert name.endswith('.metal') and snapshot not in data,name
    raw=(native/'source-snapshot'/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==sha and digest(root/'source'/name)==sha
    for archive_name in (source,snapshot):
        omitted=archive_name.removeprefix('evidence/')
        assert proof['omitted'][omitted]==dict(bytes=len(raw),sha256=sha)
        data[archive_name]=raw
        del proof['omitted'][omitted]
    added.append(name)
assert len(added)==13
data['evidence/archival-repair/original-manifest.json']=original_manifest
for path in controller.rglob('*'):
    if path.is_file():data['evidence/'+path.relative_to(root).as_posix()]=path.read_bytes()
data['evidence/archival-repair/complete_mlp_native_source_archive.py']=Path(__file__).read_bytes()
proof['archival_repair']=dict(original_archive_sha256=expected,
    original_manifest_sha256=hashlib.sha256(original_manifest).hexdigest(),
    added_source_names=added,
    reason='Original suffix whitelist omitted .metal bytes despite recording their hashes. Add only exact frozen snapshots; numerical evidence unchanged.')
proof['files']={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in sorted(data.items())}
with tarfile.open(target,'w:gz') as tar:
    for name,raw in sorted(data.items()):
        item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode()
    item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(data),sha256=digest(target),added_sources=added,full_goal_met=False)))
