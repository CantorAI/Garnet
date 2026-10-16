"""Install three guarded repairs only after the original stopped trial is archived."""
from pathlib import Path
import hashlib,json,sys,tarfile

root=Path(sys.argv[1]).resolve();stage=root/'mlp-guard-repair-staging'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((root/'logs/mlp-prefill-native-checks-20261015/exit.json').read_text())['exit_code']==1
with (root/'mlp-prefill-native-checks-20261015.tar.gz').open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()=='eada52786b8f2072c83c810d542823ff58ad9519f07c18b6fa526b5f989465fc'
old=json.loads((stage/'previous-manifest.json').read_text());new=json.loads((stage/'manifest.json').read_text());changes=json.loads((stage/'changes.json').read_text())
assert len(old)==17 and len(new)==18 and len(changes)==3
assert old==json.loads((root/'mlp-prefill-native-staging/manifest.json').read_text())
assert all(digest(root/'source'/name)==sha for name,sha in old.items())
assert all(digest(root/'source'/name)==values['before'] for name,values in changes.items())
diag=json.loads((root/'logs/mlp-prefill-native-checks-20261015/guard-process-diagnostic.json').read_text())
assert all(not v['inference_or_gpu_library_names'] and v['lsof_exit_code']==0 for v in diag['processes'])
scripts=[s for v in diag['processes'] for a in v['ancestor_chain'] for s in a['known_scripts']]
assert scripts and all(s['sha256']=='115a1d9dd0ae7181af4876e57bda55b9948d31c1febf1a12d40b150bce0c622e' and not s['inference_markers_after_exact_budget_literal_removed'] for s in scripts)
backup=root/'logs/mlp-prefill-guard-repair-20261015';backup.mkdir(exist_ok=False)
for name in changes:
    p=backup/'original'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((root/'source'/name).read_bytes())
(backup/'previous-manifest.json').write_bytes((stage/'previous-manifest.json').read_bytes())
with tarfile.open(stage/'repair.tar.gz') as tar:
    members=tar.getmembers();assert {m.name for m in members}==set(changes)
    for member in members:
        assert member.isfile();p=(root/'source'/member.name).resolve();assert p.is_relative_to(root/'source')
        raw=tar.extractfile(member).read();assert hashlib.sha256(raw).hexdigest()==changes[member.name]['after'];p.write_bytes(raw)
assert all(digest(root/'source'/name)==sha for name,sha in new.items())
(root/'mlp-prefill-native-staging/manifest.json').write_bytes((stage/'manifest.json').read_bytes())
report=dict(changes=changes,scope='Guard service fingerprint and fresh build path only; native numerical sources and tolerances unchanged',original_trial_archive_sha256='eada52786b8f2072c83c810d542823ff58ad9519f07c18b6fa526b5f989465fc',manifest_files=18)
(backup/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
