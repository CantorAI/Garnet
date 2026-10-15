"""Verify and back up every replaced source before installing the isolated prototype."""
from pathlib import Path
import hashlib,json,shutil,sys,tarfile

root=Path(sys.argv[1]).resolve();stage=root/'mlp-prefill-native-staging';source=root/'source'
for name in ('coalesced-body-vllm-confirm-20261015','dependent-mlp-graph-20261015','stage-rounded-mlp-graph-20261015'):
    assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
gates=json.loads((stage/'component-gates.json').read_text());assert gates['native_comparison_checks']==48 and not gates['full_goal_met']
original=json.loads((stage/'original.json').read_text());manifest=json.loads((stage/'manifest.json').read_text())
for name,sha in original.items():
    p=(source/name).resolve();assert p.is_relative_to(source.resolve()) and hashlib.sha256(p.read_bytes()).hexdigest()==sha,name
for name in set(manifest)-set(original):assert not (source/name).exists(),name
backup=root/'logs/mlp-prefill-native-original-source-20261015';backup.mkdir(exist_ok=False)
for name in original:
    p=backup/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,p)
with tarfile.open(stage/'source.tar.gz') as archive:
    members=archive.getmembers();assert {m.name for m in members}==set(manifest)
    assert all(m.isfile() and (source/m.name).resolve().is_relative_to(source.resolve()) for m in members)
    archive.extractall(source,filter='data')
assert all(hashlib.sha256((source/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
print(json.dumps(dict(installed_files=len(manifest),original_files=len(original),backup=str(backup))))
