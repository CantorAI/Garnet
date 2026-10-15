"""Preserve terminal native prototype checks, including failed configure/build/tests."""
from pathlib import Path
import hashlib,io,json,sys,tarfile

root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();trial=sys.argv[3] if len(sys.argv)>3 else 'mlp-prefill-native-checks-20261015'
folder=root/'logs'/trial;status=json.loads((folder/'exit.json').read_text());assert isinstance(status['exit_code'],int)
files={};omitted={}
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def add(p):
    rel=p.relative_to(root).as_posix()
    if p.suffix in ('.json','.py','.cpp','.h','.mm','.cmake','.log','.txt','.make'):
        files['evidence/'+rel]=p
    else:omitted[rel]=dict(bytes=p.stat().st_size,sha256=digest(p))
for base in (folder,root/'logs/mlp-prefill-native-original-source-20261015',root/'mlp-prefill-native-staging'):
    for p in base.rglob('*'):
        if p.is_file():add(p)
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for p in (root/'models'/model).iterdir():
        if p.is_file() and p.name in ('config.json','tokenizer.json','model.safetensors'):add(p)
    assets=root/'logs/screened-head-after-vision-20261015/jobs'/(model+'-assets')
    for p in assets.rglob('*'):
        if p.is_file():add(p)
for fixture in (folder/'jobs/full-grid').glob('*-fixture.json'):
    p=Path(json.loads(fixture.read_text())['patch_file']).resolve()
    assert p.is_relative_to(root)
    add(p)
manifest=json.loads((root/'mlp-prefill-native-staging/manifest.json').read_text())
for name,sha in manifest.items():
    p=root/'source'/name;assert digest(p)==sha;add(p)
closure=json.loads((root/'logs/coalesced-body-native-20261015/source-identities.json').read_text())
for name,sha in closure.items():
    p=root/'source'/name
    if name not in manifest:assert digest(p)==sha,name
    add(p)
meta=folder/'jobs/metadata.json'
if meta.exists():
    for name,sha in json.loads(meta.read_text()).get('source_sha256',{}).items():
        p=root/'source'/name;assert digest(p)==sha;add(p)
for build in ('build-coalesced-bf16-body','build-mlp-prefill-graph'):
    for name in ('libgarnet.dylib','libgarnet_metal.a','garnet_mlp_prefill_graph_test'):
        p=root/build/name
        if p.is_file():omitted[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=digest(p))
for name in ('archive_mlp_prefill_native_checks.py','verify_mlp_prefill_native_checks.py',
             'verify_coalesced_body_native.py','verify_interleaved_body_native.py',
             'verify_wide_batch_prefill.py','verify_interleaved_head.py',
             'verify_small_seed.py','verify_serial_vision_native.py'):
    p=root/'source/test2026/metal'/name
    if p.is_file():add(p)
proof=dict(files={},omitted=omitted,jobs={trial:status},scope='Terminal native prototype validation evidence only; no complete inference or speed claim.',full_goal_met=False)
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=status)))
