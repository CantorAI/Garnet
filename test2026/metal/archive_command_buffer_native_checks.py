"""Freeze a terminal separate-tree validation and every compiled source input."""
from pathlib import Path
import hashlib,io,json,sys,tarfile

root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
trial='mlp-command-buffer-native-checks-20261015'
folder=root/'logs'/trial;status=json.loads((folder/'exit.json').read_text())
assert isinstance(status['exit_code'],int) and not target.exists()
preparation=json.loads((root/'preparation.json').read_text())
assert preparation['target']==str(root)
files={};omitted={}
text_suffixes={'.json','.py','.h','.cpp','.mm','.metal','.cmake','.log','.txt','.make'}
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def add(path):
    name=path.relative_to(root).as_posix()
    if path.suffix in text_suffixes:files['evidence/'+name]=path
    else:omitted[name]=dict(bytes=path.stat().st_size,sha256=digest(path))
def tree(path):
    for file in path.rglob('*'):
        if file.is_file():add(file)
add(root/'preparation.json');tree(folder)
for name,sha in preparation['source_sha256'].items():
    path=root/'source'/name
    if path.suffix not in text_suffixes:continue
    assert digest(path)==sha,name
    add(path)
primitive='mlp-command-buffer-chained-20261015'
tree(root/'logs'/primitive)
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for name in ('config.json','tokenizer.json','model.safetensors'):add(root/'models'/model/name)
    tree(root/'logs/screened-head-after-vision-20261015/jobs'/(model+'-assets'))
for fixture in (folder/'jobs/full-grid').glob('*-fixture.json'):
    path=Path(json.loads(fixture.read_text())['patch_file'])
    assert path.is_relative_to(root);add(path)
libraries={}
for name in ('build-mlp-command-buffer','build-coalesced-bf16-body','build-resident-kv-inplace'):
    path=root/name/'libgarnet.dylib';add(path);libraries[name]=digest(path)
meta=folder/'jobs/metadata.json'
if meta.exists():
    info=json.loads(meta.read_text())
    assert libraries['build-mlp-command-buffer']==info['candidate_sha256']
    assert libraries['build-resident-kv-inplace']==info['control_sha256']
proof=dict(files={},omitted=omitted,jobs={trial:status},library_sha256=libraries,
    scope='Fresh original28 validation in a separate generated-source tree; terminal success or failure only. No inference timing or vLLM comparison.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
    for name,path in sorted(files.items()):
        raw=path.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode()
    item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=status)))
