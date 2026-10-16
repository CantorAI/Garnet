"""Freeze terminal validation, transformed/original sources and asset identities."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
trial='shape-selected-two-checks-20261015';folder=root/'logs'/trial
status=json.loads((folder/'exit.json').read_text());assert isinstance(status['exit_code'],int) and not target.exists()
preparation=json.loads((root/'preparation.json').read_text());parent=Path(preparation['parent'])
files={};omitted={};text_suffixes={'.json','.py','.h','.cpp','.mm','.metal','.cmake','.log','.txt','.make'}
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def add(path,name=None):
    key=name or path.relative_to(root).as_posix()
    if path.suffix in text_suffixes:files['evidence/'+key]=path
    else:omitted[key]=dict(bytes=path.stat().st_size,sha256=digest(path))
def tree(path):
    for p in path.rglob('*'):
        if p.is_file():add(p)
add(root/'preparation.json');tree(folder)
for name,sha in preparation['source_sha256'].items():
    p=root/'source'/name
    if p.suffix not in text_suffixes:continue
    assert digest(p)==sha,name
    add(p)
for name,change in preparation['changes'].items():
    if change['before'] is None:continue
    p=parent/'source'/name;assert digest(p)==change['before']
    add(p,'original-source/'+name)
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for name in ('config.json','tokenizer.json','model.safetensors'):add(root/'models'/model/name)
    tree(root/'logs/screened-head-after-vision-20261015/jobs'/(model+'-assets'))
for fixture in (folder/'jobs/full-grid').glob('*-fixture.json'):
    p=Path(json.loads(fixture.read_text())['patch_file']);assert p.is_relative_to(root);add(p)
libraries={}
for name in ('build-shape-selected-two-body','build-coalesced-bf16-body','build-resident-kv-inplace'):
    p=root/name/'libgarnet.dylib'
    if p.exists():add(p);libraries[name]=digest(p)
proof=dict(files={},omitted=omitted,jobs={trial:status},library_sha256=libraries,
    scope='Fresh28 validation and selected-route primitive, terminal success or failure preserved. No performance claim.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=status)))
