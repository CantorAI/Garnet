"""Freeze terminal fresh32 native checks, exact source closure, fixtures and external auditors."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();trial='delta-value-tile64-native-checks-20261016';folder=root/'logs'/trial
status=json.loads((folder/'exit.json').read_text());assert isinstance(status['exit_code'],int) and not target.exists();prep=json.loads((root/'preparation.json').read_text());files={};omitted={}

def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def add(p,name=None):
 key=name or p.relative_to(root).as_posix()
 if p.suffix in ('.json','.py','.h','.hpp','.cpp','.mm','.metal','.cmake','.log','.txt','.make','.x','.md'):files['evidence/'+key]=p
 else:omitted[key]=dict(bytes=p.stat().st_size,sha256=digest(p))

def tree(folder):
 for p in folder.rglob('*'):
  if p.is_file():add(p)

add(root/'preparation.json');tree(folder);tree(root/'selected-component')
for name,sha in prep['source_sha256'].items():
 p=root/'source'/name;assert digest(p)==sha,name;files['evidence/source/'+name]=p
for name,change in prep['changes'].items():
 p=root/'original-source'/name;assert digest(p)==change['before'];add(p)
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 for name in ('config.json','tokenizer.json','model.safetensors'):add(root/'models'/model/name)
 tree(root/'logs/screened-head-after-vision-20261015/jobs'/(model+'-assets'))
for fixture in (folder/'jobs/full-grid').glob('*-fixture.json'):
 p=Path(json.loads(fixture.read_text())['patch_file']);assert p.is_relative_to(root);add(p)
libraries={}
for name in ('build-delta-value-tile64-prefill','build-vector-vision-width4','build-coalesced-bf16-body','build-resident-kv-inplace'):
 p=root/name/'libgarnet.dylib'
 if p.exists():add(p);libraries[name]=digest(p)
build=root/'build-delta-value-tile64-prefill'
for p in (build.iterdir() if build.exists() else []):
 if p.is_file() and (p.suffix=='.a' or (p.name.startswith('garnet') and not p.suffix)):add(p)
for name in ('archive_delta_value_tiles_native_checks.py','verify_delta_value_tiles_native_checks.py'):
 add(Path(__file__).with_name(name),'external-archive-tools/'+name)
for name,sha in prep['audit_dependency_sha256'].items():
 p=Path(__file__).with_name(name);assert digest(p)==sha;add(p,'external-validation-tools/'+name)
proof=dict(files={},omitted=omitted,jobs={trial:status},library_sha256=libraries,audit_dependency_sha256=prep['audit_dependency_sha256'],scope='Fresh original30 and real-library original-bit recurrent chunk primitive compile/execution. Terminal success or failure; no model timing claim.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
 for name,p in sorted(files.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=status)))
