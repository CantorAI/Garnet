"""Freeze any terminal private34 paired norm-RoPE check result, full source and fixtures."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();trial='paired-norm-rope-native-checks-20261016';folder=root/'logs'/trial
status=json.loads((folder/'exit.json').read_text());assert isinstance(status['exit_code'],int) and not target.exists();prep=json.loads((root/'preparation.json').read_text())
files={};omitted={};text={'.json','.py','.h','.hpp','.cpp','.mm','.metal','.cmake','.log','.txt','.make','.x','.md'}
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def add(p,name=None):
 key=name or p.relative_to(root).as_posix()
 if p.suffix in text:files['evidence/'+key]=p
 else:omitted[key]=dict(bytes=p.stat().st_size,sha256=digest(p))
def tree(folder):
 for p in folder.rglob('*'):
  if p.is_file():add(p)
add(root/'preparation.json');tree(folder)
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
for name in ('build-paired-norm-rope-decode','build-vector-vision-width4','build-coalesced-bf16-body','build-resident-kv-inplace'):
 p=root/name/'libgarnet.dylib'
 if p.exists():add(p);libraries[name]=digest(p)
for p in ((root/'build-paired-norm-rope-decode').iterdir() if (root/'build-paired-norm-rope-decode').exists() else []):
 if p.is_file() and (p.suffix=='.a' or (p.name.startswith('garnet') and not p.suffix)):add(p)
for name in ('archive_paired_norm_rope_native_checks.py','verify_paired_norm_rope_native_checks.py'):
 p=Path(__file__).with_name(name);add(p,'external-archive-tools/'+name)
# The private runtime tree predates the vector auditor. Preserve that tree and
# archive the auditors separately, checked against the original frozen controller.
parent=Path(prep['parent']);controller=parent/'logs/paired-norm-rope-native-continuation-20261016'
assert json.loads((controller/'exit.json').read_text())['exit_code']==1
identities=json.loads((controller/'metadata.json').read_text())['tool_source_sha256'];dependencies={}
for name in ('verify_vector_vision_native_checks.py','verify_mlp_prefill_native_checks.py'):
 p=Path(__file__).with_name(name);assert digest(p)==digest(controller/'controller-source'/name)==identities[name]
 dependencies[name]=identities[name];add(p,'external-validation-tools/'+name)
add(controller/'metadata.json','archive-repair/original-controller-metadata.json')
proof=dict(files={},omitted=omitted,jobs={trial:status},library_sha256=libraries,audit_dependency_sha256=dependencies,scope='Fresh original30, actual linked paired norm-RoPE primitive and two original-bound trained graph/session decode/fallback suites; terminal failure or success preserved, no performance claim.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
 for name,p in sorted(files.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=status)))
