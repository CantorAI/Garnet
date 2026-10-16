"""Extend the frozen fresh30 archive with full native ABBA and both driver/source closures."""
from pathlib import Path
import hashlib,io,json,struct,subprocess,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();trial=sys.argv[3] if len(sys.argv)>3 else 'ndarray-native-20261015'
assert trial in ('ndarray-native-20261015','ndarray-native-confirm-20261015') and not target.exists()
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
jobs=root/'logs'/trial/'jobs';status=json.loads((jobs.parent/'exit.json').read_text());assert isinstance(status['exit_code'],int)
audit=json.loads((jobs/'checks-verification.json').read_text());validation=root/'ndarray-native-checks-repaired-20261015.tar.gz'
assert digest(validation)==audit['archive_sha256']
files={};headers={}
with tarfile.open(validation,'r:gz') as old:
    proof=json.load(old.extractfile('evidence/manifest.json'));omitted=proof['omitted'];libraries=proof['library_sha256']
    for name,info in proof['files'].items():
        raw=old.extractfile(name).read();assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256'];headers[name]=raw
def add(p):
    name=p.relative_to(root).as_posix()
    if p.suffix in ('.json','.py','.h','.cpp','.mm','.metal','.cmake','.log','.txt','.make'):files['evidence/'+name]=p
    else:omitted[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
for p in jobs.parent.rglob('*'):
    if p.is_file():add(p)
identities=json.loads((jobs/'source-identities.json').read_text())
for name,sha in identities.items():assert digest(root/'source'/name)==digest(jobs/'source-snapshot'/name)==sha
external=json.loads((jobs/'external-tool-identities.json').read_text())
for name,sha in external['files'].items():
    p=Path(external['directory'])/name;assert digest(p)==digest(jobs/'tool-source-snapshot'/name)==sha
    files['evidence/external-tools/'+name]=p
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    with (root/'models'/model/'model.safetensors').open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0];raw=stream.read(size)
    assert len(raw)==size and isinstance(json.loads(raw),dict);headers['evidence/checkpoint-headers/'+model+'.json']=raw
hardware={name:subprocess.check_output(['sysctl','-n',name],text=True).strip() for name in ('machdep.cpu.brand_string','hw.memsize','hw.model')}
hardware['sw_vers']=subprocess.check_output(['sw_vers'],text=True)
proof=dict(files={},omitted=omitted,hardware=hardware,jobs={**proof['jobs'],trial:status},library_sha256=libraries,
    checks_archive_sha256=audit['archive_sha256'],scope='Fresh30 and native full-model row80 NDArray ABBA; no fresh vLLM ratio.',full_goal_met=False)
with tarfile.open(target,'w:gz') as archive:
    for name in sorted(set(headers)|set(files)):
        raw=files[name].read_bytes() if name in files else headers[name];proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(proof['files']),sha256=digest(target),status=status)))
