"""Freeze terminal native ABBA, both source closures and original validation."""
from pathlib import Path
import hashlib,io,json,struct,subprocess,sys,tarfile
p=Path(__file__).with_name('archive_shape_selected_two_checks.py')
source=p.read_text();exec(compile(source[:source.index('proof=dict(')],str(p),'exec'))
native_trial=sys.argv[3] if len(sys.argv)>3 else 'shape-selected-two-native-20261015'
assert native_trial in ('shape-selected-two-native-20261015','shape-selected-two-native-confirm-20261015')
native=root/'logs'/native_trial;native_status=json.loads((native/'exit.json').read_text());assert isinstance(native_status['exit_code'],int)
tree(native)
jobs=native/'jobs';identities=json.loads((jobs/'source-identities.json').read_text())
for name,sha in identities.items():
    assert digest(root/'source'/name)==sha and digest(jobs/'source-snapshot'/name)==sha
    add(root/'source'/name)
external=json.loads((jobs/'external-tool-identities.json').read_text())
for name,sha in external['files'].items():
    path=Path(external['directory'])/name
    assert digest(path)==sha and digest(jobs/'tool-source-snapshot'/name)==sha
    files['evidence/external-tools/'+name]=path
audit=json.loads((jobs/'checks-verification.json').read_text())
assert digest(root/'shape-selected-two-checks-20261015.tar.gz')==audit['archive_sha256']
headers={}
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    path=root/'models'/model/'model.safetensors'
    with path.open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0];raw=stream.read(size)
    assert len(raw)==size and isinstance(json.loads(raw),dict)
    headers['evidence/checkpoint-headers/'+model+'.json']=raw
hardware={name:subprocess.check_output(['sysctl','-n',name],text=True).strip() for name in ('machdep.cpu.brand_string','hw.memsize','hw.model')}
hardware['sw_vers']=subprocess.check_output(['sw_vers'],text=True)
proof=dict(files={},omitted=omitted,hardware=hardware,jobs={trial:status,native_trial:native_status},
    library_sha256=libraries,checks_archive_sha256=audit['archive_sha256'],
    scope='Native full-model comparison: selected two-output shapes versus retained c186; no fresh vLLM ratio.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
    for name in sorted(set(files)|set(headers)):
        raw=headers[name] if name in headers else files[name].read_bytes()
        proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(proof['files']),sha256=digest(target),status=native_status)))
