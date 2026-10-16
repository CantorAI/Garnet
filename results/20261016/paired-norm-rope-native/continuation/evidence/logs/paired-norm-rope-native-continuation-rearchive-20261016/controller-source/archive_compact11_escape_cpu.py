"""Freeze terminal CPU-only exponent study, prior fixture/checkpoint identities and hardware."""
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();assert not target.exists()
trial='compact11-escape-cpu-20261016';folder=root/'logs'/trial
status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==0
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
files={};omitted={}
for p in folder.rglob('*'):
    if p.is_file():files['evidence/'+p.relative_to(root).as_posix()]=p
report=json.loads((folder/'report.json').read_text());meta=json.loads((folder/'metadata.json').read_text())
for p,sha in meta['source_sha256'].items():assert digest(Path(p))==sha
for model in report['models']:
    name=model['model'];checkpoint=root/'models'/name/'model.safetensors'
    assert digest(checkpoint)==model['checkpoint_sha256']
    omitted[checkpoint.relative_to(root).as_posix()]=dict(bytes=checkpoint.stat().st_size,sha256=model['checkpoint_sha256'])
    assets=root/'logs/decoder-codec-gpu-20261013/jobs'/(name+'-assets')
    files['evidence/'+(assets/'manifest.json').relative_to(root).as_posix()]=assets/'manifest.json'
    manifest=json.loads((assets/'manifest.json').read_text());assert len(manifest['cases'])==len(model['cases'])
    for c,r in zip(manifest['cases'],model['cases']):
        p=assets/c['name']/'raw.bin';assert digest(p)==c['sha256']['raw.bin']==r['raw_sha256']
        omitted[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=r['raw_sha256'])
p=Path(__file__);files['evidence/tools/'+p.name]=p
proof=dict(files={},omitted=omitted,jobs={trial:status},full_goal_met=False,
    hardware={k:subprocess.check_output(['sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize','hw.model')},
    scope='CPU-only exponent histograms and lossless reconstruction expression. No packet implementation, GPU timing, model speed, or RSS claim.')
with tarfile.open(target,'w:gz') as tar:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        member=tarfile.TarInfo(name);member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();member=tarfile.TarInfo('evidence/manifest.json');member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=status)))
