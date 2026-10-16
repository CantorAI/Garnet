"""Preserve rejected full-QKV native evidence, failed countfix finalizer and exact-input control."""
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();parent=Path(json.loads((root/'preparation.json').read_text())['parent']);assert not target.exists()
prior=parent/'prepared-qkv-native-checks-countfix-20261016.tar.gz'
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert digest(prior)=='544e11d2c2f18582ceffb6a12404fb3113d4f03d5b382b55944076894e849c1a';headers={};files={}
with tarfile.open(prior) as previous:
 proof=json.load(previous.extractfile('evidence/manifest.json'));omitted=dict(proof['omitted']);jobs=dict(proof['jobs']);libraries=dict(proof['library_sha256'])
 for name,info in proof['files'].items():
  raw=previous.extractfile(name).read();assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256'];headers[name]=raw
for folder in (root/'logs/prepared-qkv-exact-control-diagnostic-20261016',parent/'logs/prepared-qkv-native-finalize-countfix-20261016'):
 pid=json.loads((folder/'pid.json').read_text())['pid'];assert not subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True).stdout.strip();status=json.loads((folder/'exit.json').read_text());jobs[folder.name]=status
 for p in folder.rglob('*'):
  if p.is_file():files['evidence/logs/'+folder.name+'/'+p.relative_to(folder).as_posix()]=p
for name in ('archive_prepared_qkv_trained_failure.py','verify_prepared_qkv_trained_failure.py','diagnose_prepared_qkv_trained_failure.py','launch_prepared_qkv_failure_diagnostic.py'):
 p=Path(__file__).with_name(name);files['evidence/diagnostic-tools/'+name]=p
p=parent/'vector-vision-native-investigation-20261016/source/test2026/metal/run_dense_reference.py';files['evidence/control-source/run_dense_reference.py']=p
manifest=dict(files={},omitted=omitted,jobs=jobs,library_sha256=libraries,prior_archive_sha256=digest(prior),scope='Rejected native full-QKV fusion: trained original-bound candidate failure and accepted d52 exact-input control; no model speed measurement or integration.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
 for name in sorted(set(headers)|set(files)):
  raw=files[name].read_bytes() if name in files else headers[name]
  if name in headers and name in files:assert raw==headers[name]
  manifest['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 raw=(json.dumps(manifest,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(manifest['files']),sha256=digest(target),jobs=jobs)))
