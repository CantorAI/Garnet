"""Freeze a bounded live-status snapshot without treating it as final evidence."""
import hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]);files={};status={}
for name,pid,script in [('residual-norm-checks-20261014',35350,'launch_residual_norm_checks.py'),('residual-norm-native-20261014',36987,'queue_residual_norm_profiles.py')]:
 folder=root/'logs'/name;terminal=folder/'exit.json'
 if terminal.exists():status[name]=dict(terminal=True,**json.loads(terminal.read_text()))
 else:
  assert json.loads((folder/'pid.json').read_text())['pid']==pid
  probe=subprocess.run(['/bin/ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
  if terminal.exists():status[name]=dict(terminal=True,**json.loads(terminal.read_text()))
  else:
   assert probe.returncode==0 and shlex.split(probe.stdout.strip())[1:]==[str(root/'source/test2026/metal'/script),str(root),str(folder),'--worker']
   status[name]=dict(terminal=False,verified_live=True,pid=pid)
 for relative in ('pid.json','exit.json','output.log','dependency-wait.json','jobs/metadata.json','jobs/statuses.json','jobs/CTest-LastTest.log'):
  p=folder/relative
  if p.is_file():files['native/pending/'+name+'/'+relative]=p
for name,expected in [('build-resident-kv-inplace','73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2'),('build-attention-prefix','f2c5c448ec4f3c8431be37b48a2634d8fa4cc31de4aa163ec367a319c9263088')]:
 actual=hashlib.file_digest((root/name/'libgarnet.dylib').open('rb'),'sha256').hexdigest();assert actual==expected
status['selected_libraries_unchanged']=True
ctest=root/'build-residual-norm/Testing/Temporary/LastTest.log'
if ctest.is_file():files['native/pending/CTest-LastTest.log']=ctest
for relative in ('original.json','manifest.json','source.tar.gz','install_residual_norm_sources.py'):
 files['native/pending/staging/'+relative]=root/'residual-norm-staging'/relative
manifest={}
with tarfile.open(destination,'w:gz') as tar:
 for name,p in files.items():
  raw=p.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 for name,value in [('native/pending/files.json',manifest),('native/pending/status.json',status)]:
  raw=(json.dumps(value,indent=2)+'\n').encode();info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
print(json.dumps(status))
