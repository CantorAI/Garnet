"""Download official dense Qwen files pinned by manifest size and SHA-256.

ModelScope master URLs may change; an unexpected hash fails verification.
Only a verified complete file replaces its final name. Partial data is retained
for an explicit retry through curl's resume support.
"""
import hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlencode
manifest=json.loads(Path(sys.argv[1]).read_text())
folder=Path(sys.argv[2]);folder.mkdir(parents=True,exist_ok=True)
base='https://modelscope.cn/api/v1/models/'+manifest['repo']+'/repo?'

def emit(event,**fields):
    print(json.dumps(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),**fields)),flush=True)
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def fetch(name,target,size,sha,resume=False):
    if target.exists() and target.stat().st_size==size and digest(target)==sha:
        emit('existing_verified',name=target.name,bytes=size);return
    temporary=Path(str(target)+'.part') if resume else Path(str(target)+'.download')
    url=base+urlencode(dict(Revision=manifest['revision'],FilePath=name))
    for attempt in range(1,13):
        count=temporary.stat().st_size if temporary.exists() else 0
        if count>size:raise ValueError('Download exceeds pinned size')
        if count==size:break
        command=['curl','--fail','--location','--silent','--connect-timeout','15',
            '--speed-time','30','--speed-limit','1024','--output',str(temporary)]
        if resume and count:command+=['--continue-at','-']
        command.append(url)
        emit('download_started',name=name,attempt=attempt,bytes=count,total=size)
        process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        while process.poll() is None:
            time.sleep(5)
            emit('download_progress',name=name,pid=process.pid,
                bytes=temporary.stat().st_size if temporary.exists() else 0,total=size)
        emit('download_finished',name=name,exit_code=process.returncode,
            bytes=temporary.stat().st_size if temporary.exists() else 0)
        if process.returncode==0:break
        time.sleep(2)
    if not temporary.exists() or temporary.stat().st_size!=size or digest(temporary)!=sha:
        raise ValueError('Downloaded file failed pinned size/SHA256 verification; partial retained')
    os.replace(temporary,target);emit('file_verified',name=target.name,bytes=size,sha256=sha)

for entry in manifest['files']:
    fetch(entry['name'],folder/entry['name'],entry['size'],entry['sha256'],entry['name'].endswith('.safetensors'))
emit('checkpoint_ready',repo=manifest['repo'])
