"""Download the exact pinned checkpoint directly on the Mac, with resumable curl.

The official ModelScope file listing advertises the same bytes/SHA256 as the
HF revision already verified locally. Never publish a partial checkpoint.
"""
import hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlencode
folder=Path(sys.argv[1]);folder.mkdir(parents=True,exist_ok=True)
expected_size=1746942600
expected_sha='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
base='https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-0.8B/repo?'
def emit(event,**fields):
    print(json.dumps(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),**fields)),flush=True)
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def fetch(name,target,size,sha,resume=False):
    if target.exists() and target.stat().st_size==size and digest(target)==sha:
        emit('existing_verified',name=target.name,bytes=size);return
    temporary=Path(str(target)+'.part') if resume else Path(str(target)+'.download')
    url=base+urlencode(dict(Revision='master',FilePath=name))
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

# These are the same metadata files used by the native tokenizer and image prep.
metadata=[
 ('config.json',2907,'b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204'),
 ('preprocessor_config.json',390,'27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516'),
 ('tokenizer_config.json',16709,'49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c'),
 ('tokenizer.json',12807982,'5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42')]
for name,size,sha in metadata:fetch(name,folder/name,size,sha)
fetch('model.safetensors-00001-of-00001.safetensors',folder/'model.safetensors',expected_size,expected_sha,True)
emit('checkpoint_ready',bytes=expected_size,sha256=expected_sha)
