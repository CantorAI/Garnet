"""Resumable checkpoint transfer through the verified Mac SSH identity.

Each received block is checked before appending, so a dropped connection leaves
the previous verified prefix intact. No SSH configuration or keys are copied.
"""
import hashlib,json,shlex,subprocess,sys,time,zlib
from datetime import datetime,timezone
from pathlib import Path

source=Path(sys.argv[1]).resolve()
expected_size=1746942600
expected_sha='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
remote='/Users/zsdx/garnet-metal-investigation/models/Qwen3.5-0.8B/model.safetensors'
logfile=source.parents[2]/'logs'/'checkpoint-transfer.jsonl'
logfile.parent.mkdir(parents=True,exist_ok=True)
def emit(record):
    record['utc']=datetime.now(timezone.utc).isoformat()
    line=json.dumps(record)
    with logfile.open('a',encoding='utf-8') as log:log.write(line+'\n')
    print(line,flush=True)
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10',
     '-o','ServerAliveCountMax=2','-o','HostKeyAlias=192.168.193.215','zsdx@100.112.239.81']
if source.stat().st_size!=expected_size:raise ValueError('Local checkpoint size mismatch')
with source.open('rb') as stream:
    if hashlib.file_digest(stream,'sha256').hexdigest()!=expected_sha:
        raise ValueError('Local pinned checkpoint digest mismatch')

def execute(code,payload=None):
    result=subprocess.run(ssh+['/opt/homebrew/bin/python3 -c '+shlex.quote(code)],
        input=payload,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    if result.returncode:raise ConnectionError('SSH operation did not complete')
    return json.loads(result.stdout)

def inspect():
    return execute("import hashlib,json,pathlib; p=pathlib.Path("+repr(remote)+
        "); p=p if p.exists() else pathlib.Path(str(p)+'.part'); "
        "p.parent.mkdir(parents=True,exist_ok=True); p.touch(exist_ok=True); "
        "f=p.open('rb'); h=hashlib.file_digest(f,'sha256').hexdigest(); f.close(); "
        "print(json.dumps(dict(size=p.stat().st_size,sha256=h,complete=p.name=='model.safetensors')))")

offset=None;attempts=0;block_size=4*1024*1024
with source.open('rb') as stream:
    while True:
        try:
            if offset is None:
                info=inspect();offset=info['size']
                if offset>expected_size:raise ValueError('Remote checkpoint exceeds pinned size')
                stream.seek(0);prefix=hashlib.sha256();remaining=offset
                while remaining:
                    block=stream.read(min(block_size,remaining));prefix.update(block);remaining-=len(block)
                if prefix.hexdigest()!=info['sha256']:
                    raise ValueError('Remote prefix differs from pinned local checkpoint; retained for inspection')
                if info['complete'] and offset!=expected_size:raise ValueError('Incomplete finalized checkpoint')
                emit(dict(event='verified_prefix',bytes=offset))
            if offset==expected_size:
                result=execute("import hashlib,json,os,pathlib; p=pathlib.Path("+repr(remote)+
                    "); q=p if p.exists() else pathlib.Path(str(p)+'.part'); "
                    "f=q.open('rb'); h=hashlib.file_digest(f,'sha256').hexdigest(); f.close(); "
                    "assert q.stat().st_size=="+str(expected_size)+" and h=="+repr(expected_sha)+"; "
                    "os.replace(q,p) if q!=p else None; print(json.dumps(dict(event='checkpoint_ready',sha256=h)))")
                emit(result);break
            stream.seek(offset);block=stream.read(block_size);digest=hashlib.sha256(block).hexdigest()
            payload=zlib.compress(block,3);block_started=time.monotonic()
            code=("import hashlib,json,os,pathlib,sys,zlib; p=pathlib.Path("+repr(remote+'.part')+
                "); data=zlib.decompress(sys.stdin.buffer.read()); assert len(data)=="+str(len(block))+" and hashlib.sha256(data).hexdigest()=="+repr(digest)+
                "; f=p.open('r+b'); f.seek(0,2); assert f.tell()=="+str(offset)+
                "; f.write(data); f.flush(); os.fsync(f.fileno()); f.close(); print(json.dumps(dict(bytes=p.stat().st_size)))")
            result=execute(code,payload)
            if result['bytes']!=offset+len(block):raise ValueError('Remote append length mismatch')
            offset=result['bytes'];attempts=0
            emit(dict(event='appended',bytes=offset,total=expected_size,wire_bytes=len(payload),
                block_seconds=time.monotonic()-block_started))
        except ConnectionError:
            # A disconnected acknowledgement can follow a successful write.
            # Inspect and verify the prefix before resuming; never repeat blindly.
            offset=None;attempts+=1
            emit(dict(event='connection_interrupted',attempt=attempts))
            if attempts>=12:raise
            time.sleep(5)

# Metadata stays in the same isolated model folder. Legacy SCP avoids depending
# on a separately configured SFTP subsystem; it transfers no credentials.
metadata=[str(p) for p in sorted(source.parent.glob('*.json'))]
result=subprocess.run(['scp','-O','-q']+ssh[1:-1]+metadata+
    ['zsdx@100.112.239.81:'+remote.rsplit('/',1)[0]+'/' ],stderr=subprocess.DEVNULL)
emit(dict(event='metadata_transfer',exit_code=result.returncode))
raise SystemExit(result.returncode)
