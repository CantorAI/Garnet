"""Download and verify the full pinned checkpoint without logging signed URLs."""
import hashlib,json,shutil,sys,urllib.request
from pathlib import Path
REVISION='2fc06364715b967f1860aea9cf38778875588b17'
MODEL='Qwen/Qwen3.5-0.8B'
WEIGHT='model.safetensors-00001-of-00001.safetensors'
SIZE=1746942600
def digest_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(4*1024*1024):digest.update(chunk)
    return digest
def main():
    folder=Path(sys.argv[1]);folder.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true',timeout=30) as response:
        info=json.load(response)
    if info['sha']!=REVISION:raise ValueError('unexpected checkpoint revision')
    files={entry['rfilename']:entry for entry in info['siblings']}
    metadata=files[WEIGHT];sha=metadata['lfs']['sha256']
    if metadata['size']!=SIZE or len(sha)!=64:raise ValueError('unexpected checkpoint size/hash metadata')
    target=folder/'model.safetensors';partial=folder/'model.safetensors.part'
    if target.exists():
        if target.stat().st_size!=SIZE or digest_file(target).hexdigest()!=sha:raise ValueError('existing full checkpoint failed verification')
    else:
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>SIZE:raise ValueError('oversized partial checkpoint')
        if shutil.disk_usage(folder).free<SIZE-offset+128*1024*1024:
            raise ValueError('insufficient free space for checkpoint and reserve')
        digest=digest_file(partial) if offset else hashlib.sha256()
        if offset<SIZE:
            url=f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{WEIGHT}'
            request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
            with urllib.request.urlopen(request,timeout=60) as response,partial.open('ab') as stream:
                if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-{SIZE-1}/')):
                    raise ValueError('resume byte range was not honored')
                total=offset;reported=total
                while total<SIZE:
                    chunk=response.read(min(1024*1024,SIZE-total))
                    if not chunk:raise ValueError('short checkpoint download')
                    stream.write(chunk);digest.update(chunk);total+=len(chunk)
                    if total-reported>=64*1024*1024:
                        stream.flush()
                        if shutil.disk_usage(folder).free<64*1024*1024:raise ValueError('disk reserve exhausted during download')
                        print('checkpoint bytes',total,'of',SIZE,flush=True);reported=total
        if digest.hexdigest()!=sha:raise ValueError('full checkpoint SHA256 mismatch')
        partial.replace(target)
    wanted=('config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json',
        'processor_config.json','generation_config.json','chat_template.jinja','chat_template.json')
    for name in wanted:
        if name not in files:continue
        url=f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
        with urllib.request.urlopen(url,timeout=30) as response:data=response.read()
        (folder/name).write_bytes(data)
        print('saved checkpoint metadata',name,'bytes',len(data),flush=True)
    (folder/'garnet_checkpoint_revision.json').write_text(json.dumps(dict(model=MODEL,revision=REVISION,sha256=sha,size=SIZE),indent=2))
    print('full pinned checkpoint verified SHA256',sha,'bytes',SIZE,flush=True)
if __name__=='__main__':
    try:main()
    except Exception as error:
        # HTTP redirect exceptions can contain signed download query strings.
        print('full checkpoint preparation failed:',type(error).__name__,file=sys.stderr)
        raise SystemExit(1)
