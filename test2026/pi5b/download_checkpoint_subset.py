"""Extract pinned pretrained tensors using validated HTTP byte ranges."""
import hashlib,json,struct,sys,urllib.request
from pathlib import Path
REVISION="2fc06364715b967f1860aea9cf38778875588b17"
URL=f"https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/{REVISION}/model.safetensors-00001-of-00001.safetensors"
def main():
    header=json.loads(Path(sys.argv[1]).read_text())
    output=Path(sys.argv[2])
    prefix=sys.argv[3]
    with urllib.request.urlopen(URL,timeout=30) as response:
        raw=response.read(8)
        if len(raw)!=8:raise ValueError("short header length")
        data_start=8+struct.unpack('<Q',raw)[0]
    names=sorted(name for name in header if name.startswith(prefix+'.'))
    if not names:raise ValueError("no matching pretrained tensors")
    selected={};cursor=0;hashes={}
    for name in names:
        entry=header[name];start,end=entry['data_offsets']
        selected[name]=dict(dtype=entry['dtype'],shape=entry['shape'],data_offsets=[cursor,cursor+end-start])
        cursor+=end-start
    encoded=json.dumps(selected,separators=(',',':')).encode()
    encoded+=b' '*((-len(encoded))%8)
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix(output.suffix+'.part')
    with partial.open('wb') as stream:
        stream.write(struct.pack('<Q',len(encoded)));stream.write(encoded)
        for name in names:
            start,end=header[name]['data_offsets'];start+=data_start;end+=data_start
            request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end-1}'})
            digest=hashlib.sha256()
            with urllib.request.urlopen(request,timeout=30) as response:
                content_range=response.headers.get('Content-Range','')
                if response.status!=206 or not content_range.startswith(f'bytes {start}-{end-1}/'):
                    raise ValueError("server did not honor the tensor byte range")
                remaining=end-start
                while remaining:
                    chunk=response.read(min(1024*1024,remaining))
                    if not chunk:raise ValueError("short tensor byte range")
                    stream.write(chunk);digest.update(chunk);remaining-=len(chunk)
            hashes[name]=digest.hexdigest()
            print('downloaded pretrained tensor',name,'bytes',end-start,flush=True)
    partial.replace(output)
    output.with_suffix('.manifest.json').write_text(json.dumps(dict(revision=REVISION,tensors=hashes,total_tensor_bytes=cursor),indent=2))
    print('pretrained subset saved',len(names),'tensors',cursor,'bytes',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as error:
        # Never print redirects or signed download URLs.
        print('pretrained subset download failed:',type(error).__name__,file=sys.stderr)
        raise SystemExit(1)
