"""Fetch the pinned official model into the isolated investigation folder."""
import hashlib,json,shutil,sys,time,urllib.request
from pathlib import Path
REVISION='2fc06364715b967f1860aea9cf38778875588b17'
MODEL_SHA256='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=True)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
manifest={}
for name in ('config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json','model.safetensors'):
 target=root/name;partial=root/(name+'.part')
 if target.is_file():
  sha=digest(target)
  if name=='model.safetensors' and sha!=MODEL_SHA256:raise RuntimeError('Existing checkpoint checksum mismatch; preserved for inspection')
  manifest[name]=dict(bytes=target.stat().st_size,sha256=sha);continue
 if shutil.disk_usage(root).free<8*1024**3:raise RuntimeError('Need at least 8 GiB free before downloading')
 url='https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/'+REVISION+'/'+name
 for attempt in range(3):
  try:
   offset=partial.stat().st_size if partial.exists() else 0
   request=urllib.request.Request(url,headers={'Range':'bytes='+str(offset)+'-'} if offset else {})
   with urllib.request.urlopen(request,timeout=90) as response:
    resume=response.status==206 and response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-')
    if response.status not in (200,206) or (response.status==206 and not resume):raise RuntimeError('Invalid download range response')
    with partial.open('ab' if resume else 'wb') as stream:
     for chunk in iter(lambda:response.read(8*1024*1024),b''):stream.write(chunk)
   sha=digest(partial)
   if name=='model.safetensors' and sha!=MODEL_SHA256:raise RuntimeError('Downloaded checkpoint checksum mismatch')
   if name.endswith('.json'):json.loads(partial.read_text())
   partial.replace(target);manifest[name]=dict(bytes=target.stat().st_size,sha256=sha)
   print('verified',name,manifest[name]['bytes'],flush=True);break
  except Exception:
   if attempt==2:raise
   time.sleep(2)
(root/'download-manifest.json').write_text(json.dumps(dict(model='Qwen/Qwen3.5-0.8B',revision=REVISION,files=manifest),indent=2))
print('Pinned checkpoint download complete',flush=True)
