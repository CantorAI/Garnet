import hashlib,json,subprocess,sys
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation').resolve();logs=(root/'logs').resolve();models=(root/'models').resolve()
plan=PLAN_PLACEHOLDER
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for line in subprocess.check_output(['/bin/ps','-axo','pid=,comm='],text=True).splitlines():
    parts=line.split()
    if parts and parts[-1].endswith(('_gpu','_probe')):raise RuntimeError('GPU probe is live; do not clean its inputs')
validated=[];checkpoints={};manifests={};seen=set()
for item in plan['items']:
    path=(root/item['path']).resolve();manifest=(root/item['manifest']).resolve();checkpoint=(root/item['checkpoint']).resolve()
    assert path.is_relative_to(logs) and manifest.is_relative_to(logs) and checkpoint.is_relative_to(models)
    assert path.suffix=='.bin' and path.parent==manifest.parent and path.is_file()
    job=root/'logs'/Path(item['path']).parts[1]
    assert (job/'exit.json').is_file(),'Asset job has no terminal result'
    if manifest not in manifests:manifests[manifest]=digest(manifest)
    if checkpoint not in checkpoints:checkpoints[checkpoint]=digest(checkpoint)
    assert manifests[manifest]==item['manifest_sha256'] and checkpoints[checkpoint]==item['checkpoint_sha256']
    assert path.stat().st_size==item['bytes'] and digest(path)==item['sha256']
    if path not in seen:validated.append((path,item));seen.add(path)
removed=[]
for path,item in sorted(validated,key=lambda value:value[1]['bytes'],reverse=True):
    # Delete individual verified derived files; never delete a directory.
    path.unlink();removed.append(item)
print(json.dumps(dict(removed=removed,total_bytes=sum(v['bytes'] for v in removed),checkpoint_sha256={str(p.relative_to(root)):sha for p,sha in checkpoints.items()}),indent=2))
