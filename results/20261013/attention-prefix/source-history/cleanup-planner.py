import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation').resolve();logs=(root/'logs').resolve();models=(root/'models').resolve();cache={};items=[]
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for manifest in sorted(logs.rglob('manifest.json')):
    try:info=json.loads(manifest.read_text())
    except (ValueError,OSError):continue
    if not {'checkpoint','checkpoint_sha256','cases'}<=info.keys():continue
    checkpoint=Path(info['checkpoint']).resolve()
    if not checkpoint.is_relative_to(models) or not checkpoint.is_file():continue
    if checkpoint not in cache:cache[checkpoint]=digest(checkpoint)
    if cache[checkpoint]!=info['checkpoint_sha256']:continue
    for case in info['cases']:
        if not {'name','columns','inner','sha256'}<=case.keys():continue
        name=case['name']
        if '/' in name or '\\' in name:continue
        path=(manifest.parent/(name+'.bin')).resolve()
        if not path.is_relative_to(logs) or not path.is_file() or path.is_symlink():continue
        size=path.stat().st_size
        if size!=int(case['columns'])*int(case['inner'])*2:continue
        actual=digest(path)
        if actual!=case['sha256']:continue
        items.append(dict(path=str(path.relative_to(root)),bytes=size,sha256=actual,manifest=str(manifest.relative_to(root)),manifest_sha256=digest(manifest),checkpoint=str(checkpoint.relative_to(root)),checkpoint_sha256=cache[checkpoint]))
plan=dict(scope='Only verified reproducible BF16/synthetic matrix .bin assets in own investigation logs; models, libraries, text/JSON evidence and source retained',items=items,total_bytes=sum(v['bytes'] for v in items))
print(json.dumps(plan,indent=2))
