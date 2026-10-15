"""Preserve a terminal MPSGraph attempt, including failed compilation and raw identities."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();trials=sys.argv[3:] or ['mpsgraph-prefill-20261015']
files={};omitted={};jobs={};admission={}
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for name in trials:
    folder=root/'logs'/name;jobs[name]=json.loads((folder/'exit.json').read_text());assert isinstance(jobs[name]['exit_code'],int)
    for path in folder.rglob('*'):
        if not path.is_file():continue
        rel=str(path.relative_to(root))
        if path.suffix in ('.json','.py','.mm','.h','.cpp','.metal','.log','.txt'):files['evidence/'+rel]=path
        else:omitted[rel]=dict(bytes=path.stat().st_size,sha256=digest(path))
    metadata=folder/'jobs/metadata.json'
    if not metadata.exists():continue
    import numpy as np
    meta=json.loads(metadata.read_text());admission[name]={}
    for model,assets in meta['assets'].items():
        path=root/'models'/model/'model.safetensors';assert digest(path)==assets['checkpoint_sha256']
        omitted[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=assets['checkpoint_sha256'])
        base=root/'logs/decoder-codec-gpu-20261013/jobs'/(model+'-assets');path=base/'manifest.json';files['evidence/'+str(path.relative_to(root))]=path;cases=[]
        for case in assets['cases']:
            path=base/case['name']/'raw.bin';assert digest(path)==case['sha256']['raw.bin'];omitted[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=case['sha256']['raw.bin'])
            bits=np.fromfile(path,dtype='<u2').astype(np.int32);mag=bits&32767;largest=int(mag.max());e=(largest>>7)&255;scale=0 if largest==0 else 142-e
            normal=(mag>>7)&255;exponent=normal-127+scale;nonzero=mag!=0;valid=(~nonzero)|((normal!=0)&(normal!=255)&(exponent>=-24)&(exponent<=15))
            low=nonzero&(exponent<-17);discarded=np.clip(-exponent-17,0,24);valid&=(~low)|(((128+(mag&127))&((1<<discarded)-1))==0)
            special=case['tensor'].startswith('model.layers.') and case['tensor'].endswith('.self_attn.k_proj.weight')
            ready=not special and path.stat().st_size<=16*1024*1024 and (largest==0 or e not in (0,255)) and -64<=scale<=64 and bool(valid.all())
            cases.append(dict(name=case['name'],global_half_ready=ready,common_scale=scale,raw_sha256=case['sha256']['raw.bin']))
        admission[name][model]=cases
path=root/'build-interleaved-bf16-body/libgarnet_metal.a';omitted[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=digest(path))
for name in ('archive_mpsgraph_prefill.py','launch_mpsgraph_prefill.py'):
    path=root/'source/test2026/metal'/name;files['evidence/'+str(path.relative_to(root))]=path
proof=dict(files={},omitted=omitted,jobs=jobs,global_half_cases=admission,full_goal_met=False)
with tarfile.open(target,'w:gz') as archive:
    for name,path in sorted(files.items()):
        raw=path.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),jobs=jobs)))
