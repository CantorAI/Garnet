"""Archive completed component evidence; omit large matrix assets and binaries."""
import hashlib,json,platform,subprocess,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();entries=[];omitted=[]
jobs=('extended-half-scale-20261013','extended-half-projection-20261013','extended-half-projection-diagnostic-20261013','scale-bits-20261013','extended-half-bits-projection-20261013','extended-half-hybrid-projection-20261013')
for job in jobs:
    folder=root/'logs'/job
    assert (folder/'exit.json').is_file(),'Job not terminal: '+job
    for path in sorted(folder.rglob('*')):
        if not path.is_file():continue
        entry=dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest())
        if path.suffix in ('.json','.log','.txt') or path.name=='pid':entries.append((path,entry))
        else:omitted.append(entry)
for name in ('extended_half_scale_gpu.mm','run_extended_half_scale_gpu.py','extended_half_projection_probe.mm','experimental_extended_half_codec.h','run_extended_half_projection.py','archive_extended_half_scale.py','prepare_prefill_tile_probe.py','compare_vllm_metal.py','background_comparison.py','scale_bits_gpu.mm','run_scale_bits_gpu.py','experimental_scale_bits.metal','extended_half_bits_projection_probe.mm','run_extended_half_bits_projection.py','extended_half_hybrid_projection_probe.mm','run_extended_half_hybrid_projection.py'):
    path=root/'source/test2026/metal'/name
    entries.append((path,dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest())))
metadata=dict(scope=__doc__,files=[e for _,e in entries],omitted=omitted,versions=dict(python=sys.version,macos=platform.platform(),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)))
manifest=target.with_suffix('.manifest.json');manifest.write_text(json.dumps(metadata,indent=2))
with tarfile.open(target,'w:gz') as archive:
    archive.add(manifest,arcname='manifest.json')
    for path,entry in entries:archive.add(path,arcname=entry['path'])
print(json.dumps(dict(archive=str(target),bytes=target.stat().st_size,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),files=len(entries),omitted=len(omitted))))
