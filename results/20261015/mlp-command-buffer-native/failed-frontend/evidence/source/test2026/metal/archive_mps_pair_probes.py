"""Archive completed scheduling/packing component probes, without model matrices."""
import hashlib,json,platform,subprocess,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();entries=[];omitted=[];jobs={}
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
for name in ('concurrent-mps-pair-20261013','concurrent-mps-dependency-20261013','joined-mps-pair-20261013','joined-mps-pair-confirm-20261013'):
    folder=r/'logs'/name;assert (folder/'exit.json').is_file(),'Job not terminal: '+name
    jobs[name]=json.loads((folder/'exit.json').read_text())
    for path in sorted(folder.rglob('*')):
        if not path.is_file():continue
        entry=dict(path=str(path.relative_to(r)),bytes=path.stat().st_size,sha256=digest(path))
        if path.suffix in ('.json','.log','.txt') or path.name=='pid':entries.append((path,entry))
        else:omitted.append(entry)
for name in ('test2026/metal/concurrent_mps_pair_gpu.mm','test2026/metal/run_concurrent_mps_pair.py','test2026/metal/concurrent_mps_dependency_gpu.mm','test2026/metal/run_concurrent_mps_dependency.py','test2026/metal/joined_mps_pair_gpu.mm','test2026/metal/run_joined_mps_pair.py','test2026/metal/archive_mps_pair_probes.py','test2026/metal/prepare_prefill_tile_probe.py','test2026/metal/compare_vllm_metal.py','test2026/metal/background_comparison.py','src/backends/metal/exact_half_codec.h'):
    path=r/'source'/name;entries.append((path,dict(path=str(path.relative_to(r)),bytes=path.stat().st_size,sha256=digest(path))))
metadata=dict(scope=__doc__,jobs=jobs,files=[e for _,e in entries],omitted=omitted,versions=dict(python=sys.version,macos=platform.platform(),compiler=subprocess.check_output(['/usr/bin/xcrun','clang++','--version'],text=True)))
manifest=target.with_suffix('.manifest.json');manifest.write_text(json.dumps(metadata,indent=2))
with tarfile.open(target,'w:gz') as archive:
    archive.add(manifest,arcname='manifest.json')
    for path,entry in entries:archive.add(path,arcname=entry['path'])
print(json.dumps(dict(archive=str(target),bytes=target.stat().st_size,sha256=digest(target),files=len(entries))))
