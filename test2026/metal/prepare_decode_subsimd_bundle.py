"""Package explicit candidate sources with verified original identities."""
import hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
names=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp',
       'test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py',
       'test2026/metal/run_decode_subsimd_checks.py','test2026/metal/compare_decode_subsimd.py',
       'test2026/metal/integrate_decode_subsimd.py']
manifest={}
with tarfile.open(out/'decode-subsimd-source.tar.gz','w:gz') as archive:
    for name in names:
        old=subprocess.run(['git','show','HEAD:'+name],cwd=root,capture_output=True)
        raw=(root/name).read_bytes().replace(b'\r\n',b'\n')
        manifest[name]=dict(before_sha256=hashlib.sha256(old.stdout).hexdigest() if old.returncode==0 else None,sha256=hashlib.sha256(raw).hexdigest())
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
(out/'decode-subsimd-manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
print(json.dumps(dict(files=len(names),archive_sha256=hashlib.sha256((out/'decode-subsimd-source.tar.gz').read_bytes()).hexdigest())))
