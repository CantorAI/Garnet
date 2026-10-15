"""Pin native integration after two complete original-bit component trials."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/interleaved-body-native-staging';stage.mkdir(exist_ok=False)
base='8c9520cd87b4bce796f7ad0a2bc19c6532f81d1c';original={};manifest={}
existing=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/metal_context.h','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','src/entry/metal_main.cpp','test2026/metal/frame_worker_garnet.py','test2026/metal/run_matched_garnet.py']
extra=['src/backends/metal/interleaved_bf16_body_shader.h','test2026/metal/interleaved_bf16_body_test.cpp','test2026/metal/run_guarded_interleaved_body_native_checks.py','test2026/metal/launch_interleaved_body_native_checks.py']
raw=(root/'results/20261015/interleaved-bf16-body/verification.json').read_bytes();audit=json.loads(raw)
assert audit['numeric_checks']==144 and audit['control_profiles']==24 and audit['timed_executions']==2304
(stage/'component-verification.json').write_bytes(raw)
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
    for name in existing+extra:
        if name in existing:original[name]=hashlib.sha256(subprocess.check_output(['git','show',base+':'+name],cwd=root)).hexdigest()
        raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();item=tarfile.TarInfo(name);item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
print('Prepared',len(manifest),'native body source identities; no remote mutation')
