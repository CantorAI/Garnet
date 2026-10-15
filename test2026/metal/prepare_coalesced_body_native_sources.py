"""Pin the native exact-block integration after repeated bit-identical components."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/coalesced-body-native-staging';stage.mkdir(exist_ok=False)
base='eaf421d160c04ab0e5259345123279c07678d278';original={};manifest={}
existing=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/metal_context.h','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','src/entry/metal_main.cpp','test2026/metal/frame_worker_garnet.py','test2026/metal/run_matched_garnet.py']
extra=['src/backends/metal/coalesced_body_block_shader.h','test2026/metal/coalesced_bf16_body_test.cpp','test2026/metal/run_guarded_coalesced_body_native_checks.py','test2026/metal/launch_coalesced_body_native_checks.py']
raw=(root/'results/20261015/coalesced-body-block/verification.json').read_bytes();audit=json.loads(raw)
assert audit['numeric_checks']==144 and audit['control_profiles']==378 and audit['timed_executions']==2304
(stage/'component-verification.json').write_bytes(raw)
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
    for name in existing+extra:
        if name in existing:original[name]=hashlib.sha256(subprocess.check_output(['git','show',base+':'+name],cwd=root)).hexdigest()
        raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();item=tarfile.TarInfo(name);item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
print('Prepared',len(manifest),'pinned native block sources; originals from',base)
