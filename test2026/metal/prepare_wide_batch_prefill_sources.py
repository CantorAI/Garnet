"""Pin the wide-batch candidate and preserve every replaced remote source byte."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[2];stage=root/'.work/wide-batch-prefill-staging';stage.mkdir(exist_ok=False)
base='e492e0541aec806b8efe30f6581895401b55da8b'
original={};manifest={}
existing=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/metal_context.h','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','src/entry/metal_main.cpp','xModel/qwen35/vl_0_8b/metal_decoder.py','test2026/metal/frame_worker_garnet.py']
extra=['test2026/metal/'+name for name in ('wide_batch_prefill_test.cpp','test_wide_batch_prefill_reference.py','run_guarded_wide_batch_prefill_checks.py','launch_wide_batch_prefill_checks.py')]
for folder,field,count in [('replanned-vl-prefill','full_output_requests',36),('interleaved-head-native','verified_files',1861)]:
    raw=(root/'results/20261015'/folder/'verification.json').read_bytes();assert json.loads(raw)[field]==count
    (stage/(folder+'-prior-verification.json')).write_bytes(raw)
with tarfile.open(stage/'source.tar.gz','w:gz') as archive:
    for name in existing+extra:
        if name in existing:original[name]=hashlib.sha256(subprocess.check_output(['git','show',base+':'+name],cwd=root)).hexdigest()
        raw=(root/name).read_bytes();manifest[name]=hashlib.sha256(raw).hexdigest();entry=tarfile.TarInfo(name);entry.size=len(raw);archive.addfile(entry,io.BytesIO(raw))
for name,value in [('original.json',original),('manifest.json',manifest)]:(stage/name).write_bytes((json.dumps(value,indent=2)+'\n').encode())
print('Prepared',len(manifest),'wide-batch candidate source identities')
