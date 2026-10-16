"""Archive terminal attention-layout experiments, source and compiler definitions."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]).resolve();files={}
required={'attention-prefix-probe-20261013':1,'attention-prefix-probe-retry1-20261013':0,'attention-prefix-probe-confirm-20261013':0,'attention-prefix-checks-20261013':0,'attention-prefix-inference-20261013':1,'attention-prefix-inference-confirm-20261013':1,'attention-prefix-inference-retry1-20261013':1,'attention-prefix-inference-confirm-retry1-20261013':1,'attention-prefix-inference-retry2-20261013':0,'attention-prefix-inference-confirm-retry2-20261013':0,'attention-prefix-guard-checks-20261013':0}
for name,code in required.items():assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==code,name
for folder in sorted((root/'logs').glob('attention-prefix-*')):
    assert (folder/'exit.json').is_file(),'Archive only terminal jobs: '+str(folder)
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.txt','.log','.metal','.py'):
            files['evidence/logs/'+str(path.relative_to(root/'logs'))]=path
scripts=root/'source/test2026/metal'
names=('make_attention_prefix_shader.py','attention_prefix_gpu.mm','run_attention_prefix_probe.py','run_attention_prefix_checks.py','compare_attention_prefix.py','archive_attention_prefix.py','compare_vllm_metal.py','run_vision_fused_context_probe.py','run_joined_prefill_checks.py','compare_joined_prefill.py','run_after_owned_job.py','background_comparison.py','run_matched_garnet.py','run_matched_frame_process.py','frame_worker_garnet.py','frame_preprocessing.py','test_guard_process_exit.py','test_qwen35_extended_prefill_reference.py')
for name in names:files['evidence/source/test2026/metal/'+name]=scripts/name
for name in ('src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','cmake/GarnetMetal.cmake','test2026/pi5b/reference_decoder.py'):
    files['evidence/source/'+name]=root/'source'/name
for path in (root/'work/attention-prefix-original').iterdir():
    if path.is_file():files['evidence/original/'+path.name]=path
metadata=json.loads((root/'logs/attention-prefix-checks-20261013/jobs/metadata.json').read_text())
for name in metadata['source_sha256']:files['evidence/check-source/'+name]=root/'source'/name
for target in ('garnet','garnet_metal'):
    files['evidence/build/'+target+'-flags.make']=root/'build-attention-prefix/CMakeFiles'/(target+'.dir')/'flags.make'
manifest={}
with tarfile.open(destination,'w:gz') as archive:
    for name,path in files.items():
        data=path.read_bytes();manifest[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());archive.add(path,arcname=name,recursive=False)
    encoded=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/files.json');info.size=len(encoded);archive.addfile(info,io.BytesIO(encoded))
print(json.dumps(dict(files=len(manifest),archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))
