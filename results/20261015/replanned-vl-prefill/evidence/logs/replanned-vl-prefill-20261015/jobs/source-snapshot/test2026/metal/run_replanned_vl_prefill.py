"""Independent trained chunk references before two sequential native profile trials."""
from pathlib import Path
source=Path(__file__).with_name('run_joined_prefill_checks.py').read_text().split("assert not build.exists()",1)[0]
exec(compile(source,__file__,'exec'))
import shutil
build=r/'build-interleaved-screen-index';runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
manifest=json.loads((r/'interleaved-head-staging/manifest.json').read_text())
def unchanged():
    assert all(digest(r/'source'/n)==sha for n,sha in manifest.items())
    assert digest(build/'libgarnet.dylib')==metadata['library_sha256']
    assert digest(Path(runtime))==metadata['runtime_sha256']
    assert all(digest(r/'source'/n)==sha for n,sha in metadata['source_sha256'].items())
metadata=dict(scope='Retuned internal VL B2 chunks on the same native library; logical inputs/caps and original bounds unchanged; no fresh vLLM ratio',library_sha256=digest(build/'libgarnet.dylib'),runtime_sha256=digest(Path(runtime)),source_manifest=manifest,source_sha256={},checkpoint_sha256=digest(r/'models/Qwen3.5-0.8B/model.safetensors'),config_sha256=digest(r/'models/Qwen3.5-0.8B/config.json'))
import numpy as np
metadata['versions']=dict(python=sys.version,numpy=np.__version__,macos=subprocess.check_output(['/usr/bin/sw_vers'],text=True))
metadata['hardware']={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')}
helpers=('run_replanned_vl_prefill.py','test_replanned_vl_prefill_reference.py','test_qwen35_extended_prefill_reference.py','run_chunk_graph.py','compare_replanned_vl_prefill.py','run_joined_prefill_checks.py','compare_vllm_metal.py','compare_interleaved_head.py','compare_small_seed.py','compare_serial_vision.py','compare_screened_prefill_reset.py','compare_screened_head.py','compare_parallel_vision.py','compare_vision_global_alpha.py','run_threaded_striped_frame_process.py','run_matched_frame_process.py','frame_worker_garnet.py','frame_threaded_striped.py','frame_webp_threaded.py','frame_preprocessing_striped.py','frame_preprocessing.py')
helpers+=('compare_dirty_prefix_reset.py','run_screened_head_profiles.py','run_screened_head_session_reference.py','run_screened_prefill_reset_profiles.py','run_serial_vision_profiles.py','run_webp_thread_frame_process.py')
paths={s/name for name in helpers}|{r/'source'/name for name in manifest}|set((r/'source/xModel').rglob('*.py'))|{r/'source/test2026/pi5b/reference_decoder.py'}
for p in sorted(paths):
    rel=str(p.relative_to(r/'source'));metadata['source_sha256'][rel]=digest(p);target=output/'source-snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
(output/'metadata.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode());unchanged()
run('trained-chunk-references',[sys.executable,str(s/'test_replanned_vl_prefill_reference.py'),runtime,str(build),str(r/'source'),str(r/'models/Qwen3.5-0.8B'),str(output/'reference')])
profiles=[]
for maximum in (16,64):
    unchanged();guard('chunk'+str(maximum)+'-before');assert shutil.disk_usage(r).free>2*2**30
    name='chunk'+str(maximum);folder=output/name
    command=[sys.executable,str(s/'compare_replanned_vl_prefill.py'),str(r),str(folder),str(maximum)]
    with (output/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
    statuses.append(dict(name=name,command=command,exit_code=result.returncode));(output/'statuses.json').write_bytes((json.dumps(statuses,indent=2)+'\n').encode());assert result.returncode==0,name
    group=json.loads((folder/'summary.json').read_text())['profiles'];assert len(group)==1
    profiles.append(dict(maximum=maximum,**group[0]));(output/'summary.json').write_bytes((json.dumps(dict(profiles=profiles),indent=2)+'\n').encode());guard(name+'-after')
print('Original trained bounds and both complete-output prefill replans passed',flush=True)
