"""CPU fresh-frame ABBA timings with exact patch/RGB compatibility checks."""
import ast,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from frame_frontend_candidates import prepare_borrowed,prepare_cv_frame,png_fast_eligible,simple_webp_eligible
from frame_preprocessing import prepare_frame
r=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'guard','exec'),globals())
def check():
    guard()
    for line in snapshot().splitlines():
        f=line.split()
        if f and f[-1].lower().endswith(('_gpu','_probe')):raise RuntimeError('Other GPU probe PID '+f[0])
model=r/'models/Qwen3.5-0.8B';config=json.loads((model/'config.json').read_text())['vision_config'];processor=json.loads((model/'preprocessor_config.json').read_text());records=[]
cv2.ocl.setUseOpenCL(False)
metadata=dict(scope=__doc__,versions=dict(python=sys.version,numpy=np.__version__,pillow=Image.__version__,opencv=cv2.__version__),sources={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in (Path(__file__),s/'frame_frontend_candidates.py',s/'frame_preprocessing.py')})
for batch in (1,2):
    workload=json.loads((r/f'logs/joined-prefill-native-20261013/jobs/Qwen3.5-0.8B-b{batch}-attempt0/Qwen3.5-0.8B-b{batch}-case.json').read_text());meta=workload['image_metadata'];image=meta['image']
    raw=Path(image).read_bytes();metadata['fast_image_eligible']=png_fast_eligible(raw) or simple_webp_eligible(raw)
    with Image.open(image) as opened:metadata['actual_source_format']=opened.format
    check()
    def run(policy):
        begin=time.perf_counter()
        if policy=='opencv':patches,rgb,hw=prepare_cv_frame(image,config,processor,batch)
        else:
            with Image.open(image) as opened:patches,rgb,hw=(prepare_frame if policy=='control' else prepare_borrowed)(opened,config,processor,batch)
        path=output/'frame.f32';rgbpath=output/'frame.rgb';patches.tofile(path);rgbpath.write_bytes(rgb)
        digest=hashlib.sha256(path.read_bytes()).hexdigest();elapsed=(time.perf_counter()-begin)*1000
        assert digest==meta['patch_sha256'] and list(hw)==meta['resized_hw'];return elapsed,hashlib.sha256(rgb).hexdigest()
    for candidate in ('borrowed','opencv'):
        for round in range(2):
            for policy in ('control',candidate,candidate,'control'):
                check();run(policy);samples=[]
                for trial in range(12):
                    ms,rgb_hash=run(policy);samples.append(ms)
                    records.append(dict(batch=batch,candidate=candidate,round=round,policy=policy,trial=trial,elapsed_ms=ms,rgb_sha256=rgb_hash,patch_sha256=meta['patch_sha256']))
                print(json.dumps(dict(batch=batch,candidate=candidate,round=round,policy=policy,median_ms=statistics.median(samples))),flush=True)
    check()
results=[]
for batch in (1,2):
    for candidate in ('borrowed','opencv'):
        control=statistics.median(v['elapsed_ms'] for v in records if v['batch']==batch and v['candidate']==candidate and v['policy']=='control');value=statistics.median(v['elapsed_ms'] for v in records if v['batch']==batch and v['candidate']==candidate and v['policy']==candidate)
        results.append(dict(batch=batch,candidate=candidate,control_ms=control,candidate_ms=value,frontend_speedup=control/value))
assert len(records)==384 and len({v['rgb_sha256'] for v in records})==1
(output/'summary.json').write_text(json.dumps(dict(metadata=metadata,records=records,results=results,full_inference_claim=False),indent=2))
print(json.dumps(dict(results=results,checks=len(records),opencl_enabled_after_run=cv2.ocl.useOpenCL())),flush=True)
