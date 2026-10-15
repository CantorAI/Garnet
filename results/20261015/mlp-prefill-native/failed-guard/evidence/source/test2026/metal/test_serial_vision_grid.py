"""Actual trained full image-grid equality and small-grid fallback against control."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();candidate=Path(sys.argv[3]).resolve();out.mkdir(exist_ok=False)
worker=Path(__file__).with_name('run_serial_vision_grid.py');runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3';reports=[]
for batch in (1,2):
    prior=root/'logs/threaded-striped-inference-confirm-20261015/jobs'/('Qwen3.5-0.8B-b'+str(batch)+'-0');case=json.loads((prior/'case.json').read_text());meta=case['image_metadata']
    original=prior/'garnet-selected-0/protocol/frame-0.f32';assert hashlib.sha256(original.read_bytes()).hexdigest()==meta['patch_sha256']
    config=json.loads((root/'models/Qwen3.5-0.8B/config.json').read_text())['vision_config']
    for label in ('small','full'):
        if label=='full':patch=original;h,w=meta['grid_thw'][1:];expected=config['depth']
        else:
            patch=out/('small-b'+str(batch)+'.f32');np.fromfile(original,dtype='<f4').reshape(batch,216,1536)[:,:4].copy().tofile(patch);h=w=2;expected=0
        results={}
        for policy,build in (('control',root/'build-screened-prefill-reset'),('candidate',candidate)):
            prefix=policy+'-'+label+'-b'+str(batch);destination=out/(prefix+'.json');fixture=out/(prefix+'-fixture.json')
            f=dict(checkpoint=str(root/'models/Qwen3.5-0.8B/model.safetensors'),config=config,grid_h=h,grid_w=w,batch=batch,patch_file=str(patch),output_file=str(destination),expected_prefetch_dispatches=expected)
            fixture.write_bytes((json.dumps(f,indent=2)+'\n').encode());subprocess.run([runtime,str(worker),str(fixture),str(build),str(root/'source')],check=True)
            result=json.loads(destination.read_text());assert result['enabled']==(policy=='candidate') and result['counts']==[expected if policy=='candidate' else 0]*2
            results[policy]=np.asarray(result['values'],dtype='<f4')
        a,b=results['control'],results['candidate'];assert a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        assert np.array_equal(a.view('<u4'),b.view('<u4')),'Trained vision embedding bits changed'
        reports.append(dict(batch=batch,profile=label,patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),values=a.size,bitwise_equal=True,embedding_sha256=hashlib.sha256(a.tobytes()).hexdigest(),expected_candidate_dispatches=expected,fresh_encodes_per_policy=2))
(out/'summary.json').write_bytes((json.dumps(dict(profiles=reports,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),worker)}),indent=2)+'\n').encode())
print('Trained full-grid bit identity, small-grid fallback and fresh/closed counters passed B1/B2',flush=True)
