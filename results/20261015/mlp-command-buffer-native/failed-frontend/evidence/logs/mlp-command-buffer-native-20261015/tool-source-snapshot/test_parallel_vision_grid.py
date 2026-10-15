"""Same canonical real-image patches, both trained towers, unchanged float bounds."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();candidate=Path(sys.argv[3]).resolve();out.mkdir(exist_ok=False)
worker=Path(__file__).with_name('run_parallel_vision_grid.py');reports=[]
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
for batch in (1,2):
    old=root/'logs/threaded-striped-inference-confirm-20261015/jobs'/('Qwen3.5-0.8B-b'+str(batch)+'-0')
    case=json.loads((old/'case.json').read_text());patch=old/'garnet-selected-0/protocol/frame-0.f32'
    assert hashlib.sha256(patch.read_bytes()).hexdigest()==case['image_metadata']['patch_sha256']
    config=json.loads((root/'models/Qwen3.5-0.8B/config.json').read_text())['vision_config'];outputs={}
    for policy,build in (('control',root/'build-global-half-alpha'),('candidate',candidate)):
        destination=out/(policy+'-b'+str(batch)+'.json');fixture=out/(policy+'-b'+str(batch)+'-fixture.json')
        meta=case['image_metadata'];f=dict(checkpoint=str(root/'models/Qwen3.5-0.8B/model.safetensors'),config=config,grid_h=meta['grid_thw'][1],grid_w=meta['grid_thw'][2],batch=batch,patch_file=str(patch),output_file=str(destination))
        fixture.write_text(json.dumps(f));subprocess.run([runtime,str(worker),str(fixture),str(build),str(root/'source')],check=True)
        result=json.loads(destination.read_text());assert result['enabled']==(policy=='candidate')
        outputs[policy]=np.asarray(result['values'],dtype=np.float64)
    a=outputs['control'];b=outputs['candidate'];assert a.shape==b.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    bound=5e-6+5e-5*np.abs(a);error=np.abs(b-a);assert np.all(error<=bound),dict(batch=batch,worst_ratio=float(np.max(error/bound)))
    reports.append(dict(batch=batch,patch_sha256=case['image_metadata']['patch_sha256'],values=a.size,max_abs_error=float(error.max()),worst_ratio=float(np.max(error/bound)),numerically_equal=bool(np.array_equal(a,b))))
(out/'summary.json').write_text(json.dumps(dict(profiles=reports,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),worker)}),indent=2)+'\n')
print('Actual trained 216-token image-grid embeddings passed original bounds at B1/B2',flush=True)
