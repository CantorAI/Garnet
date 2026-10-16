"""Independent coverage, route admission, canonical-byte and CPU-ABBA audit."""
import hashlib,json,math,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text());trials=[]
for name,p in proof['files'].items():
    f=root/name;assert f.resolve().is_relative_to(root) and f.stat().st_size==p['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==p['sha256']
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0;j=root/'evidence/logs'/trial/'jobs'
    summary=json.loads((j/'summary.json').read_text());compat=json.loads((j/'compatibility/verification.json').read_text())
    assert compat['pillow']==summary['pillow']=='12.3.0'
    assert compat['comparisons']==len(compat['checks'])==960 and compat['all_complete_bytes_match']
    assert compat['parallel_calls']==96 and compat['fallback_calls']==864
    observed={(tuple(v['size']),v['mode'],v['orientation'],v['batch'],v['temporal'],tuple(v['mean']),tuple(v['std'])) for v in compat['checks']}
    expected={(size,mode,orientation,batch,temporal,tuple(mean),tuple(std)) for size in ((1,1),(31,47),(64,96),(513,257),(1372,1818)) for mode in ('RGB','RGBA','L') for orientation in range(1,9) for batch in (1,2) for temporal in (1,2) for mean,std in (([.5]*3,[.5]*3),([.123,.234,.345],[.271,.382,.493]))}
    assert observed==expected and sum(v['route']=='parallel' for v in compat['checks'])==96
    assert all(v['route']=='fallback' or (v['size']==[1372,1818] and v['orientation']<=4) for v in compat['checks'])
    assert json.loads((j/'compatibility-status.json').read_text())['exit_code']==0
    for name,sha in summary['source_sha256'].items():assert hashlib.sha256((j/'source-snapshot'/name).read_bytes()).hexdigest()==sha
    records=summary['records'];assert len(records)==192 and len({v['rgb_sha256'] for v in records})==1
    assert all(math.isfinite(v['elapsed_ms']) and v['elapsed_ms']>0 and v['route']==('parallel' if v['candidate'] else 'control') for v in records)
    assert len(summary['warmups'])==2 and all(v['seconds']>=.25 and v['executions']>=2 for v in summary['warmups'])
    for v in records:assert v['patch_sha256']==summary['cases'][str(v['batch'])]['image_metadata']['patch_sha256']
    for batch in (1,2):
        for repeat in (0,1):
            for window,policy in enumerate((False,True,True,False)):
                values=[v for v in records if v['batch']==batch and v['repeat']==repeat and v['window']==window]
                assert len(values)==12 and [v['sample'] for v in values]==list(range(12)) and all(v['candidate']==policy for v in values)
        p=next(v for v in summary['results'] if v['batch']==batch)
        for policy,key in ((False,'control_ms'),(True,'candidate_ms')):assert abs(p[key]-statistics.mean(v['elapsed_ms'] for v in records if v['batch']==batch and v['candidate']==policy))<1e-12
        assert abs(p['fraction']-p['candidate_ms']/p['control_ms'])<1e-12
    trials.append(dict(trial=trial,comparisons=960,parallel_compatibility_calls=96,real_image_samples=192,results=summary['results']))
result=dict(verified_files=len(proof['files']),trials=trials,scope='CPU preparation only; fresh full image/resize/patch/write/hash; complete RGB and FP32 bytes match; no inference or vLLM gain claimed',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
