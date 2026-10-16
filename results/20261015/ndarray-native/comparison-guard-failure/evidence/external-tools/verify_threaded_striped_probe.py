"""Independently audit combined decode/resize parity and fresh CPU timing windows."""
import hashlib,json,math,statistics,sys
from collections import Counter
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text());trials=[]
for name,v in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
assert proof['library_sha256']=='3c86ac196227b3729477cd24704ea4cea4e796fe5f9933c7a5a20361a32804a3'
for trial,status in proof['jobs'].items():
    assert status['exit_code']==0;j=root/'evidence/logs'/trial/'jobs';s=json.loads((j/'summary.json').read_text());c=json.loads((j/'compatibility/summary.json').read_text());m=c['metadata']
    assert m['checks']==956 and m['fixtures']==len(c['records'])==239 and m['native_fixtures']==16
    assert m['parallel_resize_calls']==148 and m['fallback_resize_calls']==808
    expected=Counter(f'{w}x{h}-{mode}-{lossless}-{orientation}' for w,h in ((32,48),(128,160),(511,257),(1024,768)) for mode in ('RGB','RGBA','L') for lossless in (False,True) for orientation in (None,1,2,3,4,5,6,7,8))
    expected.update({'animated':4,'PNG-fallback':4,'JPEG-fallback':4,'BMP-fallback':4});expected.update('late-exif-'+str(i) for i in range(2,9))
    assert Counter(v['case'] for v in c['records'])==expected and sum(v['native_applied'] for v in c['records'])==16
    assert all(not v['native_applied'] or v['container_eligible'] for v in c['records'])
    assert m['library_sha256']==s['library_sha256']==proof['library_sha256'] and s['pillow']=='12.3.0'
    assert json.loads((j/'compatibility-status.json').read_text())['exit_code']==0
    for name,sha in s['source_sha256'].items():assert hashlib.sha256((j/'source-snapshot'/name).read_bytes()).hexdigest()==sha
    for name,sha in m['source_sha256'].items():assert hashlib.sha256((root/'evidence/source'/name).read_bytes()).hexdigest()==sha
    records=s['records'];assert len(records)==192 and len({v['rgb_sha256'] for v in records})==1
    assert len(s['warmups'])==2 and all(v['seconds']>=.25 and v['executions']>=2 for v in s['warmups'])
    for v in records:
        assert math.isfinite(v['elapsed_ms']) and v['elapsed_ms']>0
        assert v['decode_route']==('native-threaded' if v['candidate'] else 'pillow') and v['resize_route']==('parallel' if v['candidate'] else 'selected')
        assert v['patch_sha256']==s['cases'][str(v['batch'])]['image_metadata']['patch_sha256']
    for batch in (1,2):
        for repeat in (0,1):
            for window,policy in enumerate((False,True,True,False)):
                values=[v for v in records if v['batch']==batch and v['repeat']==repeat and v['window']==window]
                assert len(values)==12 and [v['sample'] for v in values]==list(range(12)) and all(v['candidate']==policy for v in values)
        p=next(v for v in s['results'] if v['batch']==batch)
        for policy,key in ((False,'control_ms'),(True,'candidate_ms')):assert abs(p[key]-statistics.mean(v['elapsed_ms'] for v in records if v['batch']==batch and v['candidate']==policy))<1e-12
        assert abs(p['fraction']-p['candidate_ms']/p['control_ms'])<1e-12
    trials.append(dict(trial=trial,compatibility_checks=956,parallel_resize_calls=148,frozen_image_samples=192,results=s['results']))
result=dict(verified_files=len(proof['files']),trials=trials,scope='CPU composition only, exact full RGB and FP32 patch bytes; no inference or vLLM speed claim',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
