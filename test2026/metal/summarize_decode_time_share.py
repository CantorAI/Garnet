"""Measure recorded decode clocks; GPU spans overlap waits and cannot be added."""
import hashlib,json,math,statistics,sys
from pathlib import Path
folder=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
outer=json.loads((folder/'summary.json').read_text());profiles=[];inputs={}
for p in outer['profiles']:
    path=folder/Path(p['evidence']).name/'summary.json'
    raw=path.read_bytes();inputs[path.relative_to(folder).as_posix()]=hashlib.sha256(raw).hexdigest()
    records=json.loads(raw)['records']
    warm=[r for r in records if r['build_policy']=='control' and r['request']>0]
    assert len(warm)==4
    phases=[];windows=[];native=[];external=[]
    for r in warm:
        assert r['decode_partial_submissions']==0
        count=r['decode_steps_after_first']
        assert count>0
        if 'decode_phases' in r:
            assert count==len(r['decode_phases'])==len(r['native_decode_ms'])==len(r['decode_step_seconds'])
            phases.extend(r['decode_phases']);native.extend(r['native_decode_ms'])
            external.extend(v*1000 for v in r['decode_step_seconds'])
        else:
            assert p['model']=='Qwen3.5-0.8B'
        windows.append(r['decode_window_seconds']*1000/count)
    means={key:statistics.mean(x[key] for x in phases) for key in ('prepare_ms','encode_ms','gpu_ms','wait_ms')} if phases else None
    assert means is None or all(math.isfinite(v) and v>=0 for v in means.values())
    wall=statistics.mean(windows)
    profiles.append(dict(model=p['model'],batch=p['batch'],warm_requests=4,decode_steps=sum(r['decode_steps_after_first'] for r in warm),
        wall_ms_per_step=wall,native_run_ms_per_step=statistics.mean(native) if native else None,
        individually_timed_step_ms=statistics.mean(external) if external else None,phase_means_ms=means,
        gpu_span_fraction_of_decode_wall=means['gpu_ms']/wall if means else None,
        host_encode_fraction_of_decode_wall=means['encode_ms']/wall if means else None,
        observation='Single-command decode; clocks overlap; no isolated matrix or bandwidth attribution' if means else 'VL worker did not retain individual decode phase clocks; GPU/host shares unavailable'))
assert len(profiles)==6
result=dict(scope='Recorded native control only; no fresh comparator ratio or hypothetical speed claim',
    sources_sha256=inputs,profiles=profiles)
target.write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(profiles,indent=2))
