"""Describe frozen c186 control clocks; no new performance run or causal attribution."""
from pathlib import Path
import hashlib,json,math,statistics,sys
base=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]).resolve()
proof=json.loads((base/'evidence/manifest.json').read_text());audit=json.loads((base/'verification.json').read_text())
assert audit['full_output_requests']==72 and audit['archive_sha256']=='1861996b1455f4a6aa7e81fb89bac30a2be2526d359eb332cc974d31aa6e520e'
root=base/'evidence';jobs=root/'logs/ndarray-selective-native-20261016/jobs';profiles=[];inputs={}
def read(path):
    raw=path.read_bytes();relative=path.relative_to(base).as_posix();info=proof['files'][relative]
    assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256'];inputs[relative]=info['sha256'];return raw.decode()
for folder in sorted(jobs.glob('Qwen*-b*-attempt*')):
    if not folder.is_dir():continue
    meta=json.loads(read(folder/'metadata.json'))
    assert meta['library_sha256']['control']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
    rows=[]
    for log in sorted(folder.glob('*control*/output.log')):
        records=[json.loads(line.split(' ',1)[1]) for line in read(log).splitlines() if line.startswith(('matched_frame_result ','matched_result '))]
        assert len(records)==3 and {r['request'] for r in records}=={0,1,2}
        rows.extend(r for r in records if r['request']>0)
    assert len(rows)==4
    model=next(m for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') if folder.name.startswith(m+'-b'))
    batch=next(b for b in (1,2) if folder.name.startswith(model+'-b'+str(b)+'-'))
    seconds=('ttft_seconds','vision_seconds','assembly_seconds','prefill_wall_seconds')
    ms={key.removesuffix('_seconds')+'_ms':statistics.mean(r[key] for r in rows)*1000 for key in seconds if all(key in r for r in rows)}
    prefill={key:statistics.mean(r['prefill_profile'][key] for r in rows) for key in ('prepare_ms','encode_ms','wait_ms','gpu_ms')}
    assert all(math.isfinite(v) and v>=0 for v in [*ms.values(),*prefill.values()])
    observed=next(p for p in audit['profiles'] if (p['model'],p['batch'])==(model,batch))
    assert abs(ms['ttft_ms']-observed['means']['control']['ttft_ms'])<1e-9
    profiles.append(dict(model=model,batch=batch,warm_requests=4,wall_intervals_ms=ms,prefill_profile_ms=prefill))
assert len(profiles)==6
result=dict(source_archive_sha256=audit['archive_sha256'],source_audit_sha256=hashlib.sha256((base/'verification.json').read_bytes()).hexdigest(),raw_file_sha256=inputs,profiles=profiles,
 scope='Derived descriptions of frozen retained-control warm clocks only; no new execution, causal isolation, latency prediction, current-candidate gain or fresh vLLM ratio. GPU span overlaps CPU encode/wait and must never be added to them. Vision/assembly/prefill are worker intervals; end-to-end TTFT also spans frontend/IPC, without isolated attribution.',full_goal_met=False)
destination.write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(dict(profiles=len(profiles),raw_files_verified=len(inputs),full_goal_met=False)))
