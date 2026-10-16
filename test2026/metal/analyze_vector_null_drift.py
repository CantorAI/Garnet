"""Describe every failing null window; do not filter data or revise its failed screen."""
from pathlib import Path
import hashlib,json,statistics,sys
base=Path(sys.argv[1]).resolve();folder=base/'initial/evidence/logs/vector-vision-context-20261016/jobs'
audit=json.loads((base/'initial/verification.json').read_text());assert audit['shape_variant_checks']==72 and audit['failed_admissions']==0
assert json.loads((base/'initial/archive-verification.json').read_text())['archive_sha256']=='e2ec3e3531b6503e68c4d30b6be1eabe1df18e29ca1959346caf39c20dd4cbc9'
cpp=(folder/'vector_vision_context_gpu.mm').read_text()
assert 'auto candidatePipeline=variant==0?kernels[1]:kernels[variant+1];' in cpp
assert 'candidateWidth[]={1,2,4}' in cpp
records=[json.loads(line) for line in (folder/'variant-0.log').read_text().splitlines() if line.startswith('{')]
windows=[r for r in records if r['kind']=='samples' and (r['batch'],r['tokens'],r['heads'],r['dim'])==(1,216,12,64)]
assert len(windows)==8
result=[]
for position,r in enumerate(windows):
    assert r['round']==position//4 and r['candidate']==(position%4 in (1,2))
    stats={k:dict(mean=statistics.mean(r[k]),minimum=min(r[k]),maximum=max(r[k]),samples=len(r[k])) for k in ('gpu_ms','wall_ms')}
    result.append(dict(round=r['round'],position=position%4,candidate=r['candidate'],**stats))
analysis=dict(windows=result,compiled_harness_sha256=hashlib.sha256(cpp.encode()).hexdigest(),
    observation='Null B1 GPU means change from ~1.58 ms to ~0.79 ms across the recorded windows. Both arms bind the same scalar pipeline and dispatch geometry; no samples are removed.',
    limitation='No frequency/power counters were captured; timing drift does not prove a particular clock/thermal cause. Initial cohort remains uncalibrated and rejected; no unchanged repeat or native admission.',
    proposed_next_step='Separate prospective calibration-only study with fixed longer conditioning and shorter ABBA windows before any new candidate timing.',full_goal_met=False)
(base/'null-drift-analysis.json').write_bytes((json.dumps(analysis,indent=2)+'\n').encode());print(json.dumps(analysis))
