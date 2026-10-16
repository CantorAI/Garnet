"""Preserve summary-parser failure; validate raw completeness without admitting calibration ratios."""
from pathlib import Path
import hashlib,json,math,runpy,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert json.loads((base/'archive-verification.json').read_text())['archive_sha256']=='7af5fad7078e3869d313bf12d2725b802d19969d01557feb6415a2f2219ac409'
trial='vector-vision-calibration-20261016';assert proof['jobs'][trial]['exit_code']==1
out=root/'logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text());snap=out/'source-snapshot'
assert meta['selected_variants']==[0] and meta['timing_design']==dict(warmup_seconds=2.0,abba_rounds=10,samples_per_window=5,common_alternating_warmup=True,original_numerical_gates=True)
for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha
sys.path.insert(0,str(snap/'test2026/metal'))
factory=runpy.run_path(str(snap/'test2026/metal/make_calibrated_vector_vision_context.py'))
shader,cpp=factory['generate']((snap/'src/backends/metal/metal_context.mm').read_text(),(snap/'test2026/metal/vision_fused_context_gpu.mm').read_text(),(snap/'src/backends/metal/serial_vision_prefetch_shader.h').read_text())
assert (out/'vector-vision.metal').read_bytes()==shader.encode() and (out/'vector_vision_context_gpu.mm').read_bytes()==cpp.encode()
assert digest(out/'vector-vision.metal')==meta['shader_sha256'] and digest(out/'vector_vision_context_gpu.mm')==meta['harness_sha256']
stages=json.loads((out/'statuses.json').read_text());assert [v['name'] for v in stages]==['compile','variant-0']
assert all(v['exit_code']==0 for v in stages)
for stage in stages:
    samples=json.loads((out/(stage['name']+'-monitor.json')).read_text());assert samples and all(v['guard_passed'] for v in samples)
assert proof['omitted']['logs/'+trial+'/jobs/vector_vision_context_gpu']['sha256']==meta['binary_sha256']
raw=[json.loads(line) for line in (out/'variant-0.log').read_text().splitlines() if line.startswith('{')]
checks=[v for v in raw if v['kind']=='check'];samples=[v for v in raw if v['kind']=='samples']
assert len(checks)==24 and all(c['variant']==0 and c['generations']==4 and c['passed'] and c['bits_exact'] and c['max_abs_error']==0 and 0<=c['fp64_worst_ratio']<=1 for c in checks)
keys=lambda c:tuple(c[k] for k in ('batch','tokens','heads','dim'))
assert {keys(c) for c in checks}=={(b,t,h,d) for b in (1,2) for t in (4,24,216) for h in (2,12) for d in (8,64)}
total=0
for check in checks:
    windows=[v for v in samples if keys(v)==keys(check)];assert len(windows)==40
    for rnd in range(10):assert [v['candidate'] for v in windows if v['round']==rnd]==[False,True,True,False]
    assert all(len(v[k])==5 and all(math.isfinite(x) and x>0 for x in v[k]) for v in windows for k in ('gpu_ms','wall_ms'))
    total+=sum(len(v['gpu_ms']) for v in windows)
assert total==4800 and len(samples)==960 and not (out/'summary.json').exists()
expanded=(out.parent/'jobs.calibrated.executed.py').read_text()
assert 'assert len(windows)==8' in expanded and 'len(v[k])==20' in expanded and 'for rnd in (0,1):' in expanded
assert 'round<10' in cpp and 'trial<5' in cpp
assert 'AssertionError' in (out.parent/'output.log').read_text()
result=dict(verified_files=len(proof['files']),overall_exit_code=1,gpu_stage_exit_code=0,numerical_records=24,
    captured_timed_submissions=total,parser_expected_windows=8,actual_windows=40,parser_expected_samples=20,actual_samples=5,
    calibration_ratios_admitted=False,scope=__doc__,full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
