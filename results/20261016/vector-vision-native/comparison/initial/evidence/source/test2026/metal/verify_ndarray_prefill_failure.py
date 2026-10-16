"""Narrow source/command audit of the archived native logical-shape assertion failure."""
from pathlib import Path
import hashlib,json,runpy,sys
base=Path(sys.argv[1]).resolve();proof=json.loads((base/'evidence/manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base)
    assert p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert set(proof['jobs'])=={'ndarray-prefill-20261015'}
trial,status=next(iter(proof['jobs'].items()));assert status['exit_code']==1
out=base/'evidence/logs'/trial/'jobs';meta=json.loads((out/'metadata.json').read_text());snap=out/'source-snapshot'
for name,sha in meta['source_sha256'].items():assert digest(snap/name)==sha
for name,sha in meta['sdk_headers'].items():assert digest(out/name)==sha
assert 'Apple M4' in meta['hardware']['machdep.cpu.brand_string'] and int(meta['hardware']['hw.memsize'])==16*2**30
factory=runpy.run_path(str(snap/'test2026/metal/make_ndarray_prefill.py'))
for file,key,operation,original in [
    ('metal_context_ndarray_prefill.mm','context_generated_sha256','context','src/backends/metal/metal_context.mm'),
    ('metal_context.h','private_header_sha256','header','src/backends/metal/metal_context.h'),
    ('ndarray_prefill_gpu.mm','harness_generated_sha256','harness','test2026/metal/packed_body_prefill_gpu.mm')]:
    assert (out/file).read_bytes()==factory[operation]((snap/original).read_text()).encode()
    assert digest(out/file)==meta[key]
stages=json.loads((out/'statuses.json').read_text())
assert [(v['name'],v['exit_code']) for v in stages]==[('compile',0),('Qwen3-0.6B-gpu',1)]
command=stages[0]['command'];binary=Path(command[command.index('-o')+1]).name
assert proof['omitted']['logs/'+trial+'/jobs/'+binary]['sha256']==meta['binary_sha256']
for v in stages:
    monitor=json.loads((out/(v['name']+'-monitor.json')).read_text());assert monitor and all(q['guard_passed'] for q in monitor)
assert (out/'Qwen3-0.6B-gpu.log').read_text().strip()=='ndarray_prefill_failed: NDArray transposed view failed'
assert not (out/'summary.json').exists()
assert set(meta['assets'])=={'Qwen3-0.6B'}
assets=meta['assets']['Qwen3-0.6B'];assert len(assets['cases'])==7
assert proof['omitted']['models/Qwen3-0.6B/model.safetensors']['sha256']==assets['checkpoint_sha256']
for c in assets['cases']:
    assert proof['omitted']['logs/decoder-codec-gpu-20261013/jobs/Qwen3-0.6B-assets/'+c['name']+'/raw.bin']['sha256']==c['sha256']['raw.bin']
result=dict(verified_files=len(proof['files']),failure='NDArray transposed view failed',compile_passed=True,
    emitted_numerical_records=0,performance_claim=False,scope='Frozen failure source/command/guard/asset provenance only; no trained numerical, speed or complete-model proof.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
