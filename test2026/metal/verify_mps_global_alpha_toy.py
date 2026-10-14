"""Verify toy execution; do not elevate CPU eligibility to real-model support."""
import hashlib,json,sys
from pathlib import Path
base=Path(sys.argv[1]);files=json.loads((base/'evidence/files.json').read_text())
for name,item in files.items():
 raw=(base/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
j=base/'evidence/logs/mps-global-alpha-toy-20261014/jobs';meta=json.loads((j/'metadata.json').read_text())
assert json.loads((j.parent/'exit.json').read_text())['exit_code']==0
stages=json.loads((j/'statuses.json').read_text());assert len(stages)==3 and all(s['exit_code']==0 for s in stages)
for name,sha in meta['source_sha256'].items():assert hashlib.sha256((j/'source-snapshot'/name).read_bytes()).hexdigest()==sha
assert hashlib.sha256((j/'global_alpha_gpu.mm').read_bytes()).hexdigest()==meta['probe_sha256']
rows=[json.loads(line.split(' ',1)[1]) for line in (j/'gpu.log').read_text().splitlines() if line.startswith('mps_global_alpha_probe ')];assert len(rows)==2
for r in rows:assert r['supported'] and r['numerical_pass'] and r['weight_bits_roundtrip_exact'] and r['fp32_activations'] and r['fp32_output'] and not r['separate_rescale_dispatch'] and r['alpha']==2**-16
audit=json.loads((base/'evidence/feasibility.json').read_text());assert not audit['gpu_execution_performed'] and len(audit['matrices'])==22 and audit['eligible_count']==20
assert all(m['restored_bits_exact'] for m in audit['matrices'] if m['eligible'])
result=dict(verified_files=len(files),toy_shapes=2,toy_max_abs_error=max(r['max_abs_error'] for r in rows),sampled_body_matrices=22,lossless_global_scale_eligible=20,real_trained_gpu_projections_tested=False,full_model_gain_established=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
