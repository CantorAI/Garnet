"""Prepare original20-call WS10 screen only after resource gate passes."""
from pathlib import Path
import ast,json,hashlib
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');p=root/'work/worldsense-dense-prefetch';w=p/'production-speed';old=p.parent/'worldsense-isolated-primary/production-speed';slug='worldsense-dense-prefetch'
assert json.loads((p/'resource-analysis.json').read_text())['passed']
assert json.loads((p/'functional-smoke/functional-verification.json').read_text())['actualWorldSense_Galaxy_native_candidate_functional_pass']
for name in ['prepare.py','run_stage.py']:(w/name).write_text((old/name).read_text().replace('worldsense-isolated-primary',slug))
q=w/'prepare.py';t=q.read_text();needle='.replace("str(w/\'wait_cantor.py\')","str(w.parent/\'wait_cantor.py\')")';assert t.count(needle)==1
t=t.replace(needle,needle+'.replace("str(w/\'restore_normal.py\')","str(w.parent/\'restore_normal.py\')")');ast.parse(t);q.write_text(t)
q=w/'run_stage.py';t=q.read_text();needle=" subprocess.run([sys.executable,str(w/'analyze.py')],check=True)";assert t.count(needle)==1
t=t.replace(needle," identity=json.loads((w/'driver-identities-before-dispatch.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in identity['files'].items())\n"+needle);ast.parse(t);q.write_text(t)
(w/'prospective-stage-driver-note.json').write_text(json.dumps(dict(original20calls_onefirst_fourwarm_eachworker=True,actual_WS10_control=True,new_prefetch_route_validation_without_metric_gate_change=True,stage_input_driver_hashes_verified_after_workers=True,worker_dispatched=False,files={q.name:hashlib.sha256(q.read_bytes()).hexdigest()for q in [w/'prepare.py',w/'run_stage.py',w/'analyze.py']}),indent=2)+'\n')
print('Original actualWS10screen prepared with new route guards; no dispatch')
