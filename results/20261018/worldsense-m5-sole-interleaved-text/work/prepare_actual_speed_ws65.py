"""Prepare unchanged original20-call screen against actual selected WS10."""
from pathlib import Path
import ast,json,hashlib,subprocess,sys
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');p=b/'work/worldsense-sole-interleaved-text';w=p/'production-speed';old=b/'work/worldsense-isolated-primary/production-speed';slug='worldsense-sole-interleaved-text';r=Path('/Users/shawn/CantorAI')
assert json.loads((p/'resource-independent-audit.json').read_text())['original_resource_thresholds_pass']
assert json.loads((p/'functional-smoke/functional-verification.json').read_text())['actualWorldSense_Galaxy_native_candidate_functional_pass']
for name in ['prepare.py','run_stage.py']:(w/name).write_text((old/name).read_text().replace('worldsense-isolated-primary',slug))
q=w/'prepare.py';t=q.read_text();needle='.replace("str(w/\'wait_cantor.py\')","str(w.parent/\'wait_cantor.py\')")';assert t.count(needle)==1;t=t.replace(needle,needle+'.replace("str(w/\'restore_normal.py\')","str(w.parent/\'restore_normal.py\')")');ast.parse(t);q.write_text(t)
q=w/'run_stage.py';t=q.read_text();needle=" subprocess.run([sys.executable,str(w/'analyze.py')],check=True)";assert t.count(needle)==1;t=t.replace(needle," identity=json.loads((w/'driver-identities-before-dispatch.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in identity['files'].items())\n"+needle);ast.parse(t);q.write_text(t)
(w/'prospective-stage-driver-note.json').write_text(json.dumps(dict(original20calls_onefirst_fourwarm_eachworker=True,actual_WS10_control=True,new_layout_route_validation_without_metric_gate_change=True,stage_input_driver_hashes_verified_after_workers=True,worker_dispatched=False,files={q.name:hashlib.sha256(q.read_bytes()).hexdigest()for q in [w/'prepare.py',w/'run_stage.py',w/'analyze.py']}),indent=2)+'\n')
subprocess.run([sys.executable,str(p/'dependency_closure.py'),'pre-production-speed'],check=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
files={}
for model,assets in json.loads((p/'model-assets.json').read_text()).items():
 for rel,info in assets.items():
  q=r/'out/models'/model/rel;assert q.stat().st_size==info['bytes']and sha(q)==info['sha256'];files[str(q)]=info
(w/'learned-assets-before-dispatch.json').write_text(json.dumps(files,indent=2)+'\n');print('Original actualWS10 screen prepared; dependencies/models rechecked; no dispatch')
