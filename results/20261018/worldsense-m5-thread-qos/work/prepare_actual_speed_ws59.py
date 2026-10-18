"""Prepare original20-call actualWS10 screen; never dispatch benchmarking."""
from pathlib import Path
import ast,json,hashlib
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');p=root/'work/worldsense-thread-qos';w=p/'production-speed';old=p.parent/'worldsense-isolated-primary/production-speed';slug='worldsense-thread-qos'
assert json.loads((p/'resource-analysis.json').read_text())['passed'];assert json.loads((p/'cohort-native-identity-audit.json').read_text())['exact_two_child_calls_no_hidden_retry']
for name in ['prepare.py','run_stage.py']:(w/name).write_text((old/name).read_text().replace('worldsense-isolated-primary',slug))
p0=w/'prepare.py';t=p0.read_text();needle='.replace("str(w/\'wait_cantor.py\')","str(w.parent/\'wait_cantor.py\')")';assert t.count(needle)==1;t=t.replace(needle,needle+'.replace("str(w/\'restore_normal.py\')","str(w.parent/\'restore_normal.py\')")');ast.parse(t);p0.write_text(t)
p0=w/'prepare.py';t=p0.read_text();needle="(w/'environment.json').write_text(json.dumps(env,indent=2))";assert t.count(needle)==1;t=t.replace(needle,"env['native_text_worker_raw_library_sha256']=sha(r/'out/build/garnet-metal-worldsense-thread-qos-off/libgarnet.dylib')\n"+needle);ast.parse(t);p0.write_text(t)
# Additional sequence audit changes no original metric/golden/threshold code.
t=(p/'audit_native_cohort.py').read_text();t=t.replace("assert len(calls)==4","assert len(calls)==10").replace('seen_remote==2 and last_count==2','seen_remote==5 and last_count==5').replace("r/('out/build/garnet-metal-worldsense-thread-qos-'+mode)/'libgarnet.dylib'","(r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib')if mode=='off'else r/('out/build/garnet-metal-worldsense-thread-qos-'+mode)/'libgarnet.dylib'").replace('resource_scope_only=True','actual_WS10_speed_scope_only=True').replace('exact_two_child_calls_no_hidden_retry=True','exact_five_child_calls_no_hidden_retry=True').replace('Actual resource cohort','Actual WS10 speed cohort')
# Legacy WS10 has no native_gateway just like resource OFF; measured candidate
# must retain one stable owned child across all five real Skill calls.
raw_guard="raw=r/'out/build/garnet-metal-worldsense-thread-qos-off/libgarnet.dylib';assert hashlib.sha256(raw.read_bytes()).hexdigest()==environment['native_text_worker_raw_library_sha256']\n"
needle="(w/'cohort-native-identity-audit.json').write_text";assert t.count(needle)==1;t=t.replace(needle,raw_guard+needle)
(w/'audit_native_cohort.py').write_text(t)
p0=w/'run_stage.py';t=p0.read_text();needle=" subprocess.run([sys.executable,str(w/'analyze.py')],check=True)";assert t.count(needle)==1;t=t.replace(needle," subprocess.run([sys.executable,str(w/'audit_native_cohort.py')],check=True)\n"+needle);ast.parse(t);p0.write_text(t)
for p0 in [w/'prepare.py',w/'run_stage.py',w/'audit_native_cohort.py']:ast.parse(p0.read_text())
(w/'prospective-stage-driver-note.json').write_text(json.dumps(dict(original20calls_onefirst_fourwarm_eachworker=True,actual_WS10_control=True,parent_and_child_costs_counted=True,stable_child_five_calls_no_retry_additional_guard=True,numeric_golden_metrics_threshold_AST_unchanged=True,worker_dispatched=False,files={p0.name:hashlib.sha256(p0.read_bytes()).hexdigest()for p0 in [w/'prepare.py',w/'run_stage.py',w/'audit_native_cohort.py',w/'analyze.py']}),indent=2)+'\n')
print('ActualWS10 original20call stage drivers prepared; no dispatch')
