from pathlib import Path
import json,shutil
w=Path(__file__).resolve().parent;p=w.parent
assert json.loads((p/'resource-analysis.json').read_text())['passed']
assert json.loads((p/'restoration-verification.json').read_text())['benchmark_valid_request_http404']
assert not (w/'control-1.json').exists() and not (w/'candidate-1.json').exists()
for name in ['environment.json','starting-source-manifest.json']:shutil.copy2(p/name,w/name)
env=json.loads((w/'environment.json').read_text());env['libraries']['off']=env['libraries']['primary'];env['comparison_scope']='Only PRIMARY_VECTOR_PREFILL differs; both full-primary and eager';(w/'environment.json').write_text(json.dumps(env,indent=2))
c=json.loads((p/'cases.json').read_text());c['repeats']=4;(w/'cases.json').write_text(json.dumps(c,indent=2))
template=(p/'controller.py').read_text();assert template.count("else'off'))/'libgarnet.dylib'")==1
flags=json.loads((p/'build-flag-comparison.json').read_text());assert {k:(flags['primary'][k],flags['on'][k])for k in flags['on']if flags['primary'][k]!=flags['on'][k]}=={'GARNET_METAL_PRIMARY_VECTOR_PREFILL':('OFF','ON')}
s=(p/'controller.py').read_text().replace("work/worldsense-vector-prefill'","work/worldsense-vector-prefill/primary-speed'").replace("else'off'))/'libgarnet.dylib'","else'primary'))/'libgarnet.dylib'").replace("stage(w/'before-libgarnet.dylib')","stage(w.parent/'before-libgarnet.dylib')").replace("str(w/'wait_cantor.py')","str(w.parent/'wait_cantor.py')")
assert "work/worldsense-vector-prefill/primary-speed'"in s and "str(w.parent/'wait_cantor.py')"in s
assert s.count("else'primary'))/'libgarnet.dylib'")==1
(w/'controller.py').write_text(s)
(w/'execution-plan.json').write_text(json.dumps(dict(status='prepared_after_resource_pass',fresh_workers=['control-1','candidate-1'],calls=20,warmups_retained=True,no_promotion_without_two_fresh_full_confirmations=True),indent=2))
