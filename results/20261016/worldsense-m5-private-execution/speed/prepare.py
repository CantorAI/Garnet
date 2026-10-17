from pathlib import Path
import json,shutil
w=Path(__file__).resolve().parent;p=w.parent
assert json.loads((p/'resource-analysis.json').read_text())['passed']
assert json.loads((p/'restoration-verification.json').read_text())['benchmark_valid_request_http404']
assert not (w/'control-1.json').exists() and not (w/'candidate-1.json').exists()
for name in ['environment.json','starting-source-manifest.json']:shutil.copy2(p/name,w/name)
c=json.loads((p/'cases.json').read_text());c['repeats']=4;(w/'cases.json').write_text(json.dumps(c,indent=2))
s=(p/'controller.py').read_text().replace("work/worldsense-private-execution'","work/worldsense-private-execution/speed'").replace("stage(w/'before-libgarnet.dylib')","stage(w.parent/'before-libgarnet.dylib')").replace("str(w/'wait_cantor.py')","str(w.parent/'wait_cantor.py')")
assert "work/worldsense-private-execution/speed'"in s and "str(w.parent/'wait_cantor.py')"in s
(w/'controller.py').write_text(s)
(w/'execution-plan.json').write_text(json.dumps(dict(status='prepared_after_resource_pass',fresh_workers=['control-1','candidate-1'],calls=20,warmups_retained=True,no_promotion_without_two_fresh_full_confirmations=True),indent=2))
