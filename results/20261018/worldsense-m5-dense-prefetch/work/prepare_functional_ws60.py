"""Freeze current source/assets and prepare candidate-only real pipeline smoke."""
from pathlib import Path
import json,hashlib,ast,subprocess,sys
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');p=root/'work/worldsense-dense-prefetch';w=p/'functional-smoke';r=Path('/Users/shawn/CantorAI')
assert all(json.loads((p/'full-proof-complete.json').read_text()).values())
assert json.loads((p/'admission-complete.json').read_text())['ten_expected_config_rejections']
assert json.loads((p/'acceptance-gate-audit.json').read_text())['full_route_mutation_audit_pass']
subprocess.run([sys.executable,str(p/'freeze.py')],check=True)
w.mkdir(exist_ok=False)
t=(p/'controller.py').read_text().replace("work/worldsense-dense-prefetch'","work/worldsense-dense-prefetch/functional-smoke'")
t=t.replace("stage(w/'before-libgarnet.dylib')","stage(w.parent/'before-libgarnet.dylib')").replace("str(w/'wait_cantor.py')","str(w.parent/'wait_cantor.py')").replace("str(w/'restore_normal.py')","str(w.parent/'restore_normal.py')")
ast.parse(t);(w/'controller.py').write_text(t)
cases=json.loads((p/'cases.json').read_text());cases['repeats']=1;(w/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
source=r/'out/research/worldsense-dense-prefetch/source';m=json.loads((p/'correctness-source-manifest.json').read_text())
sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert len(m)==1720 and all(sha(source/k)==v for k,v in m.items())
paths=[w/'controller.py',w/'cases.json',p/'analyze_resource.py',p/'restore_normal.py',p/'wait_cantor.py',r/'WorldSense/tools/benchmark_recorded.py',r/'out/build/macos-arm64-release/bin/libgalaxy_garnetfilter.dylib']
(w/'before-functional-identities.json').write_text(json.dumps(dict(scope='Four candidate-only actual WorldSense/Galaxy native calls. Correctness evidence, not timing comparison.',source=m,libraries={role:sha(r/('out/build/garnet-metal-worldsense-dense-prefetch-'+role)/'libgarnet.dylib')for role in ['off','primary','on','fault']},drivers={str(x):sha(x)for x in paths},no_performance_comparison=True),indent=2)+'\n')
print('Candidate-only realWorldSense functional smoke prepared; no dispatch')
