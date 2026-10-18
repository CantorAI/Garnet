"""Prepare four candidate-only actual WorldSense/Galaxy functional calls."""
from pathlib import Path
import json,hashlib,ast,subprocess,sys
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');p=b/'work/worldsense-direct-interleaved-prefill';w=p/'functional-smoke';r=Path('/Users/shawn/CantorAI')
assert json.loads((p/'learned-independent-audit.json').read_text())['all_original_model_word_state_token_vision_gates_pass']
assert json.loads((p/'admission-complete.json').read_text())['twentyone_expected_config_rejections']
assert json.loads((p/'acceptance-gate-audit.json').read_text())['full_route_mutation_audit_pass']
subprocess.run([sys.executable,str(p/'freeze.py')],check=True);w.mkdir(exist_ok=False)
t=(p/'controller.py').read_text().replace("work/worldsense-direct-interleaved-prefill'","work/worldsense-direct-interleaved-prefill/functional-smoke'")
t=t.replace("stage(w/'before-libgarnet.dylib')","stage(w.parent/'before-libgarnet.dylib')").replace("str(w/'wait_cantor.py')","str(w.parent/'wait_cantor.py')").replace("str(w/'restore_normal.py')","str(w.parent/'restore_normal.py')")
ast.parse(t);(w/'controller.py').write_text(t);cases=json.loads((p/'cases.json').read_text());cases['repeats']=1;(w/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
source=r/'out/research/worldsense-direct-interleaved-prefill/source';m=json.loads((p/'correctness-source-manifest.json').read_text());sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert len(m)==1721 and all(sha(source/k)==v for k,v in m.items())
paths=[w/'controller.py',w/'cases.json',p/'analyze_resource.py',p/'restore_normal.py',p/'wait_cantor.py',r/'WorldSense/tools/benchmark_recorded.py',r/'out/build/macos-arm64-release/bin/libgalaxy_garnetfilter.dylib',b/'work/worldsense-opt/verify_functional_ws64.py']
(w/'before-functional-identities.json').write_text(json.dumps(dict(scope='Four candidate-only actual WorldSense/Galaxy native calls. Correctness, no timing comparison.',source=m,libraries={role:sha(r/('out/build/garnet-metal-worldsense-direct-interleaved-prefill-'+role)/'libgarnet.dylib')for role in ['off','primary','on','fault','restore']},drivers={str(x):sha(x)for x in paths},no_performance_comparison=True),indent=2)+'\n');print('WS64 candidate-only four-call real pipeline functional smoke prepared; no dispatch')
