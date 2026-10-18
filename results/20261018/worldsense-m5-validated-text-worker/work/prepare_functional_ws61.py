"""Prepare four candidate-only real WorldSense calls after full admission."""
from pathlib import Path
import json, hashlib, ast, subprocess, sys
root = Path(__file__).resolve().parent.parent
p = root / 'worldsense-validated-text-worker'
w = p / 'functional-smoke'
r = Path('/Users/shawn/CantorAI')
assert json.loads((p/'learned-independent-audit.json').read_text())['original_learned_native_state_token_vision_verifier_pass']
assert json.loads((p/'admission-complete.json').read_text())['thirteen_expected_config_rejections']
assert json.loads((p/'acceptance-gate-audit.json').read_text())['full_route_mutation_audit_pass']
subprocess.run([sys.executable, str(p/'freeze.py')], check=True)
w.mkdir(exist_ok=False)
text = (p/'controller.py').read_text().replace("work/worldsense-validated-text-worker'", "work/worldsense-validated-text-worker/functional-smoke'")
text = text.replace("stage(w/'before-libgarnet.dylib')", "stage(w.parent/'before-libgarnet.dylib')")
text = text.replace("str(w/'wait_cantor.py')", "str(w.parent/'wait_cantor.py')")
text = text.replace("str(w/'restore_normal.py')", "str(w.parent/'restore_normal.py')")
ast.parse(text)
(w/'controller.py').write_text(text)
cases = json.loads((p/'cases.json').read_text()); cases['repeats'] = 1
(w/'cases.json').write_text(json.dumps(cases, indent=2)+'\n')
source = r/'out/research/worldsense-validated-text-worker/source'
manifest = json.loads((p/'gateway-source-manifest.json').read_text())
sha = lambda q: hashlib.sha256(q.read_bytes()).hexdigest()
assert len(manifest) == 1726 and all(sha(source/k)==v for k,v in manifest.items())
paths = [w/'controller.py', w/'cases.json', p/'analyze_resource.py', p/'restore_normal.py', p/'wait_cantor.py',
         r/'WorldSense/tools/benchmark_recorded.py', r/'out/build/macos-arm64-release/bin/libgalaxy_garnetfilter.dylib']
paths += [r/('out/build/garnet-metal-worldsense-validated-text-worker-'+role)/'libgarnet_worker_host.dylib' for role in ['on','primary','fault']]
paths += [r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib']
(w/'before-functional-identities.json').write_text(json.dumps(dict(
    scope='Four candidate-only actual WorldSense/Galaxy native calls; correctness, not speed.', source=manifest,
    legacy_source=json.loads((p/'validated-source-manifest.json').read_text()),
    libraries={role:sha(r/('out/build/garnet-metal-worldsense-validated-text-worker-'+role)/'libgarnet.dylib') for role in ['off','primary','on','fault']},
    drivers={str(x):sha(x) for x in paths}, no_performance_comparison=True), indent=2)+'\n')
print('WS61 candidate-only actual WorldSense functional smoke prepared; no dispatch')
