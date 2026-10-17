"""Freeze canonical integration and immutable deployed control before dispatch."""
from pathlib import Path
import ast,datetime,hashlib,json,shutil,subprocess
w=Path(__file__).resolve().parent;p=w.parent;r=Path('/Users/shawn/CantorAI');repo=r/'Garnet';old=p.parent/'worldsense-primary-u32';n=r/'out/build/macos-arm64-release/bin'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert not list(w.glob('control-full-*.json'))and not list(w.glob('candidate-full-*.json'))
assert all(json.loads((p/'api-correctness-result.json').read_text()).values())
assert all(v==0 for v in json.loads((p/'native-results.json').read_text()).values())
m=json.loads((p/'canonical-source-before-build.json').read_text());assert all(sha(repo/rel)==digest for rel,digest in m.items())
for mode in ['off','on','fault']:
    a=r/('out/build/garnet-metal-worldsense-primary-u32-integration-'+mode)/'primary_projection_words.bin';b=r/('out/build/garnet-metal-worldsense-primary-u32-'+mode)/'primary_projection_words.bin'
    assert a.read_bytes()==b.read_bytes(),'Canonical primitive outputs differ from validated prototype'
control_source=r/'out/research/worldsense-prefix-cache/source'
cm={str(q.relative_to(control_source)):sha(q)for q in sorted(control_source.rglob('*'))if q.is_file()and '__pycache__'not in q.parts and '.git'not in q.parts}
(p/'control-source-before-dispatch.json').write_text(json.dumps(cm,indent=2))
libs={'off':r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib','on':r/'out/build/garnet-metal-worldsense-primary-u32-integration-on/libgarnet.dylib'}
assert sha(n/'libgarnet.dylib')==sha(libs['off'])
shutil.copy2(n/'libgarnet.dylib',p/'before-libgarnet.dylib');shutil.copy2(w/'cases.json',p/'cases.json')
assets=json.loads((old/'model-assets.json').read_text())
for model,files in assets.items():
    for rel,info in files.items():
        path=r/'out/models'/model/rel;assert path.stat().st_size==info['bytes']and sha(path)==info['sha256'],(model,rel)
(p/'model-assets.json').write_text(json.dumps(assets,indent=2))
env={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'libraries':{k:sha(q)for k,q in libs.items()},'control_source':str(control_source),'control_serving_sha256':cm['runtime/metal_serving.py'],'candidate_source':str(repo),'scope':'Actual immutable WS10deployed control and canonical-sourceON candidate. Separate source manifests. No fabricated control memory counters.','os':subprocess.check_output(['/usr/bin/sw_vers'],text=True),'xcode':subprocess.check_output(['xcodebuild','-version'],text=True)}
(p/'environment.json').write_text(json.dumps(env,indent=2))
subprocess.run(['python3',str(p/'dependency_closure.py'),'capture'],check=True)

# Preflight actual old control data and canonical correctness calls against the selected route checks.
fn=next(v for v in ast.parse((w/'analyze.py').read_text()).body if isinstance(v,ast.FunctionDef)and v.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'native_route','exec'),ns)
g=json.loads((r/'Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json').read_text());control_calls=[]
for item in g['items']:
    if item['policy']=='candidate':control_calls.extend(c for c in item['result'].get('model_response',{}).values()if isinstance(c,dict)and c.get('status')=='ok')
control_calls.extend(q['result']['benchmark_model_response']for q in g['search_plans']if q['policy']=='candidate')
assert control_calls and all(ns['native_route'](c,False)for c in control_calls)
calls=json.loads((p/'api-state-progress-on.json').read_text())['records'];assert len(calls)==5 and all(ns['native_route'](r['call'],True)for r in calls)

# Preserve all latency/golden/input calculations. This additional acceptance stage has no legacy memory API.
original=(old/'confirmation/analyze.py').read_text();candidate=(w/'analyze.py').read_text()
start='models={';end='mem={';a=original[original.index(start):original.index(end)];b=candidate[candidate.index(start):candidate.index(end)];assert ast.dump(ast.parse(a),include_attributes=False)==ast.dump(ast.parse(b),include_attributes=False)
(w/'preflight.json').write_text(json.dumps({'control_native_route_calls':len(control_calls),'canonical_native_route_calls':len(calls),'model_median_latency_and_decode_AST_unchanged':True,'all_prerequisites_checked':True,'scope':'Additional actualproduction comparison; prototype memory gates already passed and remain prerequisites. No timing worker dispatched.'},indent=2))
paths=list(p.rglob('*.py'))+[r/'WorldSense/tools/benchmark_recorded.py']
(w/'driver-identities-before-dispatch.json').write_text(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{str(q):sha(q)for q in paths}},indent=2))
print('Canonical correctness/golden/production route/source/models/build/SDK/host/drivers frozen; no worker dispatched',flush=True)
