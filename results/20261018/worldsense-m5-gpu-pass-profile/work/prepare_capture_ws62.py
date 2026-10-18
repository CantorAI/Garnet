"""Freeze the complete candidate-only actual WorldSense diagnostic before launch."""
from pathlib import Path
import subprocess,sys,json,hashlib,datetime,shutil,plistlib,os,ast
opt=Path(__file__).resolve().parent;base=opt.parent;w=base/'worldsense-gpu-counter-profile';p=w/'capture';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-gpu-counter-profile/source'
assert json.loads((w/'learned-independent-audit.json').read_text())['original_model_state_token_word_vision_verifier_pass']
assert json.loads((w/'learned-independent-audit.json').read_text())['actual_counter_error_recovery_pass']
assert not p.exists();p.mkdir()
external=r/'out/research/worldsense-gpu-counter-profile/capture';external.mkdir(exist_ok=False)
counter=external/'candidate-1.counters.jsonl';assert not counter.exists()
cases=json.loads((w/'cases.json').read_text());cases['repeats']=1;assert not cases['queries']and len(cases['cases'])==1
(p/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
route_seed=(base/'worldsense-isolated-primary/analyze_resource.py').read_text();tree=ast.parse(route_seed);fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='native_route')
(p/'native_route_admission.py').write_text(ast.get_source_segment(route_seed,fn)+'\n')
original=Path('/Users/shawn/Library/LaunchAgents/com.cantorai.cantor.plist');plist=plistlib.loads(original.read_bytes());assert set(plist.get('EnvironmentVariables',{}))=={'PATH'}
plist['EnvironmentVariables'].update(GARNET_METAL_COUNTER_TRACE='1',GARNET_METAL_COUNTER_TRACE_FILE=str(counter));temporary=p/'counter-benchmark.plist';temporary.write_bytes(plistlib.dumps(plist))
# Compiler dependency union and host files; selected libgarnet is staged later,
# so the inherited capture explicitly excludes that mutable deployment path.
text=(base/'worldsense-isolated-primary/dependency_closure.py').read_text().replace('worldsense-isolated-primary','worldsense-gpu-counter-profile').replace("['off','primary','on','fault']","['off','primary','on','quiet','fault']")
(w/'dependency_closure.py').write_text(text)
subprocess.run([sys.executable,str(w/'dependency_closure.py'),'capture'],check=True)
def sha(q):
 h=hashlib.sha256()
 with q.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
frozen=json.loads((w/'learned-identities-before.json').read_text());assert all(sha(s/k)==v for k,v in frozen['source'].items())
assert all(sha(Path(k))==v for k,v in frozen['external'].items())
paths=[q for q in w.rglob('*')if q.is_file()and q.suffix in ['.py','.cpp','.inc','.cmake']and '__pycache__'not in q.parts]
paths += [opt/name for name in ['prepare_capture_ws62.py','run_capture_ws62.py','verify_capture_ws62.py','summarize_counter_ws62.py','verify_learned_ws62.py','run_learned_ws62.py']]
paths += [p/'cases.json',p/'native_route_admission.py',temporary,original,w/'before-libgarnet.dylib',r/'Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json',r/'WorldSense/tools/benchmark_recorded.py',r/'out/build/macos-arm64-release/bin/libgalaxy_garnetfilter.dylib']
assert sha(w/'before-libgarnet.dylib')=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
flags={}
for mode in ['off','primary','on','quiet','fault']:
 cache=r/('out/build/garnet-metal-worldsense-gpu-counter-profile-'+mode)/'CMakeCache.txt';paths.append(cache)
 flags[mode]={line.split(':BOOL=')[0]:line.split(':BOOL=')[1]for line in cache.read_text().splitlines()if line.startswith('GARNET_')and ':BOOL='in line}
 assert flags[mode]['GARNET_METAL_GPU_PASS_COUNTER_PROFILE']==('OFF'if mode=='quiet'else'ON')
 for key in ['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS']:assert flags[mode][key]==('OFF'if mode=='off'else'ON')
 for key in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']:assert flags[mode][key]==('ON'if mode in ['on','quiet','fault']else'OFF')
(p/'five-build-flags.json').write_text(json.dumps(flags,indent=2)+'\n')
frozen['external'].update({str(q):sha(q)for q in paths})
frozen.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),capture_trace_path=str(counter),capture_plist=str(temporary),scope='Before the first actual pipeline diagnostic; all consumed sources, binaries, model assets, mathematical guards and outer/capture/verification drivers included. Compiler/SDK/host closure has its own manifest. No credentials/private authentication helper copied or hashed.')
(p/'identities-before-dispatch.json').write_text(json.dumps(frozen,indent=2)+'\n')
print('Complete actual WorldSense diagnostic source/models/runtime/host/drivers/request frozen; no worker dispatched')
