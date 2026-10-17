"""Fail-fast provenance preparation only. Never starts a benchmark worker."""
from pathlib import Path
import json,hashlib,subprocess,shutil,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin'
assert all(v==0 for v in json.loads((w/'native-results.json').read_text()).values())
assert json.loads((w/'api-correctness-result.json').read_text())['states_tokens_prefix_and_recovery']
assert json.loads((w/'private-trained-complete.json').read_text())['private4_and6_off_on']
faults=json.loads((w/'fault-trace-results.json').read_text());assert len(faults)==4 and all(v['exit']==0 for v in faults)
subprocess.run(['python3',str(w/'freeze.py')],check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(n/'libgarnet.dylib')==sha(r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib')
shutil.copy2(n/'libgarnet.dylib',w/'before-libgarnet.dylib')
c=json.loads((w/'cases.json').read_text());c['repeats']=1;c['queries']=['a passenger car','a passenger car'];(w/'cases.json').write_text(json.dumps(c,indent=2))
(w/'resource-case-identity.json').write_text(json.dumps({'sha256':sha(w/'cases.json'),'scope':'1warm+1followup,carSkill+twoidenticalqueries,4freshmirroredworkers32calls;diagnosticnotpromotion'},indent=2))
sdk=r/'xlang3/sdk';headers={str(p.relative_to(sdk)):sha(p)for p in sorted((sdk/'xlang3').rglob('*'))if p.is_file()};assert len(headers)>10
(w/'external-dependencies-pre-cohort.json').write_text(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Before benchmark dispatch; preparation process never dispatches benchmark','sdk_headers':headers,'runtime_binaries':{k:sha(n/k)for k in ['xlang3','libxlang3_runtime.dylib']},'xlang3_status':subprocess.check_output(['git','-C',str(r/'xlang3'),'status','--short'],text=True),'xcode':subprocess.check_output(['xcodebuild','-version'],text=True)},indent=2))
for mode in ['off','on','fault']:
 b=r/('out/build/garnet-metal-worldsense-latency-trace-'+mode)
 for rel,dest in [('CMakeCache.txt',mode+'-CMakeCache.txt'),('Testing/Temporary/LastTest.log',mode+'-LastTest.log'),('CMakeFiles/garnet_metal.dir/flags.make',mode+'-flags.make')]:
  assert (b/rel).exists(),(mode,rel)
  shutil.copy2(b/rel,w/dest)
(w/'cohort-preparation-complete.json').write_text(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_prerequisites_checked':True,'worker_dispatched':False},indent=2))
print('All prerequisites/source/model/build/dependencies/backup/request frozen; no worker dispatched',flush=True)
