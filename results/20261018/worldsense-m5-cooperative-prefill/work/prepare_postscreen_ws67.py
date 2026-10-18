from pathlib import Path
import json,ast,hashlib,subprocess,sys,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');p=b/'work/worldsense-cooperative-prefill';stage=sys.argv[1];assert stage in ['primary-speed','speed','confirmation'];w=p/stage;old=b/'work/worldsense-isolated-primary'/stage
assert json.loads((p/'production-speed/independent-screen-audit.json').read_text())['speed_passed']
if stage in ['speed','confirmation']:assert json.loads((p/'primary-speed/analysis.json').read_text())['passed']
if stage=='confirmation':assert json.loads((p/'speed/analysis.json').read_text())['passed']
assert not(w/'driver-identities-before-dispatch.json').exists(),'Do not overwrite prepared/dispatched stage'
t=(old/'prepare.py').read_text().replace('worldsense-isolated-primary','worldsense-cooperative-prefill')
if stage=='primary-speed':
 a="'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY':('OFF','ON')}";assert t.count(a)==1;t=t.replace(a,"'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY':('OFF','ON'),'GARNET_METAL_SOLE_INTERLEAVED_MLP':('OFF','ON'),'GARNET_METAL_SOLE_INTERLEAVED_DIRECT_PREFILL':('OFF','ON'),'GARNET_METAL_SOLE_DUAL_PROJECTION_DECODE':('OFF','ON'),'GARNET_METAL_SOLE_COOPERATIVE_PREFILL':('OFF','ON')}")
 t=t.replace('Sixflags(vector/isolation/directupload/K32/cached/full-isolation): samewideVLM/rawtextpartition andglobalretirement, differentmetadata/prefill/directupload','Tenflags: originalvector/isolation/directupload/K32/cached/full-isolation plussoleMLP/direct/dual/cooperative. Samewide primary partition/globalretirement; wholeprofile comparison, no isolatedkernel attribution.')
if stage in ['primary-speed','speed']:
 a='.replace("str(w/\'wait_cantor.py\')","str(w.parent/\'wait_cantor.py\')")';assert t.count(a)==1;t=t.replace(a,a+'.replace("str(w/\'restore_normal.py\')","str(w.parent/\'restore_normal.py\')")')
if stage=='confirmation':
 # The original controller reads local environment/source manifests, as the screen controllers do.
 a="c=json.loads((p.parent/'worldsense-vision-workspace/confirmation/cases.json').read_text())";assert t.count(a)==1;t=t.replace(a,"for name in ['environment.json','starting-source-manifest.json']:shutil.copy2(p/name,w/name)\n"+a)
ast.parse(t);(w/'prepare.py').write_text(t)
t=(old/'run_stage.py').read_text().replace('worldsense-isolated-primary','worldsense-cooperative-prefill');a="subprocess.run([sys.executable,str(w/'prepare.py')],check=True)";assert t.count(a)==1;t=t.replace(a,"assert (w/'execution-plan.json').exists(), 'External preparation must complete before dispatch'")
a=" subprocess.run([sys.executable,str(w.parent/'dependency_closure.py'),'"+stage+"'],check=True)";assert t.count(a)==1;t=t.replace(a,a+"\n identity=json.loads((w/'driver-identities-before-dispatch.json').read_text());assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in identity['files'].items())")
ast.parse(t);(w/'run_stage.py').write_text(t)
subprocess.run([sys.executable,str(p/'dependency_closure.py'),'pre-'+stage],check=True)
# Fresh model identities before each new stage. No GPU/inference, hardware-cold or init split claim.
def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
r=Path('/Users/shawn/CantorAI');files={}
for model,assets in json.loads((p/'model-assets.json').read_text()).items():
 for rel,info in assets.items():
  path=r/'out/models'/model/rel;assert path.stat().st_size==info['bytes']and sha(path)==info['sha256'];files[str(path)]=info
(w/'learned-assets-before-dispatch.json').write_text(json.dumps(files,indent=2)+'\n')
subprocess.run([sys.executable,str(w/'prepare.py')],check=True)
(w/'prospective-stage-driver-note.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,original_case_count_order_metric_golden_thresholds_preserved=True,changes='Paths for normal restoration, exact10flagPRIMARYcontrast, supplemental fullconfirmation manifest copies, explicit outside-worker preparation and postworker driver identities. No inference yet.',worker_dispatched=False,files={str(path):sha(path)for path in [w/'prepare.py',w/'run_stage.py',w/'analyze.py',Path(__file__)]}),indent=2)+'\n')
print('Original '+stage+' prepared; all dependencies/19assets rechecked, no dispatch')
