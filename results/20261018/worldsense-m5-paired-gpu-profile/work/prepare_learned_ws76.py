"""Build correctness tools/fixtures and freeze the complete two-source closure."""
from pathlib import Path
import ast,datetime,hashlib,json,subprocess,sys
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');main=b/'work/worldsense-paired-gpu-profile';w=b/'work/worldsense-paired-gpu-profile-current';legacy=b/'work/worldsense-paired-gpu-profile-legacy';r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin'
load=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
assert load(main/'compatibility-native-independent-audit.json')['combined_native_CTest_count']==79
assert not(w/'learned-identities-before.json').exists()
for folder in [w,legacy]:
 for p in folder.glob('*.py'):ast.parse(p.read_text())
ast.parse((b/'work/worldsense-opt/verify_learned_ws76.py').read_text())
for name in ['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py']:
 with(w/('prepare-'+name+'.log')).open('w')as log:subprocess.run([sys.executable,str(w/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS '+name,flush=True)
source=load(main/'current-source-manifest.json');legacy_source=load(main/'legacy-source-manifest.json')
s=r/'out/research/worldsense-paired-gpu-profile-current/source';ls=r/'out/research/worldsense-paired-gpu-profile-legacy/source'
assert len(source)==1723 and len(legacy_source)==1708 and all(sha(s/k)==v for k,v in source.items())and all(sha(ls/k)==v for k,v in legacy_source.items())
inputs={}
for model,assets in load(w/'model-assets.json').items():
 for rel,info in assets.items():
  p=r/'out/models'/model/rel;assert p.stat().st_size==info['bytes']and sha(p)==info['sha256'];inputs[str(p)]=info['sha256']
files=[]
for folder in [w,legacy]:files += [p for p in folder.iterdir()if p.is_file()and p.suffix in ['.py','.cpp','.json','.dylib']]
# All files currently in those fresh proof roots are prepared inputs/build IDs;
# no proof result or transient controller log is admitted as an input.
assert not(w/'learned-controller-terminal.json').exists()
files += [p for folder in ['fixture-1-8','fixture-2-64']for p in(w/folder).rglob('*')if p.is_file()]
for name,info in load(w/'learned-guard-adaptation-proof.json').items():files.append(Path(info['seed']))
for name,info in load(main/'legacy-and-counter-guard-adaptation-proof.json').items():files.append(Path(info['seed']))
files += [main/name for name in ['native-design-before-implementation.txt','full-proof-addendum-before-supplemental-build.txt','learned-plan-before-helper-adaptation.txt','protocol.json','legacy-seed-source-manifest.json','current-seed-source-manifest.json','legacy-source-manifest.json','current-source-manifest.json','legacy-source-transformations.json','current-source-transformations.json','native-independent-audit.json','native-auditor-log-format-repair.json','compatibility-proof-protocol.json','compatibility-native-independent-audit.json','learned-build-aliases.json','legacy-and-counter-guard-adaptation-proof.json','independent-learned-auditor-adaptation-proof.json','learned-preparation-stage.json']]
files += [b/'work/worldsense-opt'/name for name in ['prepare_full_ws76.py','prepare_learned_auditor_ws76.py','prepare_learned_ws76.py','verify_learned_ws76.py','run_learned_ws76.py','design_paired_counter_ws76.py','setup_paired_counter_ws76.py','prepare_compatibility_ws76.py','verify_native_ws76.py','verify_native_ws76_v2.py','verify_native_compatibility_ws76.py']]
files += [n/'xlang3',n/'libxlang3_runtime.dylib',n/'libgarnet.dylib',r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib',w.parent/'worldsense-primary-blocks/matrix-selection.json',w.parent/'worldsense-prefix-cache/state-guard.json',w.parent/'worldsense-prefix-cache/prefix-model-guard.json',w.parent/'worldsense-request-residency/vision-grid-off.json',w.parent/'worldsense-isolated-primary/api-state-guard-off.json']
files += [w.parent/'skill-expansion'/name for name in ['car.jpg','traffic.jpg','bunny.jpg']]
roles=load(main/'protocol.json')['roles']+list(load(main/'compatibility-proof-protocol.json')['roles'])
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-paired-gpu-profile-'+role)
 files += [p for p in build.iterdir()if p.is_file()and(p.name.startswith('garnet_')or p.name in ['libgarnet.dylib','libgarnet_metal.a','CMakeCache.txt'])]
recipes={}
for name in ['corpus-build-identity.json','direct-corpus-build-identity.json']:
 record=load(w/name);cmd=record['command'];executable=Path(cmd[cmd.index('-o')+1]);cpp=next(Path(arg)for arg in cmd if arg.endswith('_corpus_test.cpp'));lib=next(Path(arg)for arg in cmd if arg.endswith('/libgarnet_metal.a'))
 assert record['source_sha256']==sha(cpp)and record['library_sha256']==sha(lib)and record['executable_sha256']==sha(executable)
 files += [executable,cpp,lib];recipes[name]=dict(command=cmd,source=str(cpp),source_sha256=sha(cpp),library=str(lib),library_realpath=str(lib.resolve()),library_sha256=sha(lib),executable=str(executable),executable_realpath=str(executable.resolve()),executable_sha256=sha(executable))
inputs.update({str(p):sha(p)for p in files})
assert sum(len(v)for v in load(w/'model-assets.json').values())==19
assert all(Path(k).name!='worldsense-http.py' for k in inputs)
aliases=load(main/'learned-build-aliases.json')
assert all(str(Path(k).resolve())==v['resolved']and str(Path(k).readlink())==v['target']and sha(Path(k)/'libgarnet.dylib')==v['library_sha256']for k,v in aliases.items())
record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=source,legacy_root=str(ls),legacy_source=legacy_source,external=inputs,build_aliases=aliases,exact_corpus_build_recipes=recipes,scope='BEFORE first learned call: both source closures, all19assets, tenlibraries/native tools, all original/supplemental helpers/fixtures/oracles, exact corpus command/source/staticlibrary/executable identities, complete consumed outer controller and independent verifier. Generated model results and private authentication helper excluded. Full build/SDK/host closure remains required before actual diagnostic capture.')
(w/'learned-identities-before.json').write_text(json.dumps(record,indent=2)+'\n')
print('Both source closures/19assets/tenlibs/all guards/outer drivers and exact corpus recipes frozen BEFORE learned dispatch',flush=True)
