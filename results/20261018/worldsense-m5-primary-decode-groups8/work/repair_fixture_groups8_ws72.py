"""Pre-native review: keep forced-raw generic test keys distinct in RAW control."""
from pathlib import Path
import json,hashlib,shutil,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=b/'work/worldsense-primary-decode-groups8';s=r/'out/research/worldsense-primary-decode-groups8/source';a=w/'pre-native-generic-fixture-review';a.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text());roles=['off','primary','on','fault','compat56','compat56-fault']
assert(w/'build-controller.log').read_text().splitlines()==['BUILD PASS '+v for v in roles]
for name in ['correctness-source-manifest.json','source-inverse-proof.json','source-transformations.json','protocol.json','build.py','build-plan.json','build-controller.log','native-design-before-implementation.txt']+[f'build-{role}.log'for role in roles]:shutil.copy2(w/name,a/name)
for rel in load(w/'source-inverse-proof.json')['changed_or_added']:
 p=a/'source'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s/rel,p)
ids={}
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role);ids[role]={name:sha(build/name)for name in ['libgarnet.dylib','libgarnet_metal.a','garnet_primary_decode_groups8_test','CMakeCache.txt']}
(a/'build-identities.json').write_text(json.dumps(ids,indent=2)+'\n')
before='for(size_t columns:{260ul,272ul})for(size_t rows:{1ul,2ul}){run(rows,128,columns,true,false);run(rows,128,columns,true,true);}'
after='for(size_t columns:{260ul,272ul})for(size_t rows:{1ul,2ul}){run(rows,512,columns,true,false);run(rows,512,columns,true,true);}'
for p in [w/'primary_decode_groups8_test.cpp',s/'test2026/metal/primary_decode_groups8_test.cpp']:
 t=p.read_text();assert t.count(before)==1;p.write_text(t.replace(before,after))
note='''Pre-native fixture amendment: forced-raw generic/specialized cases use inner512, not128. In the OFF control, earlier attempted-packed cases are also raw and had already populated the same raw specialized key. Unique inner512 ensures the later generic cache-miss case really executes generic before preparation in ALL roles. Same60records/804504outputwords/20generic20specialized/16prefill4heads andGROUPS8body32/head4 expectations; no output-column/FP/numerical-bound/inference shader/host algorithm change. Static review BEFORE any native/model/timing execution. Initial six successful builds/logs/source delta/identities retained; rebuild tests before execution.\n'''
(w/'pre-native-fixture-amendment.txt').write_text(note);(a/'review.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),initial_build_session=36270,initial_build_exit_code=0,before_any_native_model_or_timing=True,before=before,after=after,reason=note),indent=2)+'\n')
m=load(w/'correctness-source-manifest.json');m['test2026/metal/primary_decode_groups8_test.cpp']=sha(s/'test2026/metal/primary_decode_groups8_test.cpp');assert len(m)==1723 and all(sha(s/k)==v for k,v in m.items());(w/'correctness-source-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
p=w/'protocol.json';d=load(p);d.update(pre_native_fixture_amendment_sha256=sha(w/'pre-native-fixture-amendment.txt'),amendment_before_native=True);p.write_text(json.dumps(d,indent=2)+'\n')
p=b/'work/worldsense-next.json';d=load(p);d.update(state='PRE_NATIVE_FIXTURE_REVIEW_REBUILD_NEXT',initial_build_session=36270,initial_build_exit_code=0,initial_six_builds_pass=True,pre_native_fixture_amendment=True,no_native_or_model_dispatched=True);p.write_text(json.dumps(d,indent=2)+'\n')
print('Pre-native generic key fixture repaired to unique inner512; initialsource/buildIDs/logs preserved, same60records/804504words. No native/model execution.')
