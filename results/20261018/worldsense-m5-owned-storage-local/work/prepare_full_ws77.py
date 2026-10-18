"""Preserve WS74 learned oracles; extend only exact new profile admission."""
from pathlib import Path
import ast,hashlib,json,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-owned-storage-local';old=w.parent/'worldsense-minimal-selective-defaults'
slug='worldsense-owned-storage-local';seed='worldsense-minimal-selective-defaults';proof={}
for name in ['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','primary_corpus_test.cpp','direct_float_corpus_test.cpp','request_session_guard.py','request_full_session_guard.py']:
 original=(old/name).read_text();text=original.replace(seed,slug);ops=[]
 def change(before,after):
  global text
  assert text.count(before)==1,(name,before,text.count(before));text=text.replace(before,after);ops.append(dict(before=before,after=after,count=1))
 if name=='api_state_guard.py':
  change("mode=sys.argv[1];assert mode in ['off','primary','on','compat56'];expected=0 if mode=='off'else 1;vector_expected=mode in ['on','compat56'];core_expected=False;compat_expected=mode=='compat56';selective_expected=mode in ['primary','on']", "mode=sys.argv[1];assert mode in ['off','primary','on','compat56','control','raw-borrow'];profile={'control':'on','raw-borrow':'off'}.get(mode,mode);expected=0 if profile=='off'else 1;vector_expected=profile in ['on','compat56'];core_expected=False;compat_expected=profile=='compat56';selective_expected=profile in ['primary','on'];owned_expected=mode in ['primary','on','raw-borrow']")
  change(" assert ('primary_full_isolated_library'in call['backend'])==vector_expected", " assert ('owned_storage_local'in call['backend'])==owned_expected\n assert ('primary_full_isolated_library'in call['backend'])==vector_expected")
 if name.endswith('.py'):ast.parse(text)
 inverse=text
 for op in reversed(ops):assert inverse.count(op['after'])==1;inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace(slug,seed)==original and not(w/name).exists()
 (w/name).write_text(text);proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops,original_math_oracles_bounds_control_flow_inverse_exact=True)
for name in ['direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
t=(old/'full_correctness.py').read_text().replace(seed,slug).replace('WS74','WS77').replace('len(manifest)==1721','len(manifest)==1722').replace("['native_tests_pass']==54","['native_tests_pass']==80").replace('source1721_and','source1722_and')
t=t.replace("for mode in ['off','primary','on','compat56']:run", "for mode in ['off','primary','on','compat56','control','raw-borrow']:run")
ast.parse(t);(w/'full_correctness.py').write_text(t)
t=(b/'work/worldsense-opt/verify_learned_ws74.py').read_text().replace(seed,slug).replace("old=w.parent/'worldsense-minimal-full-prefill'","old=w.parent/'worldsense-minimal-selective-defaults'")
t=t.replace("text.replace('worldsense-owned-storage-local','worldsense-minimal-full-prefill')","text.replace('worldsense-owned-storage-local','worldsense-minimal-selective-defaults')")
t=t.replace("len(frozen['source'])==1721","len(frozen['source'])==1722").replace("len(journal)==16","len(journal)==18").replace("['native_tests_pass']==54","['native_tests_pass']==80")
t=t.replace("for role in ['off','primary','on','compat56']}","for role in ['off','primary','on','compat56','control','raw-borrow']}")
anchor="  enabled=role!='off';compat=role=='compat56';isolated=False;vector=role in ['on','compat56']";assert t.count(anchor)==1
t=t.replace(anchor,"  profile={'control':'on','raw-borrow':'off'}.get(role,role);enabled=profile!='off';compat=profile=='compat56';isolated=False;vector=profile in ['on','compat56']\n  assert ('owned_storage_local'in call['backend'])==(role in ['primary','on','raw-borrow'])")
t=t.replace("selective=role in ['primary','on']","selective=profile in ['primary','on']")
t=t.replace('source1721_and','source1722_and').replace('four_API_modes20calls','six_API_modes30calls')
ast.parse(t);(b/'work/worldsense-opt/verify_learned_ws77.py').write_text(t)
t=(b/'work/worldsense-opt/prepare_learned_ws74.py').read_text().replace(seed,slug).replace('ws74.py','ws77.py').replace('WS74','WS77').replace("['native_tests_pass']==54","['native_tests_pass']==80").replace('len(source)==1721','len(source)==1722')
t=t.replace("w.parent/'worldsense-minimal-full-prefill'/name","w.parent/'worldsense-minimal-selective-defaults'/name")
t=t.replace("'immutable57-policy-guard.txt','read-only-consumer-prior-audit.json'","'read-only-owner-prior-audit.json','host-codegen-review.json'")
t=t.replace("'design_selective_ws77.py','setup_selective_ws77.py'","'design_owned_storage_ws77.py','setup_owned_storage_ws77.py','review_codegen_ws77.py'")
t=t.replace("for role in ['off','primary','on','fault','compat56','compat56-fault']:","for role in ['off','primary','on','fault','compat56','compat56-fault','control','raw-borrow']:")
ast.parse(t);(b/'work/worldsense-opt/prepare_learned_ws77.py').write_text(t)
print('Original full16 learned checks retained, plus exact CONTROL/RAW-BORROW API states (18 total); 12 guard inverses and new flag admission frozen before any model dispatch.')
