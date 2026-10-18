"""Prospective role/path adaptations, exact inverse; no math/oracle changes."""
from pathlib import Path
import json,hashlib,ast,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-primary-simd-metadata';old=b/'work/worldsense-cooperative-prefill';proof={}
names=['build_corpus.py','build_direct_corpus.py','build_dual_pair_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','primary_corpus_test.cpp','direct_float_corpus_test.cpp','dual_pair_corpus_test.cpp']
for name in names:
 original=(old/name).read_text();text=original.replace('worldsense-cooperative-prefill','worldsense-primary-simd-metadata');ops=[]
 def change(x,y,count=1):
  global text
  assert text.count(x)==count,(name,x,text.count(x));text=text.replace(x,y);ops.append(dict(before=x,after=y,count=count))
 if name in ['build_corpus.py','build_direct_corpus.py','build_dual_pair_corpus.py']:
  change('garnet-metal-worldsense-primary-simd-metadata-on','garnet-metal-worldsense-primary-simd-metadata-compat67')
 if name=='resident_controller.py':
  change("['off','primary','on','restore','direct64','dual66']","['off','primary','on','compat67']")
  # Profile fallback is still exercised for the original paired COMPAT profile.
  change("'resident-profile-on-'","'resident-profile-compat67-'")
  change('garnet-metal-worldsense-primary-simd-metadata-on','garnet-metal-worldsense-primary-simd-metadata-compat67')
  change("mode='profile-on'","mode='profile-compat67'")
 if name=='resident_guard.py':
  change("endswith(('-on','-restore','-direct64','-dual66'))","endswith('-compat67')")
 if name=='api_state_guard.py':
  change("['off','primary','on','restore','direct64','dual66']","['off','primary','on','compat67']")
  change("vector_expected=mode in ['on','restore','direct64','dual66'];direct_expected=mode in ['on','direct64','dual66'];dual_expected=mode in ['on','dual66'];cooperative_expected=mode=='on'", "vector_expected=mode=='compat67';direct_expected=mode=='compat67';dual_expected=mode=='compat67';cooperative_expected=mode=='compat67';simd_expected=mode=='on'")
  change(" assert ('sole_cooperative_prefill'in call['backend'])==cooperative_expected", " assert ('sole_cooperative_prefill'in call['backend'])==cooperative_expected\n assert ('primary_simd_metadata'in call['backend'])==simd_expected")
 if name.endswith('.py'):ast.parse(text)
 inverse=text
 for op in reversed(ops):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
 assert inverse.replace('worldsense-primary-simd-metadata','worldsense-cooperative-prefill')==original,name
 assert not(w/name).exists(),name;(w/name).write_text(text);proof[name]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops,original_math_oracles_bounds_control_flow_inverse_exact=True)
for name in ['direct-upload-selection.json','model-assets.json']:shutil.copy2(old/name,w/name)
(w/'learned-guard-adaptation-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print('WS68 original guards inverse exact; corpus COMPAT67, ON new helpers separately; model roles OFF/PRIMARY/ON/COMPAT67.')
