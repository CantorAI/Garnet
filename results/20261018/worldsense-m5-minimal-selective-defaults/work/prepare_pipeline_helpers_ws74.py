"""Prepare future stages without executing them or editing any live proof input."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');d=b/'work/worldsense-opt';w=b/'work/worldsense-minimal-selective-defaults';proof={}
for stem in ['prepare_functional','verify_functional','prepare_resource','verify_resource','prepare_actual_speed','verify_screen','admission','audit_routes']:
 original=(d/(stem+'_ws71.py')).read_text();t=original.replace('worldsense-minimal-full-prefill','worldsense-minimal-selective-defaults').replace('ws71.py','ws74.py').replace('WS71','WS74')
 if stem=='admission':
  x='results={}';y="""selective=['-DGARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS=ON'];selective_require='Packed-only default retirement requires primary weights, wide storage and lazy defaults'
cases += [('selective-primary',selective+['-DGARNET_METAL_WIDE_PRIMARY_STORAGE=OFF','-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF'],selective_require),('selective-wide',selective+['-DGARNET_METAL_WIDE_PRIMARY_STORAGE=OFF'],selective_require),('selective-lazy',selective+['-DGARNET_METAL_WIDE_PRIMARY_STORAGE=OFF','-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=OFF'],selective_require)]
results={}""";assert t.count(x)==1;t=t.replace(x,y).replace('len(results)==15','len(results)==18').replace('new8_core_combinations_rejected=True','new8_core_combinations_rejected=True,three_selective_dependencies_rejected=True').replace('original7+8core configuration','original7+8core+3selective configuration')
 if stem=='audit_routes':
  x="'primary_simd_metadata','native_text_worker']";assert t.count(x)==1;t=t.replace(x,"'primary_simd_metadata','native_text_worker','primary_only_lazy_defaults','primary_decode_groups2','primary_decode_groups8']")
  x="0 if mode!='off'and field!='lazy_default_rehydrated_bytes'else 1";assert t.count(x)==1;t=t.replace(x,"0 if c['timings'][field]>0 else 1")
  x="'primary_vector_prefill_k32_only','cached_matrix_metadata']";assert t.count(x)==1;t=t.replace(x,"'primary_vector_prefill_k32_only','cached_matrix_metadata','primary_only_lazy_defaults','primary_decode_groups2','primary_decode_groups8']")
  x="   for group in ['primary_storage_before','primary_storage_after','primary_encoded_delta']:";y="""   for field in ['lazy_default_retired_bytes','lazy_default_planned_slots','lazy_default_rehydrated_bytes']:
    for invalid in [None,True,False,-1,0.0,'0',float('nan')]:bad=copy.deepcopy(c);bad['timings'][field]=invalid;reject(bad)
   for group in ['primary_storage_before','primary_storage_after','primary_encoded_delta']:""";assert t.count(x)==1;t=t.replace(x,y)
  t=t.replace('Strict14field snapshots/5encoded deltas/core provenance and legacy compatibility','Strict14field snapshots/5encoded deltas/corezero/selective graphpolicy and legacy compatibility')
 ast.parse(t);p=d/(stem+'_ws74.py');assert not p.exists();p.write_text(t);proof[stem]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),adapted_sha256=hashlib.sha256(t.encode()).hexdigest(),scope='Same original numeric/golden thresholds. New selective-policy admission only, added missing-dependency/foreign-grouping-marker rejection cases and policy-appropriate corrupted-counter copies.')
(w/'pipeline-helper-adaptations.json').write_text(json.dumps(proof,indent=2)+'\n')
print('WS74 eight future stage helpers prepared; no dependent stage dispatched. Learned input closure untouched.')
