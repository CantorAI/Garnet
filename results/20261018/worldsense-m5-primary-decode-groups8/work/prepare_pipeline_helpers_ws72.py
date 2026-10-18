"""Prepare future WorldSense stages/native admission; no dependent execution."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');d=b/'work/worldsense-opt';w=b/'work/worldsense-primary-decode-groups8';proof={}
for stem in ['prepare_functional','verify_functional','prepare_resource','verify_resource','prepare_actual_speed','verify_screen','audit_routes']:
 original=(d/(stem+'_ws71.py')).read_text();t=original.replace('worldsense-minimal-full-prefill','worldsense-primary-decode-groups8').replace('ws71.py','ws72.py').replace('WS71','WS72').replace("['native_tests_pass']==54","['native_tests_pass']==60").replace('len(m)==1721','len(m)==1723').replace('source_files=1721','source_files=1723')
 if stem=='prepare_resource':t=t.replace("assert load(w/'learned-independent-audit.json')['all_five_readers_in_actual_FULL_candidate'] is True","assert load(w/'learned-independent-audit.json')['all_five_readers_in_actual_FULL_candidate'] is True and load(w/'learned-independent-audit.json')['actual_groups8_deltas_match_original_primary_and_zero_text']")
 if stem=='verify_functional':t=t.replace('actual_FULL_vector_markers_core_zero_and_native_memory_pass','actual_groups8_deltas_match_primary_core_zero_and_native_memory_pass').replace('golden/native/minimal-FULL/source/memory','golden/native/groups8/source/memory')
 if stem=='audit_routes':
  t=t.replace("'primary_core_isolated_library','request_scope_weight_residency'","'primary_core_isolated_library','primary_decode_groups8','request_scope_weight_residency'")
  t=t.replace("['core_library_decode_encoded','core_library_prefill_encoded','core_library_head_encoded']","['core_library_decode_encoded','core_library_prefill_encoded','core_library_head_encoded','decode_groups8_encoded','head_groups8_encoded']")
  t=t.replace("['primary_core_isolated_library','packed_request_weight_residency'","['primary_core_isolated_library','primary_decode_groups8','packed_request_weight_residency'")
  t=t.replace('Strict14field','Strict16field').replace('core provenance','core/groups8 provenance')
 ast.parse(t);p=d/(stem+'_ws72.py');assert not p.exists();p.write_text(t);proof[stem]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),adapted_sha256=hashlib.sha256(t.encode()).hexdigest(),scope='Path/count/profile/new encoded groups fields and negative marker tests; original quantitative/golden/metric thresholds preserved.')
original=(d/'prepare_worldsense_ws71.py').read_text();t=original.replace('worldsense-minimal-full-prefill','worldsense-primary-decode-groups8').replace('WS71','WS72')
t=t.replace("['all_five_readers_in_actual_FULL_candidate'] is True and", "['all_five_readers_in_actual_FULL_candidate'] is True and load(w/'learned-independent-audit.json')['actual_groups8_deltas_match_original_primary_and_zero_text'] and")
needle=" if 'primary_core_isolated_library'in call['backend']:return False";after=" if ('primary_decode_groups8'in call['backend'])!=enabled:return False\n"+needle;assert t.count(needle)==1;t=t.replace(needle,after)
t=t.replace("'core_library_head_encoded'}","'core_library_head_encoded','decode_groups8_encoded','head_groups8_encoded'}")
needle=" elif enabled or any(key in t0 for key in ['primary_storage_before','primary_encoded_delta']):return False"
after="""  for key,oldkey in [('decode_groups8_encoded','decode_encoded'),('head_groups8_encoded','head_encoded')]:
   if after[key]<before[key]or after[key]-before[key]!=(encoded[oldkey]if enabled else 0):return False
   if not enabled and(before[key]!=0 or after[key]!=0):return False
   if call['model_id']=='Qwen3-1.7B'and after[key]!=before[key]:return False
 elif enabled or any(key in t0 for key in ['primary_storage_before','primary_encoded_delta']):return False"""
assert t.count(needle)==1;t=t.replace(needle,after)
t=t.replace("'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']}","'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_DECODE_GROUPS8']}")
t=t.replace("'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']}\n","'GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_DECODE_GROUPS8']}\n")
t=t.replace('WS72 existing minimalFULL+vector profile, ZERO1721sourcechanges, RAW6flags and matchedPRIMARY3flags. Direct/K32/cached/rawrequest/CORE OFF. Actual three core counters zero; same native telemetry included, no isolated causal or GPU ISA proof.','WS72 packed decode8SIMD/256threads/32outputs, exact2origin substitutions andnoTail32, originalFP/math andshaderliteral preserved. RAW7flags/matchedPRIMARY4flags, vector/FULL included, direct/K32/cached/rawrequest/CORE OFF. Actual groups8body/head encoded deltas trackoldcounts, zero text/core. No isolated causal or GPU ISA proof.')
t=t.replace('source/sixflagRAWcontrast/threeflagPRIMARYcontrast','source/sevenflagRAWcontrast/fourflagPRIMARYcontrast')
ast.parse(t);p=d/'prepare_worldsense_ws72.py';assert not p.exists();p.write_text(t)
t=(d/'admission_ws71.py').read_text().replace('worldsense-minimal-full-prefill','worldsense-primary-decode-groups8').replace('WS71','WS72')
needle='results={}';extra='''groups=['-DGARNET_METAL_PRIMARY_DECODE_GROUPS8=ON','-DGARNET_METAL_PRIMARY_VECTOR_PREFILL=ON','-DGARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY=ON','-DGARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY=ON']
require8='Packed groups8 requires primary/wide/lazy/body-shape/vector/prefill-isolation/full-isolation'
cases += [('groups8-primary',groups+['-DGARNET_METAL_PRIMARY_BF16_BLOCKS=OFF','-DGARNET_METAL_WIDE_PRIMARY_STORAGE=OFF','-DGARNET_METAL_PRIMARY_VECTOR_PREFILL=OFF','-DGARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY=OFF','-DGARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY=OFF'],require8),('groups8-wide',groups+['-DGARNET_METAL_WIDE_PRIMARY_STORAGE=OFF'],require8),('groups8-lazy',groups+['-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=OFF','-DGARNET_METAL_WIDE_PRIMARY_STORAGE=OFF'],require8),('groups8-body',groups+['-DGARNET_METAL_BODY_SHAPE=OFF'],'Initial primary block profile requires body-shape support'),('groups8-vector',groups+['-DGARNET_METAL_PRIMARY_VECTOR_PREFILL=OFF','-DGARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY=OFF','-DGARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY=OFF'],require8),('groups8-prefill-isolation',groups+['-DGARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY=OFF','-DGARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY=OFF'],require8),('groups8-full',groups+['-DGARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY=OFF'],require8),('groups8-core',groups+['-DGARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY=ON'],'Primary core isolation excludes vector/full/prefill isolation and raw request residency'),('groups8-request',groups+['-DGARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY=ON'],'Packed groups8 excludes CORE and raw request residency')]
results={}''';assert t.count(needle)==1;t=t.replace(needle,extra).replace('len(results)==15','len(results)==24').replace('new8_core_combinations_rejected=True','new8_core_combinations_rejected=True,new9_groups8_combinations_rejected=True').replace('original7+8core configuration','original7+8core+9groups8 configuration');ast.parse(t);p=d/'admission_ws72.py';assert not p.exists();p.write_text(t)
(w/'pipeline-helper-adaptations.json').write_text(json.dumps(proof,indent=2)+'\n')
print('WS72 future original gates,16field/newgroup parity,all legacy markers and24CMake exclusions prepared; no dependent stage executed.')
