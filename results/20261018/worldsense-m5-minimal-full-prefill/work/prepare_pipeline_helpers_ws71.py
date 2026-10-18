"""Prepare future stage scripts in new files; execute no dependent stages."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');d=b/'work/worldsense-opt';w=b/'work/worldsense-minimal-full-prefill';proof={}
for stem in ['prepare_functional','verify_functional','prepare_resource','verify_resource','prepare_actual_speed','verify_screen','admission','audit_routes']:
 original=(d/(stem+'_ws70.py')).read_text();t=original.replace('worldsense-primary-core-isolation','worldsense-minimal-full-prefill').replace('ws70.py','ws71.py').replace('WS70','WS71')
 t=t.replace("['both_original_GPU_readers_in_actual_CORE_EACH_words']==4200857600","['all_five_readers_in_actual_FULL_candidate'] is True").replace("['actual_core_deltas_equal_original_VLM_and_zero_text']","['actual_core_deltas_zero_all_modes']")
 t=t.replace('actual_core_marker_encoded_delta_parity_zero_text_and_native_memory_pass','actual_FULL_vector_markers_core_zero_and_native_memory_pass').replace('golden/native/ordinary-core/source/memory','golden/native/minimal-FULL/source/memory')
 ast.parse(t);p=d/(stem+'_ws71.py');assert not p.exists();p.write_text(t);proof[stem]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),adapted_sha256=hashlib.sha256(t.encode()).hexdigest(),scope='Path/experiment identifiers and new learned-proof field names only; all original numeric metrics/goldens/thresholds retained. Native-route mutation audit remains same.')
(w/'pipeline-helper-adaptations.json').write_text(json.dumps(proof,indent=2)+'\n')
original=(d/'prepare_worldsense_ws70.py').read_text();t=original.replace('worldsense-primary-core-isolation','worldsense-minimal-full-prefill').replace('WS70','WS71')
t=t.replace("['both_original_GPU_readers_in_actual_CORE_EACH_words']==4200857600","['all_five_readers_in_actual_FULL_candidate'] is True")
t=t.replace("if ('primary_core_isolated_library'in call['backend'])!=enabled:return False","if 'primary_core_isolated_library'in call['backend']:return False")
t=t.replace("if after[key]-before[key]!=(encoded[oldkey]if enabled else 0):return False","if after[key]-before[key]!=0:return False").replace("if not enabled and(before[key]!=0 or after[key]!=0):return False","if before[key]!=0 or after[key]!=0:return False")
before='"  if cap in call[\'backend\']:return False"'
after='"  if cap in [\'primary_vector_prefill\',\'primary_prefill_isolated_library\',\'primary_duo_prefill\',\'primary_minimal_prefill_library\',\'primary_full_isolated_library\']:\\n   if (cap in call[\'backend\'])!=enabled:return False\\n  elif cap in call[\'backend\']:return False"'
# Only replace the generated original secondary capability loop; never the forbidden policy loop in guard.
needle='text=text.replace(needle,'+before+')';assert t.count(needle)==1;t=t.replace(needle,'text=text.replace(needle,'+after+')')
start=t.index("needle='    compiler=Path(subprocess.check_output'");end=t.index("p=w/'freeze.py'",start)
t=t[:start]+'''text=text.replace('for six builds','for six builds; actual candidate full five-reader corpus')
ast.parse(text);p.write_text(text)
'''+t[end:]
# Existing PRIMARY/FULL contrast, not the previous CORE contrast.
t=t.replace("['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY']","['GARNET_METAL_PRIMARY_BF16_BLOCKS','GARNET_METAL_WIDE_PRIMARY_STORAGE','GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']")
t=t.replace("=={'GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY':('OFF','ON')}","=={key:('OFF','ON')for key in ['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY']}")
before="['GARNET_METAL_PRIMARY_VECTOR_PREFILL','GARNET_METAL_PRIMARY_PREFILL_ISOLATED_LIBRARY','GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY','GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY','GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS']"
after="['GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY','GARNET_METAL_CACHED_MATRIX_METADATA','GARNET_METAL_DIRECT_FLOAT_WEIGHT_UPLOAD','GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY','GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY','GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS']"
assert t.count(before)==1;t=t.replace(before,after)
t=t.replace('WS71 ordinary primary-core library isolation only, original unchanged primary fragment/preamble, RAW4flags and matchedPRIMARY1flag. No vector/full/direct/K32/cached/request flags, math/weight/geometry/runtime changes. New actual-core encoded counters timed, not completed GPU or ISA proof.','WS71 existing minimalFULL+vector profile, ZERO1721sourcechanges, RAW6flags and matchedPRIMARY3flags. Direct/K32/cached/rawrequest/CORE OFF. Actual three core counters zero; same native telemetry included, no isolated causal or GPU ISA proof.')
t=t.replace('source/fourflagRAWcontrast/oneflagPRIMARYcontrast','source/sixflagRAWcontrast/threeflagPRIMARYcontrast')
ast.parse(t);(d/'prepare_worldsense_ws71.py').write_text(t)
print('WS71 future original gate/controller helpers prepared; no admission, pipeline, timing or GPU execution.')
