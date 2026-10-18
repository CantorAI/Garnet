"""Prepare later pipeline helpers in new files, without running dependent stages."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');d=b/'work/worldsense-opt';proof={}
for stem in ['prepare_functional','verify_functional','prepare_resource','verify_resource','prepare_actual_speed']:
 seed=d/(stem+'_ws68.py');original=seed.read_text();text=original.replace('worldsense-primary-simd-metadata','worldsense-packed-request-residency').replace('ws68.py','ws69.py').replace('WS68','WS69');ops=[]
 def change(x,y,count=1):
  global text
  assert text.count(x)==count,(stem,x,text.count(x));text=text.replace(x,y);ops.append(dict(before=x,after=y,count=count))
 if stem=='prepare_functional':
  change("['full_four_actual_API_modes20calls_original32VLM28text_states_tokens_vision_exact']","['four_API_modes20calls_original32VLM28text_states_tokens_vision_exact']")
  change('len(m)==1726','len(m)==1721')
  change("['off','primary','on','fault','compat67']","['off','primary','on','fault','compat56','compat56-fault']")
 if stem=='verify_functional':
  change("sole_storage=c['timings']['sole_storage_after'],sole_encoded_delta=c['timings']['sole_encoded_delta'],", "packed_request=c['timings']['packed_request_weight_residency'],post_cleanup_memory=c['timings']['memory_after_packed_request_cleanup'],")
  text=text.replace('actual_primary_SIMD_marker_counters_and_zero_sole_routes_native_memory_pass','actual_packed_request_marker_allocation_cleanup_companion_skip_and_native_memory_pass').replace('golden/native/SIMD/source/memory','golden/native/packed-lease/source/memory')
 if stem=='prepare_resource':
  change("['both_new_GPU_helpers_EACH_all201_original_FP32_words']","['both_original_GPU_readers_under_actual_lease_EACH_words']")
  change("['native_tests_pass']==50","['native_tests_pass']==54")
  change("['off','primary','on','fault','compat67']","['off','primary','on','fault','compat56','compat56-fault']")
 if stem=='verify_resource':
  change("sole_after=c['timings']['sole_storage_after'],sole_delta=c['timings']['sole_encoded_delta']","packed_request=c['timings'].get('packed_request_weight_residency'),post_cleanup_memory=c['timings'].get('memory_after_packed_request_cleanup')")
 p=d/(stem+'_ws69.py');assert not p.exists(),p;ast.parse(text);p.write_text(text);proof[stem]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),adapted_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops)
(b/'work/worldsense-packed-request-residency/pipeline-helper-adaptations.json').write_text(json.dumps(proof,indent=2)+'\n')
print('New WS69 future pipeline preparation/verification helpers written; no stage executed')
