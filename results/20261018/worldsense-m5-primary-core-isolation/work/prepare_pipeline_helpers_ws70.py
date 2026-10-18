"""Adapt future pipeline helpers in new files; do not execute dependent stages."""
from pathlib import Path
import ast,json,hashlib
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');d=b/'work/worldsense-opt';proof={}
for stem in ['prepare_functional','verify_functional','prepare_resource','verify_resource','prepare_actual_speed','verify_screen']:
 original=(d/(stem+'_ws69.py')).read_text();text=original.replace('worldsense-packed-request-residency','worldsense-primary-core-isolation').replace('ws69.py','ws70.py').replace('WS69','WS70');ops=[]
 def change(x,y,count=1):
  global text
  assert text.count(x)==count,(stem,x,text.count(x));text=text.replace(x,y);ops.append(dict(before=x,after=y,count=count))
 if stem in ['prepare_resource','verify_screen']:change('both_original_GPU_readers_under_actual_lease_EACH_words','both_original_GPU_readers_in_actual_CORE_EACH_words')
 if stem=='verify_functional':
  change("packed_request=c['timings']['packed_request_weight_residency'],post_cleanup_memory=c['timings']['memory_after_packed_request_cleanup'],","core_deltas={key:c['timings']['primary_storage_after'][key]-c['timings']['primary_storage_before'][key]for key in ['core_library_decode_encoded','core_library_prefill_encoded','core_library_head_encoded']},")
  text=text.replace('actual_packed_request_marker_allocation_cleanup_companion_skip_and_native_memory_pass','actual_core_marker_encoded_delta_parity_zero_text_and_native_memory_pass').replace('golden/native/packed-lease/source/memory','golden/native/ordinary-core/source/memory')
 if stem=='verify_resource':change("packed_request=c['timings'].get('packed_request_weight_residency'),post_cleanup_memory=c['timings'].get('memory_after_packed_request_cleanup')","core_before=c['timings']['primary_storage_before'],core_after=c['timings']['primary_storage_after'],primary_delta=c['timings']['primary_encoded_delta']")
 if stem=='verify_screen':
  change("assert load(w/'packed-full-session-guard.json')['all_cases_pass']\n",'')
  change("packed_request=t.get('packed_request_weight_residency'),post_cleanup_memory=t.get('memory_after_packed_request_cleanup')","core_deltas=({key:t['primary_storage_after'][key]-t['primary_storage_before'][key]for key in ['core_library_decode_encoded','core_library_prefill_encoded','core_library_head_encoded']}if'primary_storage_before'in t else None),primary_encoded_delta=t.get('primary_encoded_delta')")
  text=text.replace('isolated residency-hint, traversal, attachment, request, coverage or cleanup cost','isolated library-boundary/compiler/driver-memory/counter cost')
 p=d/(stem+'_ws70.py');assert not p.exists();ast.parse(text);p.write_text(text);proof[stem]=dict(seed_sha256=hashlib.sha256(original.encode()).hexdigest(),adapted_sha256=hashlib.sha256(text.encode()).hexdigest(),operations=ops)
(b/'work/worldsense-primary-core-isolation/pipeline-helper-adaptations.json').write_text(json.dumps(proof,indent=2)+'\n');print('WS70 future original pipeline preparation/verification helpers ready; no stage executed')
