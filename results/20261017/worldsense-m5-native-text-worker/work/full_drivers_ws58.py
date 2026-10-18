from pathlib import Path
import json,hashlib,ast
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-native-text-worker';old=w.parent/'worldsense-isolated-primary'
names=['build_corpus.py','build_direct_corpus.py','make_resident_fixtures.py','resident_controller.py','resident_guard.py','linear_guard.py','vision_grid_guard.py','api_state_guard.py','full_correctness.py','verify_correctness_terminal.py']
for name in names:
 text=(old/name).read_text().replace('worldsense-isolated-primary','worldsense-native-text-worker').replace('correctness-source-manifest.json','gateway-source-manifest.json').replace('1718','1723')
 if name in ['full_correctness.py','verify_correctness_terminal.py']:
  text=text.replace("program['actual_candidate_main_equals_raw_control']and program['actual_candidate_prefill_equals_minimal_preamble_plus_full_primary_and_vector_fragment']","program['all_four_actual_main_and_prefill_programs_byte_exact_WS54']")
 (w/name).write_text(text)
for name in ['primary_corpus_test.cpp','direct_float_corpus_test.cpp','direct-upload-selection.json']:(w/name).write_bytes((old/name).read_bytes())
p=w/'api_state_guard.py';text=p.read_text();text=text.replace(" m=metal_serving._models[name];result=[]", " if name=='Qwen3-1.7B'and mode!='off':return json.loads(garnet.native_text_worker_state_json(name))['states']\n m=metal_serving._models[name];result=[]")
text=text.replace("assert name in metal_serving._models", "assert (name in metal_serving._models)==(name!='Qwen3-1.7B'or mode=='off')\n if name=='Qwen3-1.7B'and mode!='off':assert json.loads(garnet.native_text_worker_info_json(''))['children'][0]['models']==[name]")
start=text.index('def infer(');end=text.index("load('Qwen3.5-4B'");part=text[start:end];part=part.replace(" call=json.loads", " remote=name=='Qwen3-1.7B'and mode!='off';packed_expected=bool(expected)and not remote;vector_expected=mode=='on'and not remote\n call=json.loads",1).replace('bool(expected)','bool(packed_expected)')
# Do not turn the assignment into a self-reference.
part=part.replace('packed_expected=bool(packed_expected)and not remote','packed_expected=bool(expected)and not remote').replace('if expected and name','if packed_expected and name').replace('elif expected:','elif packed_expected:')
startgraph=part.index(' graphs=[');endgraph=part.index(" assert ('wide_primary_storage'",startgraph)
original=part[startgraph:endgraph]
part=part[:startgraph]+" if not remote:\n"+''.join(' '+line+'\n'for line in original.rstrip('\n').split('\n'))+" else:\n  child=json.loads(garnet.native_text_worker_info_json(''))['children'][0];assert child['models']==[name]and 'wide_primary_storage'not in child['backend']\n"+part[endgraph:]
# Each remote response is RAW; its parent is independently verified as the selected role.
needle=" timings=call['timings'];"
part=part.replace(needle," if mode!='off':\n  gateway=call['timings']['native_gateway'];assert gateway['route']==('native_lrpc_raw_text'if remote else'local')and 'native_text_worker'in gateway['parent_backend'];assert ('primary_vector_prefill'in gateway['parent_backend'])==(mode=='on')\n"+needle,1)
text=text[:start]+part+text[end:];text=text.replace("print('Real native load", "garnet.shutdown()\nprint('Real native load");p.write_text(text)
p=w/'verify_correctness_terminal.py';text=p.read_text();text=text.replace("  t=b['call']['timings'];", "  remote=b['model_id']=='Qwen3-1.7B';packed_expected=not remote;vector_expected=mode=='on'and not remote\n  t=b['call']['timings'];");text=text.replace("==(mode!='off')","==packed_expected").replace("==(mode=='on')","==vector_expected");p.write_text(text)
for name in names:ast.parse((w/name).read_text())
(w/'full-driver-adaptation.json').write_text(json.dumps(dict(original_driver_source='WS54 verified source and original gates',driver_changes='Paths/source count/proof field; text state/load ownership now queried from owned native RAW child. Text raw policy markers must be absent; parent role separately checked. Local VLM graph assertions and all original numerical/state/token/corpus guards retained. Original direct graph tests still cover RAW ordinary/resident eligibility locally.',source_changed=False,full_guard_not_executed=True,driver_sha256={name:hashlib.sha256((w/name).read_bytes()).hexdigest()for name in names}),indent=2)+'\n')
print('Prepared full correctness drivers, unchanged numerical gates; no dispatch')
