"""Prepare original complete native proof for exact-source profile composition."""
from pathlib import Path
import json,ast,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-minimal-full-prefill';old=w.parent/'worldsense-primary-core-isolation'
shutil.copy2(old/'primary_core_isolation_test.cpp',w/'primary_core_isolation_test.cpp')
t=(old/'native_controller.py').read_text().replace('worldsense-primary-core-isolation','worldsense-minimal-full-prefill')
def replace(before,after):
 global t
 assert t.count(before)==1,(before,t.count(before));t=t.replace(before,after)
replace("'source-transformations.json',",'')
replace("'diagnostic-api-amendment-before-native.txt',",'')
replace("['design_core_isolation_ws70.py','setup_core_isolation_ws70.py','repair_core_program_contract_ws70.py','verify_native_ws70.py']","['design_minimal_full_ws71.py','setup_minimal_full_ws71.py','prepare_native_ws71.py','verify_native_ws71.py']")
replace("summary['core_enabled']==(role in ['on','fault'])","summary['core_enabled'] is False")
replace("([8,8,2]if role in ['on','fault']else[0,0,0])","[0,0,0]")
replace("core=(s/'src/backends/metal/primary_isolation_shader.h').read_text().split('R\"IPRIMARY(',1)[1].split(')IPRIMARY\"',1)[0]+(s/'src/backends/metal/primary_block_shader.h').read_text().split('R\"PRIMARY(',1)[1].split(')PRIMARY\"',1)[0]","full=(s/'src/backends/metal/primary_isolation_shader.h').read_text().split('R\"IPRIMARY(',1)[1].split(')IPRIMARY\"',1)[0]+(s/'src/backends/metal/primary_block_shader.h').read_text().split('R\"PRIMARY(',1)[1].split(')PRIMARY\"',1)[0]+(s/'src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R\"VPREFILL(',1)[1].split(')VPREFILL\"',1)[0]")
replace("for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and not programs[role]['prefill']and programs[role]['core']==core.encode()","for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==full.encode()and programs[role]['prefill']==programs['compat56']['prefill']")
replace("assert all(not programs[role]['core']for role in ['off','primary','compat56','compat56-fault'])","assert all(not programs[role]['core']for role in roles)")
replace('ON_FAULT_MAIN_exact_RAW_and_vector_prefill_contract_empty=True,ON_FAULT_core_exact_unchanged_preamble_plus_primary_fragment=True','ON_FAULT_MAIN_exact_RAW=True,ON_FAULT_prefill_exact_unchanged_preamble_primary_vector_fragments=True,CORE_getter_and_actual_route_counters_zero_ALL_ROLES=True')
ast.parse(t);(w/'native_controller.py').write_text(t)
v=(old.parent/'worldsense-opt/verify_native_ws70.py').read_text().replace('worldsense-primary-core-isolation','worldsense-minimal-full-prefill')
start=v.index('assert len(source)==');end=v.index('assert all(sha(Path(k))',start)
v=v[:start]+'''assert source==seed and len(source)==1721 and all(sha(s/k)==value for k,value in source.items())
published=json.loads(__import__('subprocess').check_output(['git','-C',str(r/'Garnet'),'show','80787c104fdafe7eb4671cc6c7b390bfcce2fb1f:results/20261018/worldsense-m5-primary-core-isolation/correctness-source-manifest.json']))
assert published==source
assert not load(w/'source-inverse-proof.json')['changed_or_added']
'''+v[end:]
v=v.replace("summary['core_enabled']==(role in ['on','fault'])","summary['core_enabled'] is False").replace("([8,8,2]if role in ['on','fault']else[0,0,0])","[0,0,0]")
a=v.index('core=(s/');z=v.index("for role,group in load(w/'actual-program-proof.json')",a)
block=t[t.index(' full=(s/'):t.index(" (w/'actual-program-proof.json')")]
v=v[:a]+''.join(line[1:]+'\n'for line in block.splitlines())+v[z:]
v=v.replace('actual_core_route_counts_decode8_prefill8_head2_and_raw_zero=True,actual_CORE_MAIN_exact_RAW_and_CORE_exact_original_fragment_preamble=True','all_actual_core_route_counters_and_getter_zero=True,actual_FULL_MAIN_exact_RAW_and_secondary_exact_original_fragments=True')
ast.parse(v);(b/'work/worldsense-opt/verify_native_ws71.py').write_text(v)
print('WS71 native54/original186+20complete streams, exactFULLprograms/corezero proof prepared, no native/model dispatched.')
