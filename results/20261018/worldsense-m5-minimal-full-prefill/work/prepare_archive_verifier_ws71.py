"""Canonical exact-source/profile/full numerical/data verifier, no model execution."""
from pathlib import Path
import ast,textwrap
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');f=b/'work/worldsense-opt';prior=(f/'verify_archive_ws70.py').read_text()
t=prior[:prior.index('delta=sorted(')].replace('worldsense-m5-primary-core-isolation','worldsense-m5-minimal-full-prefill').replace("seed_ref='290f5742ba51c41197f24179cfee4876e96e789a';seed_prefix='results/20261017/worldsense-m5-request-weight-residency/'","seed_ref='80787c104fdafe7eb4671cc6c7b390bfcce2fb1f';seed_prefix='results/20261018/worldsense-m5-primary-core-isolation/'").replace('and len(seed)==1720','and len(seed)==1721')
t+='''assert current==seed_source and not load(d/'source-inverse-proof.json')['changed_or_added']
assert len(recon['overlay_files'])==16 and len(legacy_recon['overlay_files'])==9
'''
a=prior.index("roles=['off','primary','on','fault','compat56','compat56-fault']");z=prior.index('def canonical_learned_checks():',a)
native=prior[a:z]
native=native.replace("('GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY:BOOL=ON'in cache)==(role in ['on','fault'])","('GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY:BOOL=ON'in cache) is False")
native=native.replace("seed_prefix+'native-off-projection-words.bin'","seed_prefix+'native-off-primary_projection_words.bin'")
native=native.replace("summary['core_enabled']==(role in ['on','fault'])","summary['core_enabled'] is False").replace("([8,8,2]if role in ['on','fault']else[0,0,0])","[0,0,0]")
native=native.replace("('compat56','on'),('compat56-fault','fault')","('compat56','compat56'),('compat56-fault','compat56-fault')")
a=native.index('core_program=');z=native.index("for role,groups in load(d/'actual-program-proof.json')",a)
native=native[:a]+'''full_program=current['src/backends/metal/primary_isolation_shader.h'].decode().split('R"IPRIMARY(',1)[1].split(')IPRIMARY"',1)[0]+current['src/backends/metal/primary_block_shader.h'].decode().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]+current['src/backends/metal/primary_vector_prefill_shader.h'].decode().split('R"PVEC(',1)[1].split(')PVEC"',1)[0]
for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==full_program.encode()==programs['compat56']['prefill']
assert all(not programs[role]['core']for role in roles)
'''+native[z:]
native=native.replace("value.replace('worldsense-primary-core-isolation','worldsense-request-weight-residency')","value.replace('worldsense-minimal-full-prefill','worldsense-primary-core-isolation')")
t+=native
t+='''# Prove the inherited immutable70 guard transformations still recover original56.
parent_proof=json.loads(batch([seed_ref+':'+seed_prefix+'learned-guard-adaptation-proof.json'])[0])
ancestor='290f5742ba51c41197f24179cfee4876e96e789a:results/20261017/worldsense-m5-request-weight-residency/'
for name,info in parent_proof.items():
 value=batch([seed_ref+':'+seed_prefix+name])[0].decode();original=batch([ancestor+name])[0].decode()
 assert hashlib.sha256(value.encode()).hexdigest()==info['current_sha256']and hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']
 for op in reversed(info['operations']):assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
 assert value.replace('worldsense-primary-core-isolation','worldsense-request-weight-residency')==original
'''
learned=(f/'verify_learned_ws71.py').read_text();core=learned[learned.index("terminal=load(w/'learned-controller-terminal.json')"):learned.index('result=dict(')]
for old,new in [("w.parent/'worldsense-primary-blocks/matrix-selection.json'","d/'frozen-native-guards/matrix-selection.json'"),("w.parent/'worldsense-prefix-cache/state-guard.json'","d/'frozen-native-guards/state-guard.json'"),("w.parent/'worldsense-prefix-cache/prefix-model-guard.json'","d/'frozen-native-guards/prefix-model-guard.json'"),("w.parent/'worldsense-request-residency/vision-grid-off.json'","d/'frozen-native-guards/vision-grid-original.json'")]:core=core.replace(old,new)
core=core.replace('load(w/','load(d/').replace('(w/','(d/')
t+='def canonical_learned_checks():\n'+textwrap.indent(core,' ')+'\ncanonical_learned_checks()\n'
tail=prior[prior.index("assert load(d/'native-independent-audit.json')['observed_exit_code']"):]
tail=tail.replace('worldsense-primary-core-isolation','worldsense-minimal-full-prefill').replace('audit_routes_ws70.py','audit_routes_ws71.py').replace('==21122','==21202').replace('==1877','==1875')
tail=tail.replace("actual==(t['primary_encoded_delta'][oldkey]if enabled else 0)","actual==0")
tail=tail.replace("if present and enabled and call['model_id']=='Qwen3.5-4B':positive_vlm+=1","if present and enabled and call['model_id']=='Qwen3.5-4B':\n  positive_vlm+=1\n  assert all(t['primary_encoded_delta'][key]>0 for key in ['decode_encoded','prefill_encoded','head_encoded'])")
tail=tail.replace('original_two_program_contract_and_separate_core_getter_partition_exact=True','original_two_program_FULL_partition_exact_and_CORE_getter_empty=True').replace('original_five_full_learned_readers_and_two_CORE_readers_states_tokens_vision_lifecycle_exact=True','original_five_full_learned_readers_in_actual_candidate_states_tokens_vision_lifecycle_exact=True').replace('offline_malformed_copies_rejected=21122','offline_malformed_copies_rejected=21202').replace('positive_CORE_VLM_calls=9','actual_candidate_VLM_route_records=9,positive_CORE_VLM_calls=0').replace('ws70-archive-independent-audit.json','ws71-archive-independent-audit.json')
t+=tail
ast.parse(t);p=f/'verify_archive_ws71.py';assert not p.exists();p.write_text(t)
print('WS71 canonical verifier ready: zero-source delta to immutable70, native54/complete words, original56guard chain/full learned,21202offline negatives,32actualpipelinecalls/CSV/metrics/rejection. No inference.')
