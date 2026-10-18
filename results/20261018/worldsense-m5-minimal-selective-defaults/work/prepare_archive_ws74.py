"""Prepare canonical reconstruction of the three-file policy/profile experiment."""
from pathlib import Path
import ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');f=b/'work/worldsense-opt'
def adapt(t):return t.replace('worldsense-minimal-full-prefill','worldsense-minimal-selective-defaults').replace('worldsense-m5-minimal-full-prefill','worldsense-m5-minimal-selective-defaults').replace('WS-M5-71','WS-M5-74').replace('WS71','WS74').replace('ws71','ws74')
t=adapt((f/'archive_ws71.py').read_text()).replace("w.parent/'worldsense-primary-core-isolation'","w.parent/'worldsense-minimal-full-prefill'")
t=t.replace('Valid rejection: textTTFT and4Bderiveddecode fail unchanged2%gates despite favorable observed4BTTFT/total/wholeSkill. No later stages/promotion.','Valid rejection: BOTHmodels derived generation rates fail unchanged2%gates despite favorable observedlatencies/wholeSkill. No later stages/promotion.')
ast.parse(t);p=f/'archive_ws74.py';assert not p.exists();p.write_text(t)
v=adapt((f/'verify_archive_ws71.py').read_text());v=v.replace("seed_ref='80787c104fdafe7eb4671cc6c7b390bfcce2fb1f';seed_prefix='results/20261018/worldsense-m5-primary-core-isolation/'","seed_ref='4c811ec4e305317efac622177d4623b1cf8c83a4';seed_prefix='results/20261018/worldsense-m5-minimal-full-prefill/'")
x="assert current==seed_source and not load(d/'source-inverse-proof.json')['changed_or_added']";y='''changed=['cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/entry/metal_main.cpp']
assert current.keys()==seed_source.keys()and sorted(k for k in current if current[k]!=seed_source[k])==changed
operations=load(d/'source-transformations.json');assert sorted(c['path']for c in operations)==changed
for change in operations:
 text=current[change['path']].decode()
 if change.get('append'):assert text.endswith(change['after']);inverse=text[:-len(change['after'])]
 else:assert text.count(change['after'])==1;inverse=text.replace(change['after'],change['before'])
 assert inverse.encode()==seed_source[change['path']]
assert load(d/'source-inverse-proof.json')['changed_or_added']==changed
policy_ref='91b1eaded0ce26ad4d03210615626f175f095a1c';policy_prefix='results/20261017/worldsense-m5-packed-only-defaults/'
policy_manifest=json.loads(batch([policy_ref+':'+policy_prefix+'correctness-source-manifest.json'])[0]);policy=dict(zip(changed,batch([policy_ref+':'+policy_prefix+'source/'+k for k in changed])))
assert all(v is not None and hashlib.sha256(v).hexdigest()==policy_manifest[k]for k,v in policy.items())
assert load(d/'read-only-consumer-prior-audit.json')['policy_source_hashes']=={k:policy_manifest[k]for k in changed}
text=policy[changed[1]].decode();start=text.index('#ifdef GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS',text.index('void MetalCompiledGraph::RetireResidentDefaults'));end=text.index('#endif',start)+len('#endif');guard=text[start:end]+'\\n'
assert guard==(d/'immutable57-policy-guard.txt').read_text()and current[changed[1]].decode().count(guard)==1
text=policy[changed[0]].decode();block=text[text.index('option(GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS'):];assert next(c for c in operations if c['path']==changed[0])['after']=='\\n'+block
text=policy[changed[2]].decode();start=text.index('#ifdef GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS');end=text.index('#endif',start)+len('#endif');assert next(c for c in operations if c['path']==changed[2])['after']==text[start:end]+'\\n        return info;\\n'
''';assert v.count(x)==1;v=v.replace(x,y)
x=" assert ('GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS:BOOL=ON'in cache)==(role=='compat56-fault')";y=x+'''
 flags=dict(re.findall(r'^GARNET_METAL_([A-Z0-9_]+):BOOL=(ON|OFF)$',cache,re.M));selective=role in ['primary','on','fault'];compat=role.startswith('compat56');vector=role in ['on','fault']or compat
 assert flags['PRIMARY_ONLY_LAZY_DEFAULTS']==('ON'if selective else'OFF')
 for key in ['PRIMARY_BF16_BLOCKS','WIDE_PRIMARY_STORAGE','LAZY_RESIDENT_DEFAULTS']:assert flags[key]==('OFF'if role=='off'else'ON')
 for key in ['PRIMARY_VECTOR_PREFILL','PRIMARY_PREFILL_ISOLATED_LIBRARY','PRIMARY_FULL_ISOLATED_LIBRARY']:assert flags[key]==('ON'if vector else'OFF')
 for key in ['PRIMARY_VECTOR_PREFILL_K32_ONLY','DIRECT_FLOAT_WEIGHT_UPLOAD','CACHED_MATRIX_METADATA','REQUEST_SCOPE_WEIGHT_RESIDENCY']:assert flags[key]==('ON'if compat else'OFF')
 assert flags['PRIMARY_TEST_FAULTS']==('ON'if role in ['fault','compat56-fault']else'OFF')
''';assert v.count(x)==1;v=v.replace(x,y)
v=v.replace("value.replace('worldsense-minimal-selective-defaults','worldsense-primary-core-isolation')==original","value.replace('worldsense-minimal-selective-defaults','worldsense-minimal-full-prefill')==original")
a=v.index('# Prove the inherited immutable70');z=v.index('def canonical_learned_checks():',a)
chain='''# Recursively prove the unchanged inherited71->70->56 mathematical guards.
for child_ref,child_prefix,child_slug,parent_ref,parent_prefix,parent_slug in [
 (seed_ref,seed_prefix,'worldsense-minimal-full-prefill','80787c104fdafe7eb4671cc6c7b390bfcce2fb1f','results/20261018/worldsense-m5-primary-core-isolation/','worldsense-primary-core-isolation'),
 ('80787c104fdafe7eb4671cc6c7b390bfcce2fb1f','results/20261018/worldsense-m5-primary-core-isolation/','worldsense-primary-core-isolation','290f5742ba51c41197f24179cfee4876e96e789a','results/20261017/worldsense-m5-request-weight-residency/','worldsense-request-weight-residency')]:
 parent_proof=json.loads(batch([child_ref+':'+child_prefix+'learned-guard-adaptation-proof.json'])[0])
 for name,info in parent_proof.items():
  value=batch([child_ref+':'+child_prefix+name])[0].decode();original=batch([parent_ref+':'+parent_prefix+name])[0].decode()
  assert hashlib.sha256(value.encode()).hexdigest()==info['current_sha256']and hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']
  for op in reversed(info['operations']):assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
  assert value.replace(child_slug,parent_slug)==original
''';v=v[:a]+chain+v[z:]
x="   assert timings['lazy_default_rehydrated_bytes']==0 and(timings['lazy_default_retired_bytes']>0)==enabled and(timings['lazy_default_planned_slots']>0)==enabled";y="   selective=role in ['primary','on'];retirement=enabled and (not selective or index<2)\n   assert ('primary_only_lazy_defaults'in call['backend'])==selective\n   assert timings['lazy_default_rehydrated_bytes']==0 and(timings['lazy_default_retired_bytes']>0)==retirement and(timings['lazy_default_planned_slots']>0)==retirement";assert v.count(x)==1;v=v.replace(x,y)
v=v.replace("len(load(d/'flag-dependency-proof.json'))==15","len(load(d/'flag-dependency-proof.json'))==18").replace('==21202','==22792').replace('offline_malformed_copies_rejected=21202','offline_malformed_copies_rejected=22792')
v=v.replace('original_programs_and_runtime_shader_math_CUDA_engine_test_bytes_preserved=True','exact57_three_policy_imports_inverse_to71=True,original_programs_and_runtime_shader_math_CUDA_engine_test_bytes_preserved=True').replace('original_guard_and_all_five_metric_gate_ASTs_inverse_exact=True','original_guard74_to71_to70_to56_and_all_five_metric_gate_ASTs_inverse_exact=True')
ast.parse(v);p=f/'verify_archive_ws74.py';assert not p.exists();p.write_text(v)
print('WS74 canonical archive/verifier prepared: threefile71inverse/exact57imports, sixroles/native54, fullguards74->71->70->56, original5metricASTs/18configs/22792negatives/32pipelinecalls. No copies or inference yet.')
