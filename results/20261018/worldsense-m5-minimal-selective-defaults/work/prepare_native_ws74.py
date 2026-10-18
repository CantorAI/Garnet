"""Preserve the complete native proof and independently audit the policy import."""
from pathlib import Path
import ast,json
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-minimal-selective-defaults';old=w.parent/'worldsense-minimal-full-prefill'
t=(old/'native_controller.py').read_text().replace('worldsense-minimal-full-prefill','worldsense-minimal-selective-defaults')
x="'read-only-consumer-prior-audit.json']";assert t.count(x)==1;t=t.replace(x,"'read-only-consumer-prior-audit.json','source-transformations.json','immutable57-policy-guard.txt','build.py','build-plan.json']")
x="['design_minimal_full_ws71.py','setup_minimal_full_ws71.py','prepare_native_ws71.py','verify_native_ws71.py']";assert t.count(x)==1;t=t.replace(x,"['design_selective_ws74.py','setup_selective_ws74.py','prepare_native_ws74.py','verify_native_ws74.py']")
ast.parse(t);(w/'native_controller.py').write_text(t)
v=(b/'work/worldsense-opt/verify_native_ws71.py').read_text().replace('worldsense-minimal-full-prefill','worldsense-minimal-selective-defaults');start=v.index('assert source==seed');end=v.index('assert all(sha(Path(k))',start)
block='''assert len(source)==len(seed)==1721 and source.keys()==seed.keys()and all(sha(s/k)==value for k,value in source.items())
published=json.loads(__import__('subprocess').check_output(['git','-C',str(r/'Garnet'),'show','4c811ec4e305317efac622177d4623b1cf8c83a4:results/20261018/worldsense-m5-minimal-full-prefill/correctness-source-manifest.json']))
assert published==seed
expected=['cmake/GarnetMetal.cmake','src/backends/metal/compiled_graph.cpp','src/entry/metal_main.cpp']
assert sorted(k for k in source if source[k]!=seed[k])==expected
changes=load(w/'source-transformations.json');assert sorted(x['path']for x in changes)==expected
for change in changes:
 text=(s/change['path']).read_text()
 if change.get('append'):assert text.endswith(change['after']);inverse=text[:-len(change['after'])]
 else:assert text.count(change['after'])==1;inverse=text.replace(change['after'],change['before'])
 assert hashlib.sha256(inverse.encode()).hexdigest()==published[change['path']]
policy_source=r/'out/research/worldsense-packed-only-defaults/source';policy_manifest=json.loads(__import__('subprocess').check_output(['git','-C',str(r/'Garnet'),'show','91b1eaded0ce26ad4d03210615626f175f095a1c:results/20261017/worldsense-m5-packed-only-defaults/correctness-source-manifest.json']))
assert all(sha(policy_source/k)==policy_manifest[k]for k in expected)
policy=(policy_source/expected[1]).read_text();a=policy.index('#ifdef GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS',policy.index('void MetalCompiledGraph::RetireResidentDefaults'));z=policy.index('#endif',a)+len('#endif');guard=policy[a:z]+'\\n';assert guard==(w/'immutable57-policy-guard.txt').read_text()and(s/expected[1]).read_text().count(guard)==1
cmake=(policy_source/expected[0]).read_text();cmake=cmake[cmake.index('option(GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS'):];assert next(c for c in changes if c['path']==expected[0])['after']=='\\n'+cmake
entry=(policy_source/expected[2]).read_text();a=entry.index('#ifdef GARNET_METAL_PRIMARY_ONLY_LAZY_DEFAULTS');z=entry.index('#endif',a)+len('#endif');assert next(c for c in changes if c['path']==expected[2])['after']==entry[a:z]+'\\n        return info;\\n'
assert load(w/'source-inverse-proof.json')['changed_or_added']==expected
for role in roles:
 cache=(r/('out/build/garnet-metal-worldsense-minimal-selective-defaults-'+role)/'CMakeCache.txt').read_text();flags=dict(re.findall(r'^GARNET_METAL_([A-Z0-9_]+):BOOL=(ON|OFF)$',cache,re.M));selective=role in ['primary','on','fault'];compat=role.startswith('compat56');vector=role in ['on','fault']or compat
 assert flags['PRIMARY_ONLY_LAZY_DEFAULTS']==('ON'if selective else'OFF')
 for key in ['PRIMARY_BF16_BLOCKS','WIDE_PRIMARY_STORAGE','LAZY_RESIDENT_DEFAULTS']:assert flags[key]==('OFF'if role=='off'else'ON')
 for key in ['PRIMARY_VECTOR_PREFILL','PRIMARY_PREFILL_ISOLATED_LIBRARY','PRIMARY_FULL_ISOLATED_LIBRARY']:assert flags[key]==('ON'if vector else'OFF')
 for key in ['PRIMARY_VECTOR_PREFILL_K32_ONLY','DIRECT_FLOAT_WEIGHT_UPLOAD','CACHED_MATRIX_METADATA','REQUEST_SCOPE_WEIGHT_RESIDENCY']:assert flags[key]==('ON'if compat else'OFF')
 assert flags['PRIMARY_CORE_ISOLATED_LIBRARY']=='OFF'
 assert flags['PRIMARY_TEST_FAULTS']==('ON'if role in ['fault','compat56-fault']else'OFF')and flags['REQUEST_RESIDENCY_TEST_FAULTS']==('ON'if role=='compat56-fault'else'OFF')
'''
v=v[:start]+block+v[end:];v=v.replace('source_files=1721,native_tests_pass=54','source_files=1721,exact57_three_file_policy_import_and_inverse71=True,six_role_selective_and_compatibility_flags_exact=True,native_tests_pass=54');ast.parse(v);(b/'work/worldsense-opt/verify_native_ws74.py').write_text(v)
print('WS74 native54 full streams/programs preserved; independent threefile policy inverse and sixrole flag audit prepared, no native/model dispatched.')
