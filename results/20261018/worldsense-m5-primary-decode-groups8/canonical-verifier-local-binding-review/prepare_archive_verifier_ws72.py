"""Canonical groups8 source/native/learned/metric reconstruction, no inference."""
from pathlib import Path
import ast,textwrap
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');f=b/'work/worldsense-opt';p=(f/'verify_archive_ws71.py').read_text();t=p[:p.index('assert current==seed_source')].replace('worldsense-m5-minimal-full-prefill','worldsense-m5-primary-decode-groups8').replace('len(current)==1721','len(current)==1723').replace("seed_ref='80787c104fdafe7eb4671cc6c7b390bfcce2fb1f';seed_prefix='results/20261018/worldsense-m5-primary-core-isolation/'","seed_ref='4c811ec4e305317efac622177d4623b1cf8c83a4';seed_prefix='results/20261018/worldsense-m5-minimal-full-prefill/'")
t+='''delta=sorted(key for key,value in current.items()if seed_source.get(key)!=value)
assert delta==sorted(['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','src/backends/metal/primary_decode_groups8_shader.h','test2026/metal/primary_decode_groups8_test.cpp'])
transforms=load(d/'source-transformations.json')
for rel,operations in transforms.items():
 value=current[rel].decode()
 for op in reversed(operations):
  if op.get('append'):assert value.endswith(op['after']);value=value[:-len(op['after'])]
  else:assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
 assert value.encode()==seed_source[rel]
assert all(current[key]==seed_source[key]for key in seed if key not in transforms)
assert len(recon['overlay_files'])==18 and len(legacy_recon['overlay_files'])==9
'''
a=p.index("roles=['off','primary','on','fault','compat56','compat56-fault']");z=p.index('# Prove the inherited immutable70',a);native=p[a:z]
native=native.replace("r'\\d+/9\\s+Test","r'\\d+/10\\s+Test").replace('list(range(1,10))','list(range(1,11))').replace('out of 9','out of 10')
needle=" assert ('GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY:BOOL=ON'in cache)==role.startswith('compat56')";assert native.count(needle)==1;native=native.replace(needle," assert ('GARNET_METAL_PRIMARY_DECODE_GROUPS8:BOOL=ON'in cache)==(role in ['on','fault'])\n"+needle)
new=(f/'verify_native_ws72.py').read_text();new=new[new.index('cases=[]'):new.index('programs={role:{kind:')]
new=new.replace("build/'primary_decode_groups8_words.bin'","d/('native-'+role+'-primary_decode_groups8_words.bin')").replace("build/'primary_decode_groups8_words.bin.summary.json'","d/('native-'+role+'-primary_decode_groups8_words.bin.summary.json')").replace('load(w/','load(d/')
needle='programs={role:{kind:';assert native.count(needle)==1;native=native.replace(needle,new+needle)
before="for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==full_program.encode()==programs['compat56']['prefill']"
after='''original_primary=current['src/backends/metal/primary_block_shader.h'].decode().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]
assert original_primary.count('ulong column=ulong(group)*16+simd*4;')==2
selected_primary=original_primary.replace('ulong column=ulong(group)*16+simd*4;','ulong column=ulong(group)*32+simd*4;');selected_full=full_program.replace(original_primary,selected_primary)
for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==selected_full.encode()
for role in roles:assert(d/('actual-'+role+'-groups8-selected-primary.metal')).read_bytes()==(selected_primary if role in ['on','fault']else original_primary).encode()
assert load(d/'pre-native-generic-fixture-review/correctness-source-manifest.json')['test2026/metal/primary_decode_groups8_test.cpp']!=load(d/'correctness-source-manifest.json')['test2026/metal/primary_decode_groups8_test.cpp']'''
assert native.count(before)==1;native=native.replace(before,after).replace("value.replace('worldsense-minimal-full-prefill','worldsense-primary-core-isolation')","value.replace('worldsense-primary-decode-groups8','worldsense-minimal-full-prefill')");t+=native
t+='''# Recover each immutable inherited guard generation, not just the latest admission.
chains=[('4c811ec4e305317efac622177d4623b1cf8c83a4','results/20261018/worldsense-m5-minimal-full-prefill/','80787c104fdafe7eb4671cc6c7b390bfcce2fb1f','results/20261018/worldsense-m5-primary-core-isolation/','worldsense-minimal-full-prefill','worldsense-primary-core-isolation'),('80787c104fdafe7eb4671cc6c7b390bfcce2fb1f','results/20261018/worldsense-m5-primary-core-isolation/','290f5742ba51c41197f24179cfee4876e96e789a','results/20261017/worldsense-m5-request-weight-residency/','worldsense-primary-core-isolation','worldsense-request-weight-residency')]
for child,child_prefix,parent,parent_prefix,child_slug,parent_slug in chains:
 proof=json.loads(batch([child+':'+child_prefix+'learned-guard-adaptation-proof.json'])[0])
 for name,info in proof.items():
  value=batch([child+':'+child_prefix+name])[0].decode();original=batch([parent+':'+parent_prefix+name])[0].decode()
  assert hashlib.sha256(value.encode()).hexdigest()==info['current_sha256']and hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']
  for op in reversed(info['operations']):assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
  assert value.replace(child_slug,parent_slug)==original
'''
learned=(f/'verify_learned_ws72.py').read_text();core=learned[learned.index("terminal=load(w/'learned-controller-terminal.json')"):learned.index('result=dict(')]
for old,new in [("w.parent/'worldsense-primary-blocks/matrix-selection.json'","d/'frozen-native-guards/matrix-selection.json'"),("w.parent/'worldsense-prefix-cache/state-guard.json'","d/'frozen-native-guards/state-guard.json'"),("w.parent/'worldsense-prefix-cache/prefix-model-guard.json'","d/'frozen-native-guards/prefix-model-guard.json'"),("w.parent/'worldsense-request-residency/vision-grid-off.json'","d/'frozen-native-guards/vision-grid-original.json'")]:core=core.replace(old,new)
core=core.replace('load(w/','load(d/').replace('(w/','(d/');t+='def canonical_learned_checks():\n'+textwrap.indent(core,' ')+'\ncanonical_learned_checks()\n'
tail=p[p.index("assert load(d/'native-independent-audit.json')['observed_exit_code']"):]
tail=tail.replace('worldsense-minimal-full-prefill','worldsense-primary-decode-groups8').replace('audit_routes_ws71.py','audit_routes_ws72.py').replace('==21202','==23617').replace('==1875','==1877').replace("len(load(d/'flag-dependency-proof.json'))==15","len(load(d/'flag-dependency-proof.json'))==24")
needle="scope=load(d/'export-scope.json');";extra='''grows=list(csv.DictReader((d/'groups8-dispatch-deltas.csv').open()));assert len(grows)==32
glookup={(row['stage'],row['worker'],int(row['repetition']),row['phase']):row for row in grows};group_available=0;group_vlm=0
for stage,label,index,phase,item,call in rawcalls:
 row=glookup[(stage,label,index,phase)];timings=call['timings'];present='primary_storage_before'in timings;enabled=label=='candidate-1'
 assert row['counter_telemetry_available']==str(present)
 if present:group_available+=1
 for key,oldkey in [('decode_groups8_encoded','decode_encoded'),('head_groups8_encoded','head_encoded')]:
  if not present:assert row[key]=='';continue
  actual=timings['primary_storage_after'][key]-timings['primary_storage_before'][key];assert int(row[key])==actual==(timings['primary_encoded_delta'][oldkey]if enabled else 0)
 if present and enabled and call['model_id']=='Qwen3.5-4B':group_vlm+=1;assert int(row['decode_groups8_encoded'])>0 and int(row['head_groups8_encoded'])>0
assert group_available==22 and group_vlm==9
scope=load(d/'export-scope.json');assert scope['groups8_delta_rows']==32
''';assert tail.count(needle)==1;tail=tail.replace(needle,extra)
tail=tail.replace('source_files=1721','source_files=1723').replace('native_CTest_pass=54','native_CTest_pass=60').replace('original_two_program_FULL_partition_exact_and_CORE_getter_empty=True','original_two_program_FULL_partition_exact_and_CORE_getter_empty=True,new60_complete804504words_generic_specialized_heads_tails_exact=True,selected_shader_only_two_output_origin_substitutions=True,newbody_head_counters_match_original_and_zero_text=True').replace('offline_malformed_copies_rejected=21202','offline_malformed_copies_rejected=23617').replace('core_counter_available_rows=22','core_counter_available_rows=22,groups8_delta_rows=32,groups8_available_rows=22').replace('ws71-archive-independent-audit.json','ws72-archive-independent-audit.json')
t+=tail;ast.parse(t);target=f/'verify_archive_ws72.py';assert not target.exists();target.write_text(t);print('WS72 canonical verifier ready: sourceinverse/18overlays, native60/original186+20/new60words/programs, guardchain71->70->56, fulllearned/newgroups,23617negatives/24configs,32pipelinecalls/CSV/metrics/rejection. No inference.')
