"""Prepare original54 plus six complete grouping reference checks before dispatch."""
from pathlib import Path
import json,ast,shutil,textwrap
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');d=b/'work/worldsense-opt';w=b/'work/worldsense-primary-decode-groups8';old=w.parent/'worldsense-minimal-full-prefill'
shutil.copy2(old/'primary_core_isolation_test.cpp',w/'primary_core_isolation_test.cpp')
t=(old/'native_controller.py').read_text().replace('worldsense-minimal-full-prefill','worldsense-primary-decode-groups8').replace('len(manifest)==1721','len(manifest)==1723').replace('Native9tests','Native10tests').replace('ALL6ROLES9TESTS','ALL6ROLES10TESTS')
t=t.replace("['design_minimal_full_ws71.py','setup_minimal_full_ws71.py','prepare_native_ws71.py','verify_native_ws71.py']","['design_groups8_ws72.py','setup_groups8_ws72.py','repair_fixture_groups8_ws72.py','prepare_native_ws72.py','verify_native_ws72.py']")
t=t.replace("'source-inverse-proof.json',","'source-inverse-proof.json','source-transformations.json','primary_decode_groups8_shader.h','primary_decode_groups8_test.cpp','pre-native-fixture-amendment.txt',")
t=t.replace("assert(w/'build-controller.log').read_text().splitlines()==['BUILD PASS '+v for v in roles]","assert(w/'build-controller.log').read_text().splitlines()==(w/'rebuild-controller.log').read_text().splitlines()==['BUILD PASS '+v for v in roles]")
new='''cases=[]
for inner in [128,256]:
 for columns in [260,272,288,1024]:
  for rows in [1,2,16,32]:
   cases.append((rows,inner,columns,0,0))
   if rows<=2:cases.append((rows,inner,columns,0,1))
cases += [(rows,128,columns,0,0)for columns in [100352,100368]for rows in [1,2]]
cases += [(rows,512,columns,1,prepare)for columns in [260,272]for rows in [1,2]for prepare in [0,1]]
assert len(cases)==60 and sum(row*col for row,inner,col,raw,prepare in cases)==804504
group_reference=None;group_summaries={}
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role);data=(build/'primary_decode_groups8_words.bin').read_bytes()
 if group_reference is None:group_reference=data
 assert data==group_reference;position=0
 for rows,inner,columns,raw,prepare in cases:
  count=rows*columns;assert struct.unpack('<7Q',data[position:position+56])==(rows,inner,columns,raw,prepare,prepare,count);position+=56
  reference=b''.join(struct.pack('<f',inner*(row+1)/16)*columns for row in range(rows));assert data[position:position+count*4]==reference;position+=count*4
 assert position==len(data)
 summary=load(build/'primary_decode_groups8_words.bin.summary.json');group_summaries[role]=summary
 assert summary['complete_records']==60 and summary['complete_FP32_words']==804504 and summary['generic_body_cases']==summary['specialized_body_cases']==20 and summary['prefill_cases']==16 and summary['head_cases']==4 and summary['groups8_enabled']==(role in ['on','fault'])
 assert [summary[k]for k in ['groups8_decode_delta','groups8_head_delta']]==([32,4]if role in ['on','fault']else[0,0])
(w/'groups8-native-word-comparison.json').write_text(json.dumps(dict(records=60,words_per_build=804504,all_six_complete_streams_equal=True,all_original_binary_fraction_references_exact=True,bytes=len(group_reference),sha256=hashlib.sha256(group_reference).hexdigest(),summaries=group_summaries),indent=2)+'\\n')
'''
needle=' programs={}';assert t.count(needle)==1;t=t.replace(needle,textwrap.indent(new,' ')+' programs={}')
before=" for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==full.encode()and programs[role]['prefill']==programs['compat56']['prefill']"
after=''' original_primary=(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]
 selected_primary=original_primary.replace('ulong column=ulong(group)*16+simd*4;','ulong column=ulong(group)*32+simd*4;');assert original_primary.count('ulong column=ulong(group)*16+simd*4;')==2
 selected_full=full.replace(original_primary,selected_primary);assert selected_full!=full
 for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==selected_full.encode()
 for role in roles:
  selected=(r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role)/'groups8_selected_primary_program.metal').read_bytes();assert selected==(selected_primary if role in ['on','fault']else original_primary).encode();(w/('actual-'+role+'-groups8-selected-primary.metal')).write_bytes(selected)'''
assert t.count(before)==1;t=t.replace(before,after).replace('ON_FAULT_prefill_exact_unchanged_preamble_primary_vector_fragments=True','ON_FAULT_prefill_exact_two_origin_selected_primary_plus_original_preamble_vector=True');ast.parse(t);(w/'native_controller.py').write_text(t)
v=(d/'verify_native_ws71.py').read_text().replace('worldsense-minimal-full-prefill','worldsense-primary-decode-groups8')
a=v.index('assert source==seed');z=v.index('assert all(sha(Path(k))',a)
v=v[:a]+'''assert len(source)==1723 and len(seed)==1721 and all(sha(s/k)==value for k,value in source.items())
published=json.loads(__import__('subprocess').check_output(['git','-C',str(r/'Garnet'),'show','4c811ec4e305317efac622177d4623b1cf8c83a4:results/20261018/worldsense-m5-minimal-full-prefill/correctness-source-manifest.json']))
assert published==seed
transforms=load(w/'source-transformations.json');parent=r/'out/research/worldsense-minimal-full-prefill/source'
for rel,operations in transforms.items():
 value=(s/rel).read_text()
 for op in reversed(operations):
  if op.get('append'):assert value.endswith(op['after']);value=value[:-len(op['after'])]
  else:assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
 assert value==(parent/rel).read_text()and hashlib.sha256(value.encode()).hexdigest()==seed[rel]
assert all(source[k]==value for k,value in seed.items()if k not in transforms)
assert len(load(w/'source-inverse-proof.json')['changed_or_added'])==6
'''+v[z:]
v=v.replace("r'\\d+/9\\s+Test", "r'\\d+/10\\s+Test").replace('list(range(1,10))','list(range(1,11))').replace('out of 9','out of 10')
v=v.replace("programs={role:{kind:",new+"programs={role:{kind:")
oldfull="for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==full.encode()and programs[role]['prefill']==programs['compat56']['prefill']"
newfull='\n'.join(line[1:]for line in after.splitlines());assert v.count(oldfull)==1;v=v.replace(oldfull,newfull)
v=v.replace('source_files=1721,native_tests_pass=54','source_files=1723,native_tests_pass=60').replace('actual_FULL_MAIN_exact_RAW_and_secondary_exact_original_fragments=True','actual_FULL_MAIN_exact_RAW_and_secondary_only_two_origin_substitutions=True,new60_complete804504words_generic20_specialized20_prefill16_heads4_exact=True,groups8_decode32_head4_ON_FAULT_and_zero_other_roles=True')
v=v.replace("(w/'groups8-native-word-comparison.json').write_text(json.dumps(","assert load(w/'groups8-native-word-comparison.json')==(").replace("summaries=group_summaries),indent=2)+'\\n')","summaries=group_summaries))")
v=v.replace(";(w/('actual-'+role+'-groups8-selected-primary.metal')).write_bytes(selected)",";assert(w/('actual-'+role+'-groups8-selected-primary.metal')).read_bytes()==selected")
ast.parse(v);p=d/'verify_native_ws72.py';assert not p.exists();p.write_text(v)
print('WS72 native60/original186+20/new60complete804504word proofs prepared; sourceinverse/generic-specialized/head/raw/tail/selectedprogram/counters allrequired. No native/model dispatched.')
