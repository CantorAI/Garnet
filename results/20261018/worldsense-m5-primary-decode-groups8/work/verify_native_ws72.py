"""Independent post-terminal source/word/program/counter audit, no model/timing."""
from pathlib import Path
import json,hashlib,re,struct,sys,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-primary-decode-groups8';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-decode-groups8/source';old=r/'out/research/worldsense-request-weight-residency/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();roles=['off','primary','on','fault','compat56','compat56-fault'];source=load(w/'correctness-source-manifest.json');seed=load(w/'seed-source-manifest.json')
assert len(source)==1723 and len(seed)==1721 and all(sha(s/k)==value for k,value in source.items())
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
assert all(sha(Path(k))==v for k,v in load(w/'native-input-identities-before-dispatch.json')['files'].items())
assert load(w/'native-results.json')=={role:0 for role in roles}
for role in roles:
 text=(w/('native-'+role+'.log')).read_text();assert sorted(int(v)for v in re.findall(r'\d+/10\s+Test\s+#\s*(\d+):[^\n]*Passed',text))==list(range(1,11))and re.search(r'100% tests passed(?:, 0 tests failed)? out of 10',text)
raw=(r/'out/build/garnet-metal-worldsense-primary-decode-groups8-off/primary_projection_words.bin').read_bytes();assert raw==(r/'out/build/garnet-metal-worldsense-request-weight-residency-off/primary_projection_words.bin').read_bytes()
assert all((r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role)/'primary_projection_words.bin').read_bytes()==raw for role in roles)
proof=load(w/'native-word-comparison.json');assert proof['records']==186 and proof['bytes']==len(raw)and proof['sha256']==hashlib.sha256(raw).hexdigest()
reference=None;cases=[(rows,inner,columns,0)for inner in [128,256]for columns in [260,1024]for rows in [1,2,16,32]]+[(rows,128,100352,0)for rows in [1,2]]+[(rows,128,260,1)for rows in [1,2]]
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role);data=(build/'primary_core_projection_words.bin').read_bytes()
 if reference is None:reference=data
 assert data==reference;position=0;words=0
 for rows,inner,columns,raw_mode in cases:
  count=rows*columns;assert struct.unpack('<6Q',data[position:position+48])==(rows,inner,columns,raw_mode,int(rows<=2 and columns<100000),count);position+=48
  expected=b''.join(struct.pack('<f',inner*(row+1)/16)*columns for row in range(rows));assert data[position:position+count*4]==expected;position+=count*4;words+=count
 assert position==len(data)and words==432804
 summary=load(build/'primary_core_projection_words.bin.summary.json');assert summary['complete_records']==20 and summary['complete_FP32_words']==432804 and summary['specialized_body_cases']==10 and summary['core_enabled'] is False
 assert [summary[k]for k in ['core_decode_delta','core_prefill_delta','core_head_delta']]==[0,0,0]
new=load(w/'core-native-word-comparison.json');assert new['records']==20 and new['words_per_build']==432804 and new['bytes']==len(reference)and new['sha256']==hashlib.sha256(reference).hexdigest()
cases=[]
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
assert load(w/'groups8-native-word-comparison.json')==(dict(records=60,words_per_build=804504,all_six_complete_streams_equal=True,all_original_binary_fraction_references_exact=True,bytes=len(group_reference),sha256=hashlib.sha256(group_reference).hexdigest(),summaries=group_summaries))
programs={role:{kind:(w/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()for kind in ['main','prefill','core']}for role in roles}
for role,previous in [('off','off'),('primary','primary'),('compat56','on'),('compat56-fault','fault')]:
 for kind in ['main','prefill']:assert programs[role][kind]==(w.parent/'worldsense-request-weight-residency'/('actual-'+previous+'-'+kind+'-program.metal')).read_bytes()
full=(s/'src/backends/metal/primary_isolation_shader.h').read_text().split('R"IPRIMARY(',1)[1].split(')IPRIMARY"',1)[0]+(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]+(s/'src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R"PVEC(',1)[1].split(')PVEC"',1)[0]
original_primary=(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]
selected_primary=original_primary.replace('ulong column=ulong(group)*16+simd*4;','ulong column=ulong(group)*32+simd*4;');assert original_primary.count('ulong column=ulong(group)*16+simd*4;')==2
selected_full=full.replace(original_primary,selected_primary);assert selected_full!=full
for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==selected_full.encode()
for role in roles:
 selected=(r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role)/'groups8_selected_primary_program.metal').read_bytes();assert selected==(selected_primary if role in ['on','fault']else original_primary).encode();assert(w/('actual-'+role+'-groups8-selected-primary.metal')).read_bytes()==selected
assert all(not programs[role]['core']for role in roles)
for role,group in load(w/'actual-program-proof.json')['programs'].items():
 for kind,info in group.items():assert len(programs[role][kind])==info['bytes']and hashlib.sha256(programs[role][kind]).hexdigest()==info['sha256']
assert load(w/'restoration-verification.json')['benchmark_valid_request_http404']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_native_session=int(sys.argv[1]),observed_exit_code=0,source_files=1723,native_tests_pass=60,original186_complete_projection_records_and_FP64_bounds_preserved=True,new20_complete_records432804words_exact_all_six=True,all_actual_core_route_counters_and_getter_zero=True,actual_FULL_MAIN_exact_RAW_and_secondary_only_two_origin_substitutions=True,new60_complete804504words_generic20_specialized20_prefill16_heads4_exact=True,groups8_decode32_head4_ON_FAULT_and_zero_other_roles=True,original_two_program_vector_prefill_contract_and_all_compat56_programs_preserved=True,source_and_all_explicit_native_inputs_binaries_unchanged=True,no_model_or_speed_result_yet=True,promotion=False)
(w/'native-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
