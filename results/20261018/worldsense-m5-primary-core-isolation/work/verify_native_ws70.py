"""Independent post-terminal source/word/program/counter audit, no model/timing."""
from pathlib import Path
import json,hashlib,re,struct,sys,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-primary-core-isolation';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-core-isolation/source';old=r/'out/research/worldsense-request-weight-residency/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();roles=['off','primary','on','fault','compat56','compat56-fault'];source=load(w/'correctness-source-manifest.json');seed=load(w/'seed-source-manifest.json')
assert len(source)==1721 and len(seed)==1720 and all(sha(s/k)==v for k,v in source.items())and all(sha(old/k)==v for k,v in seed.items())
transforms=load(w/'source-transformations.json')
for rel,ops in transforms.items():
 text=(s/rel).read_text()
 for op in reversed(ops):
  if op.get('append'):assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text==(old/rel).read_text(),rel
assert all(source[k]==v for k,v in seed.items()if k not in transforms)
assert source['test2026/metal/primary_core_isolation_test.cpp']==sha(w/'primary_core_isolation_test.cpp')
assert all(sha(Path(k))==v for k,v in load(w/'native-input-identities-before-dispatch.json')['files'].items())
assert load(w/'native-results.json')=={role:0 for role in roles}
for role in roles:
 text=(w/('native-'+role+'.log')).read_text();assert sorted(int(v)for v in re.findall(r'\d+/9\s+Test\s+#\s*(\d+):[^\n]*Passed',text))==list(range(1,10))and re.search(r'100% tests passed(?:, 0 tests failed)? out of 9',text)
raw=(r/'out/build/garnet-metal-worldsense-primary-core-isolation-off/primary_projection_words.bin').read_bytes();assert raw==(r/'out/build/garnet-metal-worldsense-request-weight-residency-off/primary_projection_words.bin').read_bytes()
assert all((r/('out/build/garnet-metal-worldsense-primary-core-isolation-'+role)/'primary_projection_words.bin').read_bytes()==raw for role in roles)
proof=load(w/'native-word-comparison.json');assert proof['records']==186 and proof['bytes']==len(raw)and proof['sha256']==hashlib.sha256(raw).hexdigest()
reference=None;cases=[(rows,inner,columns,0)for inner in [128,256]for columns in [260,1024]for rows in [1,2,16,32]]+[(rows,128,100352,0)for rows in [1,2]]+[(rows,128,260,1)for rows in [1,2]]
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-primary-core-isolation-'+role);data=(build/'primary_core_projection_words.bin').read_bytes()
 if reference is None:reference=data
 assert data==reference;position=0;words=0
 for rows,inner,columns,raw_mode in cases:
  count=rows*columns;assert struct.unpack('<6Q',data[position:position+48])==(rows,inner,columns,raw_mode,int(rows<=2 and columns<100000),count);position+=48
  expected=b''.join(struct.pack('<f',inner*(row+1)/16)*columns for row in range(rows));assert data[position:position+count*4]==expected;position+=count*4;words+=count
 assert position==len(data)and words==432804
 summary=load(build/'primary_core_projection_words.bin.summary.json');assert summary['complete_records']==20 and summary['complete_FP32_words']==432804 and summary['specialized_body_cases']==10 and summary['core_enabled']==(role in ['on','fault'])
 assert [summary[k]for k in ['core_decode_delta','core_prefill_delta','core_head_delta']]==([8,8,2]if role in ['on','fault']else[0,0,0])
new=load(w/'core-native-word-comparison.json');assert new['records']==20 and new['words_per_build']==432804 and new['bytes']==len(reference)and new['sha256']==hashlib.sha256(reference).hexdigest()
programs={role:{kind:(w/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()for kind in ['main','prefill','core']}for role in roles}
for role,previous in [('off','off'),('primary','primary'),('compat56','on'),('compat56-fault','fault')]:
 for kind in ['main','prefill']:assert programs[role][kind]==(w.parent/'worldsense-request-weight-residency'/('actual-'+previous+'-'+kind+'-program.metal')).read_bytes()
core=(s/'src/backends/metal/primary_isolation_shader.h').read_text().split('R"IPRIMARY(',1)[1].split(')IPRIMARY"',1)[0]+(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]
for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and not programs[role]['prefill']and programs[role]['core']==core.encode()
assert all(not programs[role]['core']for role in ['off','primary','compat56','compat56-fault'])
for role,group in load(w/'actual-program-proof.json')['programs'].items():
 for kind,info in group.items():assert len(programs[role][kind])==info['bytes']and hashlib.sha256(programs[role][kind]).hexdigest()==info['sha256']
assert load(w/'restoration-verification.json')['benchmark_valid_request_http404']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_native_session=int(sys.argv[1]),observed_exit_code=0,source_files=1721,native_tests_pass=54,original186_complete_projection_records_and_FP64_bounds_preserved=True,new20_complete_records432804words_exact_all_six=True,actual_core_route_counts_decode8_prefill8_head2_and_raw_zero=True,actual_CORE_MAIN_exact_RAW_and_CORE_exact_original_fragment_preamble=True,original_two_program_vector_prefill_contract_and_all_compat56_programs_preserved=True,source_and_all_explicit_native_inputs_binaries_unchanged=True,no_model_or_speed_result_yet=True,promotion=False)
(w/'native-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
