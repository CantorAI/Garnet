"""Independent post-terminal source/word/program/counter audit, no model/timing."""
from pathlib import Path
import json,hashlib,re,struct,sys,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-minimal-full-prefill';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-minimal-full-prefill/source';old=r/'out/research/worldsense-request-weight-residency/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();roles=['off','primary','on','fault','compat56','compat56-fault'];source=load(w/'correctness-source-manifest.json');seed=load(w/'seed-source-manifest.json')
assert source==seed and len(source)==1721 and all(sha(s/k)==value for k,value in source.items())
published=json.loads(__import__('subprocess').check_output(['git','-C',str(r/'Garnet'),'show','80787c104fdafe7eb4671cc6c7b390bfcce2fb1f:results/20261018/worldsense-m5-primary-core-isolation/correctness-source-manifest.json']))
assert published==source
assert not load(w/'source-inverse-proof.json')['changed_or_added']
assert all(sha(Path(k))==v for k,v in load(w/'native-input-identities-before-dispatch.json')['files'].items())
assert load(w/'native-results.json')=={role:0 for role in roles}
for role in roles:
 text=(w/('native-'+role+'.log')).read_text();assert sorted(int(v)for v in re.findall(r'\d+/9\s+Test\s+#\s*(\d+):[^\n]*Passed',text))==list(range(1,10))and re.search(r'100% tests passed(?:, 0 tests failed)? out of 9',text)
raw=(r/'out/build/garnet-metal-worldsense-minimal-full-prefill-off/primary_projection_words.bin').read_bytes();assert raw==(r/'out/build/garnet-metal-worldsense-request-weight-residency-off/primary_projection_words.bin').read_bytes()
assert all((r/('out/build/garnet-metal-worldsense-minimal-full-prefill-'+role)/'primary_projection_words.bin').read_bytes()==raw for role in roles)
proof=load(w/'native-word-comparison.json');assert proof['records']==186 and proof['bytes']==len(raw)and proof['sha256']==hashlib.sha256(raw).hexdigest()
reference=None;cases=[(rows,inner,columns,0)for inner in [128,256]for columns in [260,1024]for rows in [1,2,16,32]]+[(rows,128,100352,0)for rows in [1,2]]+[(rows,128,260,1)for rows in [1,2]]
for role in roles:
 build=r/('out/build/garnet-metal-worldsense-minimal-full-prefill-'+role);data=(build/'primary_core_projection_words.bin').read_bytes()
 if reference is None:reference=data
 assert data==reference;position=0;words=0
 for rows,inner,columns,raw_mode in cases:
  count=rows*columns;assert struct.unpack('<6Q',data[position:position+48])==(rows,inner,columns,raw_mode,int(rows<=2 and columns<100000),count);position+=48
  expected=b''.join(struct.pack('<f',inner*(row+1)/16)*columns for row in range(rows));assert data[position:position+count*4]==expected;position+=count*4;words+=count
 assert position==len(data)and words==432804
 summary=load(build/'primary_core_projection_words.bin.summary.json');assert summary['complete_records']==20 and summary['complete_FP32_words']==432804 and summary['specialized_body_cases']==10 and summary['core_enabled'] is False
 assert [summary[k]for k in ['core_decode_delta','core_prefill_delta','core_head_delta']]==[0,0,0]
new=load(w/'core-native-word-comparison.json');assert new['records']==20 and new['words_per_build']==432804 and new['bytes']==len(reference)and new['sha256']==hashlib.sha256(reference).hexdigest()
programs={role:{kind:(w/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()for kind in ['main','prefill','core']}for role in roles}
for role,previous in [('off','off'),('primary','primary'),('compat56','on'),('compat56-fault','fault')]:
 for kind in ['main','prefill']:assert programs[role][kind]==(w.parent/'worldsense-request-weight-residency'/('actual-'+previous+'-'+kind+'-program.metal')).read_bytes()
full=(s/'src/backends/metal/primary_isolation_shader.h').read_text().split('R"IPRIMARY(',1)[1].split(')IPRIMARY"',1)[0]+(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]+(s/'src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R"VPREFILL(',1)[1].split(')VPREFILL"',1)[0]
for role in ['on','fault']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==full.encode()and programs[role]['prefill']==programs['compat56']['prefill']
assert all(not programs[role]['core']for role in roles)
for role,group in load(w/'actual-program-proof.json')['programs'].items():
 for kind,info in group.items():assert len(programs[role][kind])==info['bytes']and hashlib.sha256(programs[role][kind]).hexdigest()==info['sha256']
assert load(w/'restoration-verification.json')['benchmark_valid_request_http404']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_native_session=int(sys.argv[1]),observed_exit_code=0,source_files=1721,native_tests_pass=54,original186_complete_projection_records_and_FP64_bounds_preserved=True,new20_complete_records432804words_exact_all_six=True,all_actual_core_route_counters_and_getter_zero=True,actual_FULL_MAIN_exact_RAW_and_secondary_exact_original_fragments=True,original_two_program_vector_prefill_contract_and_all_compat56_programs_preserved=True,source_and_all_explicit_native_inputs_binaries_unchanged=True,no_model_or_speed_result_yet=True,promotion=False)
(w/'native-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
