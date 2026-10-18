"""Post-terminal independent source/word/program/lifetime/flag input audit."""
from pathlib import Path
import datetime,hashlib,json,re,struct,sys
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-owned-storage-local';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-owned-storage-local/source'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
roles=['off','primary','on','fault','compat56','compat56-fault','control','raw-borrow']
source=load(w/'correctness-source-manifest.json');seed=load(w/'seed-source-manifest.json')
assert len(source)==1722 and len(seed)==1721 and all(sha(s/k)==v for k,v in source.items())
assert seed==load(w.parent/'worldsense-minimal-selective-defaults/correctness-source-manifest.json')
changes=load(w/'source-transformations.json');expected=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/owned_storage_local_test.cpp']
assert sorted(k for k in source if source[k]!=seed.get(k))==expected
for rel in {x['path']for x in changes}:
 text=(s/rel).read_text()
 for op in reversed([x for x in changes if x['path']==rel]):
  if op['append']:assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert hashlib.sha256(text.encode()).hexdigest()==seed[rel]
assert all(sha(Path(k))==v for k,v in load(w/'native-input-identities-before-dispatch.json')['files'].items())
assert load(w/'native-results.json')=={role:0 for role in roles}
build=lambda role:r/('out/build/garnet-metal-worldsense-owned-storage-local-'+role)
old_build=r/'out/build/garnet-metal-worldsense-minimal-selective-defaults-off'
raw=(old_build/'primary_projection_words.bin').read_bytes()
assert raw==(r/'out/build/garnet-metal-worldsense-request-weight-residency-off/primary_projection_words.bin').read_bytes()
tests=['metal_exact_half_codec','metal_primary_block_codec','metal_primary_storage','metal_primary_projection','metal_primary_core_isolation','metal_request_residency','metal_owned_storage_local_lifetime']
actual_names=None
core_cases=[(rows,inner,columns,0)for inner in [128,256]for columns in [260,1024]for rows in [1,2,16,32]]+[(rows,128,100352,0)for rows in [1,2]]+[(rows,128,260,1)for rows in [1,2]]
complete_lifetime=None;role_details={}
for role in roles:
 text=(w/('native-'+role+'.log')).read_text()
 passed=re.findall(r'\d+/10\s+Test\s+#\s*(\d+):\s*(\S+)[^\n]*Passed',text)
 assert sorted(int(x[0])for x in passed)==list(range(1,11)) and re.search(r'100% tests passed(?:, 0 tests failed)? out of 10',text)
 names={x[1]for x in passed}
 if actual_names is None:actual_names=names
 assert names==actual_names and 'metal_owned_storage_local_lifetime' in names
 assert (build(role)/'primary_projection_words.bin').read_bytes()==raw
 data=(build(role)/'primary_core_projection_words.bin').read_bytes();offset=0;words=0
 for rows,inner,columns,raw_mode in core_cases:
  count=rows*columns;assert struct.unpack_from('<6Q',data,offset)==(rows,inner,columns,raw_mode,int(rows<=2 and columns<100000),count);offset+=48
  expected_words=b''.join(struct.pack('<f',inner*(row+1)/16)*columns for row in range(rows))
  assert data[offset:offset+4*count]==expected_words;offset+=4*count;words+=count
 assert offset==len(data) and words==432804
 summary=load(build(role)/'primary_core_projection_words.bin.summary.json')
 assert summary['complete_records']==20 and summary['complete_FP32_words']==432804 and summary['specialized_body_cases']==10 and summary['core_enabled'] is False
 assert all(summary[k]==0 for k in ['core_decode_delta','core_prefill_delta','core_head_delta'])
 # A separate oracle recomputes EVERY lifetime output byte, header and EOF.
 data=(build(role)/'owned_storage_projection_words.bin').read_bytes();offset=0;words=0;records=0
 for raw_mode in [False,True]:
  for mode in range(5):
   shapes=[(rows,columns)for rows in [1,2,16,32,64]for columns in [260,1024]]+[(rows,100352)for rows in [1,2]]
   for rows,columns in shapes:
    count=rows*columns;assert struct.unpack_from('<6Q',data,offset)==(rows,128,columns,int(raw_mode),mode,count);offset+=48
    expected_words=b''.join(struct.pack('<f',8*(row+1))*columns for row in range(rows))
    assert data[offset:offset+4*count]==expected_words;offset+=4*count;words+=count;records+=1
 assert offset==len(data) and records==120 and words==4487160
 if complete_lifetime is None:complete_lifetime=data
 assert complete_lifetime==data
 v=load(build(role)/'owned_storage_projection_words.bin.summary.json');enabled=role in ['primary','on','fault','raw-borrow'];primary=role not in ['off','raw-borrow']
 assert v['owned_storage_local_enabled']==enabled and v['complete_records']==120 and v['complete_FP32_words']==words and v['raw_cases']==(60 if primary else 120) and v['packed_cases']==(60 if primary else 0) and v['invalid_cases_rejected']==72 and v['prepared_cases']==40
 assert all(v[k]is True for k in ['live_logical_accounting_restored','original_alias_release_before_prepare','last_owner_release_before_commit','last_owner_release_after_flush','creator_context_closed_before_use','flushed_abort_owner_release_and_recovery','no_retainCount_or_timing_claim'])
 cache=(build(role)/'CMakeCache.txt').read_text();flags=dict(re.findall(r'^GARNET_METAL_([A-Z0-9_]+):BOOL=(ON|OFF)$',cache,re.M));profile={'control':'on','raw-borrow':'off'}.get(role,role);compat=profile.startswith('compat56')
 assert flags['OWNED_STORAGE_LOCAL']==('ON'if enabled else'OFF')
 for name in ['PRIMARY_BF16_BLOCKS','WIDE_PRIMARY_STORAGE','LAZY_RESIDENT_DEFAULTS']:assert flags[name]==('OFF'if profile=='off'else'ON')
 for name in ['PRIMARY_VECTOR_PREFILL','PRIMARY_PREFILL_ISOLATED_LIBRARY','PRIMARY_FULL_ISOLATED_LIBRARY']:assert flags[name]==('ON'if compat or profile in ['on','fault']else'OFF')
 for name in ['DIRECT_FLOAT_WEIGHT_UPLOAD','CACHED_MATRIX_METADATA','REQUEST_SCOPE_WEIGHT_RESIDENCY','PRIMARY_VECTOR_PREFILL_K32_ONLY']:assert flags[name]==('ON'if compat else'OFF')
 assert flags['PRIMARY_ONLY_LAZY_DEFAULTS']==('ON'if profile in ['primary','on','fault']else'OFF') and flags['PRIMARY_CORE_ISOLATED_LIBRARY']=='OFF'
 assert flags['PRIMARY_TEST_FAULTS']==('ON'if role in ['fault','compat56-fault']else'OFF') and flags['REQUEST_RESIDENCY_TEST_FAULTS']==('ON'if role=='compat56-fault'else'OFF')
 role_details[role]=dict(enabled=enabled,records=120,complete_words=words,raw_cases=v['raw_cases'],packed_cases=v['packed_cases'],complete_lifetime_sha256=hashlib.sha256(data).hexdigest())
programs={role:{kind:(w/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()for kind in ['main','prefill','core']}for role in roles}
for role in roles:
 profile={'control':'on','raw-borrow':'off'}.get(role,role)
 for kind in ['main','prefill','core']:assert programs[role][kind]==(w.parent/'worldsense-minimal-selective-defaults'/('actual-'+profile+'-'+kind+'-program.metal')).read_bytes()
 assert not programs[role]['core']
assert load(w/'restoration-verification.json')['benchmark_valid_request_http404']
assert load(w/'host-codegen-review.json')['selected_local_ownership_removed_both_raw_and_packed']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_native_session=int(sys.argv[1]),observed_exit_code=0,source_files=1722,exact_four_path_source_inverse=True,all_other1718files_unchanged=True,native_tests_pass=80,CTest_names=sorted(actual_names),original186_complete_projection_streams_exact_all_eight=True,original20records432804words_and_zero_CORE_all_eight=True,new120records4487160words_per_role_all_bytes_independently_exact=True,all_actual_programs_exact_seed74=True,source_and_all_explicit_native_inputs_binaries_unchanged=True,roles=role_details,no_model_or_speed_result_yet=True,promotion=False)
(w/'native-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='roles'},indent=2))
