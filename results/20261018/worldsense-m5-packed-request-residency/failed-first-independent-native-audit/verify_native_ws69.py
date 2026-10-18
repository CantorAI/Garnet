"""Post-terminal source/native/program/input audit, no model or timing execution."""
from pathlib import Path
import json,hashlib,re,struct,datetime,sys
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-packed-request-residency';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-packed-request-residency/source';old=r/'out/research/worldsense-request-weight-residency/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();roles=['off','primary','on','fault','compat56','compat56-fault'];source=load(w/'correctness-source-manifest.json');seed=load(w/'seed-source-manifest.json')
assert len(source)==1721 and len(seed)==1720 and all(sha(s/k)==v for k,v in source.items())and all(sha(old/k)==v for k,v in seed.items())
transforms=load(w/'source-transformations.json')
for rel,ops in transforms.items():
 text=(s/rel).read_text()
 for op in reversed(ops):
  if op.get('append'):assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text==(old/rel).read_text(),rel
assert all(source[k]==v for k,v in seed.items()if k not in transforms)
context=load(w/'packed-context-adaptation-proof.json');assert context['original']in(old/'src/backends/metal/metal_context.mm').read_text()and context['original']in(s/'src/backends/metal/metal_context.mm').read_text()and context['new']in(s/'src/backends/metal/metal_context.mm').read_text()
text=context['new']
for op in reversed(context['operations']):assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
assert text==context['renamed_original']and '[impl_->packedRequestResidencySet requestResidency]'in context['new']and ' packedRequestResidency]'not in context['new']
assert 'constexpr uint64_t kPackedRequestResidencyLimit=128*1024;'in context['new']and 'constexpr uint64_t kPackedRequestResidencyLimit=uint64_t(8)*1024*1024*1024;'in context['new']
assert 'owner->primaryPayload,owner->primaryMetadata'in context['new']and 'seen.size()>2048'in context['new']and 'packedRequestResidencySet.allocationCount'in context['new']
session=load(w/'packed-session-adaptation-proof.json');assert session['original']in(old/'src/backends/metal/decoder_session.cpp').read_text()and session['original']in(s/'src/backends/metal/decoder_session.cpp').read_text()and session['new']in(s/'src/backends/metal/decoder_session.cpp').read_text()
assert all(sha(Path(k))==v for k,v in load(w/'native-input-identities-before-dispatch.json')['files'].items())
assert load(w/'native-results.json')=={role:0 for role in roles}
for role in roles:
 log=(w/('native-'+role+'.log')).read_text();assert sorted(int(v)for v in re.findall(r'\d+/9\s+Test\s+#\s*(\d+):[^\n]*Passed',log))==list(range(1,10))and '100% tests passed, 0 tests failed out of 9'in log
raw=(r/'out/build/garnet-metal-worldsense-packed-request-residency-off/primary_projection_words.bin').read_bytes();assert raw==(r/'out/build/garnet-metal-worldsense-request-weight-residency-off/primary_projection_words.bin').read_bytes()
for role in roles:assert(r/('out/build/garnet-metal-worldsense-packed-request-residency-'+role)/'primary_projection_words.bin').read_bytes()==raw
proof=load(w/'native-word-comparison.json');assert proof['records']==186 and proof['bytes']==len(raw)and proof['sha256']==hashlib.sha256(raw).hexdigest()
reference=None
for role in roles:
 p=r/('out/build/garnet-metal-worldsense-packed-request-residency-'+role)/'packed_request_projection_words.bin';data=p.read_bytes()
 if reference is None:reference=data
 assert data==reference;pos=0
 for pattern,value in [(0,128.),(1,-1.25)]:
  assert struct.unpack('4Q',data[pos:pos+32])==(260,128,pattern,260);pos+=32;assert data[pos:pos+1040]==struct.pack('<f',value)*260;pos+=1040
 assert pos==len(data)
 extra=Path(str(p)+'.leased.bin').read_bytes();assert extra==(data if role in ['on','fault']else b'')
programs={role:{kind:(w/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()for kind in ['main','prefill']}for role in roles}
for role in ['primary','on','fault']:assert programs[role]==programs['primary']
for role,prior in [('off','off'),('primary','primary'),('compat56','on'),('compat56-fault','fault')]:
 for kind in ['main','prefill']:assert programs[role][kind]==(w.parent/'worldsense-request-weight-residency'/('actual-'+prior+'-'+kind+'-program.metal')).read_bytes()
assert load(w/'restoration-verification.json')['benchmark_valid_request_http404']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_native_session=int(sys.argv[1]),observed_exit_code=0,source_files=1721,native_tests_pass=54,all6roles9tests_pass=True,all_original186_complete_projection_records_and_FP64_bounds_preserved=True,new_two_complete_reference_records_520words_eachbuild_exact=True,both_ON_FAULT_complete_leased_streams_match_reference=True,source_and_all_explicit_native_inputs_binaries_unchanged=True,old_raw_policy_body_budget_API_and_tests_preserved=True,original_all_shader_programs_exact=True,ON_FAULT_programs_equal_PRIMARY=True,actual_SDK_selector_correct=True,defaultOFF_and_original_raw_fault_compatibility_pass=True,no_model_or_speed_result_yet=True,promotion=False)
(w/'native-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
