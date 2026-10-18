"""Post-terminal complete native/source/input audit. No native rerun or inference."""
from pathlib import Path
import json,hashlib,datetime,re,struct
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-primary-simd-metadata';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-primary-simd-metadata/source';seed=r/'out/research/worldsense-cooperative-prefill/source';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();roles=['off','primary','on','fault','compat67']
manifest=load(w/'correctness-source-manifest.json');sm=load(w/'seed-source-manifest.json');assert len(manifest)==1726 and len(sm)==1724 and all(sha(s/k)==v for k,v in manifest.items())and all(sha(seed/k)==v for k,v in sm.items())
for rel,operations in load(w/'source-transformations.json').items():
 text=(s/rel).read_text()
 for op in reversed(operations):
  if op.get('append'):assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text==(seed/rel).read_text()
assert all(manifest[k]==v for k,v in sm.items()if k not in load(w/'source-transformations.json'))
proof=load(w/'simd-shader-inverse-proof.json');shader=(s/'src/backends/metal/primary_simd_metadata_shader.h').read_text().split('R"PSIMD(',1)[1].split(')PSIMD"',1)[0];assert shader==proof['new_source'];reader=proof['reader_append'];assert shader.endswith(reader);inverse=shader[:-len(reader)]
for op in reversed(proof['operations']):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
assert inverse==proof['original_source']==(s/'src/backends/metal/primary_block_shader.h').read_text().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]
word=proof['word_helper_supplied_metadata'].replace('primary_word_from_meta(device const uchar* payload,uint meta,uint row,uint k)','primary_word(device const uchar* payload,device const uint* metadata,uint row,uint k,uint inner)').replace('uint at=','uint block=(row/4)*(inner/128)+k/128,meta=metadata[block],at=');assert word==proof['word_helper_original']
quad=proof['quad_helper_supplied_metadata'].replace('primary_quad_from_meta(device const uchar* payload,uint meta,uint k)','primary_quad(device const uchar* payload,device const uint* metadata,uint row,uint k,uint inner)').replace('uint at=','uint meta=metadata[(row/4)*(inner/128)+k/128],at=');assert quad==proof['quad_helper_original']
identity=load(w/'native-driver-identities-before-dispatch.json')['files'];mismatch=[k for k,v in identity.items()if sha(Path(k))!=v];output=str(w/'native-results.json');assert mismatch==[output]
assert sha(w/'failed-first-native/native-results.json')==identity[output]
assert load(w/'native-results.json')=={role:0 for role in roles}
assert all(sha(Path(k))==v for k,v in load(w/'native-binaries-before.json').items())
for role in roles:
 text=(w/('native-'+role+'.log')).read_text();assert sorted(int(n)for n in re.findall(r'\d+/10 Test #\s*(\d+):.*Passed',text))==list(range(1,11))
for filename,records in [('primary_projection_words.bin',186),('sole_projection_words.bin',28),('sole_dual_projection_words.bin',36)]:
 reference=(r/'out/build/garnet-metal-worldsense-cooperative-prefill-off'/filename).read_bytes()
 for role in roles:assert(r/('out/build/garnet-metal-worldsense-primary-simd-metadata-'+role)/filename).read_bytes()==reference
 p=load(w/(filename+'.comparison.json'));assert p['records']==records and p['bytes']==len(reference)and p['sha256']==hashlib.sha256(reference).hexdigest()
paths={role:r/('out/build/garnet-metal-worldsense-primary-simd-metadata-'+role)/'primary_simd_metadata_words.bin'for role in roles};data=paths['on'].read_bytes();assert data==paths['fault'].read_bytes()and all(not paths[role].read_bytes()for role in ['off','primary','compat67']);pos=0;count=0;allwords=0
for inner in [128,256,512]:
 for pattern in range(3):
  n=260*inner
  for prefill in [0,1]:
   for start,length in [(0,n),(1,1),(127,3),(inner-1,7),(3*inner+125,517),(n-1,1)]:
    shape=struct.unpack('6Q',data[pos:pos+48]);pos+=48;assert shape==(260,inner,pattern,prefill,start,length);raw=data[pos:pos+4*length];pos+=4*length;values=struct.unpack(str(length)+'I',raw)
    for j,actual in enumerate(values):
     i=start+j;row,k=divmod(i,inner);block=row//4*(inner//128)+k//128;expected=(i&65535)if pattern==0 else 0x3f00+(i*13)%128
     if pattern==2 and block%3==1:expected=i*19&65535
     if(expected&0x7f80)==0x7f80:expected=0x8000
     assert actual==expected<<16
    count+=1;allwords+=length
assert pos==len(data)and count==108
metadata=load(w/'metadata-word-comparison.json');assert metadata['record_count']==count and metadata['bytes']==len(data)and metadata['sha256']==hashlib.sha256(data).hexdigest()
programs={role:{kind:(w/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()for kind in ['main','prefill']}for role in roles};original=proof['original_source'].encode();new=shader.encode();assert programs['primary']['main'].count(original)==1 and programs['on']['main']==programs['fault']['main']==programs['primary']['main'].replace(original,new)
for role,prior in [('off','off'),('primary','primary'),('compat67','on')]:
 for kind in ['main','prefill']:assert programs[role][kind]==(w.parent/'worldsense-cooperative-prefill'/('actual-'+prior+'-'+kind+'-program.metal')).read_bytes()
assert all(not programs[role]['prefill']for role in ['off','primary','on','fault'])
assert load(w/'restoration-verification.json')['benchmark_valid_request_http404']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_session=3206,observed_controller_exit_code=1,original_CTest_five_roles_all_exit0=True,native_tests_pass=50,original_projection_records=186,MLP_records=28,paired_records=36,original_complete_streams_match_all_five_roles_and_seed=True,new_metadata_records=108,new_diagnostic_FP32_words=allwords,new_metadata_stream_bytes=len(data),both_helpers_complete_original_finite_bits_and_ranges_exact=True,actual_constructor_programs_and_explicit_shader_inverse_pass=True,source_binaries_runtime_and_all_frozen_INPUTS_unchanged=True,sole_frozen_mismatch_is_controller_OUTPUT_native_results=True,original_failed_partial_result_SHA_verified=True,post_terminal_output_identity_classification_repair_no_rerun=True,no_model_or_speed_result=True,promotion=False,scope='Synthetic native and source proof only. Full learned201matrices/both new helpers, full original states/tokens and real WorldSense gates remain. Controller exit remains1; no fabricated successful controller termination.')
(w/'native-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
