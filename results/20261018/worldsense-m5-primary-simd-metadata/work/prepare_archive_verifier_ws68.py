from pathlib import Path
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');old=(b/'work/worldsense-opt/verify_archive_ws67.py').read_text();t=old[:old.index('assert all(v==0')]
t=t.replace('worldsense-m5-cooperative-prefill','worldsense-m5-primary-simd-metadata').replace('len(current)==1724','len(current)==1726').replace("ref='a13a77a5880103fed1c2ef44b5354ba22a2acb63';prefix='results/20261018/worldsense-m5-sole-dual-decode/'","ref='a26bf3059642478dcecff9b0687a0c9ac25959d3';prefix='results/20261018/worldsense-m5-cooperative-prefill/'").replace('len(sm)==1723','len(sm)==1724')
t=t.replace("'src/backends/metal/cooperative_sole_prefill_shader.h'","'src/backends/metal/primary_simd_metadata_shader.h'").replace("'test2026/metal/sole_interleaved_test.cpp'","'test2026/metal/primary_simd_metadata_test.cpp'")
t+='''roles=['off','primary','on','fault','compat67']
assert load(d/'native-results.json')=={role:0 for role in roles}
for role in roles:
 text=(d/('native-'+role+'.log')).read_text();assert sorted(int(i)for i in __import__('re').findall(r'\\d+/10\\s+Test\\s+#\\s*(\\d+):[^\\n]*Passed',text))==list(range(1,11))
for filename,count in [('primary_projection_words.bin',186),('sole_projection_words.bin',28),('sole_dual_projection_words.bin',36)]:
 words=(d/('native-off-'+filename)).read_bytes();assert all((d/('native-'+role+'-'+filename)).read_bytes()==words for role in roles)
 original=batch([ref+':'+prefix+'native-off-'+filename])[0];assert original==words
 proof=load(d/(filename+'.comparison.json'));assert proof['records']==count and proof['bytes']==len(words)and proof['sha256']==hashlib.sha256(words).hexdigest()
proof=load(d/'simd-shader-inverse-proof.json');original=current['src/backends/metal/primary_block_shader.h'].decode().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0];shader=current['src/backends/metal/primary_simd_metadata_shader.h'].decode().split('R"PSIMD(',1)[1].split(')PSIMD"',1)[0];assert original==proof['original_source']and shader==proof['new_source'];assert shader.endswith(proof['reader_append']);inverse=shader[:-len(proof['reader_append'])]
for op in reversed(proof['operations']):assert inverse.count(op['after'])==op['count'];inverse=inverse.replace(op['after'],op['before'])
assert inverse==original
word=proof['word_helper_supplied_metadata'].replace('primary_word_from_meta(device const uchar* payload,uint meta,uint row,uint k)','primary_word(device const uchar* payload,device const uint* metadata,uint row,uint k,uint inner)').replace('uint at=','uint block=(row/4)*(inner/128)+k/128,meta=metadata[block],at=');assert word==proof['word_helper_original']
quad=proof['quad_helper_supplied_metadata'].replace('primary_quad_from_meta(device const uchar* payload,uint meta,uint k)','primary_quad(device const uchar* payload,device const uint* metadata,uint row,uint k,uint inner)').replace('uint at=','uint meta=metadata[(row/4)*(inner/128)+k/128],at=');assert quad==proof['quad_helper_original']
programs={role:{kind:(d/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()for kind in ['main','prefill']}for role in roles}
assert programs['primary']['main'].count(original.encode())==1 and programs['on']['main']==programs['fault']['main']==programs['primary']['main'].replace(original.encode(),shader.encode())
for role,prior in [('off','off'),('primary','primary'),('compat67','on')]:
 for kind in ['main','prefill']:assert programs[role][kind]==batch([ref+':'+prefix+'actual-'+prior+'-'+kind+'-program.metal'])[0]
assert all(not programs[role]['prefill']for role in ['off','primary','on','fault'])
for role,row in load(d/'actual-program-proof.json')['programs'].items():
 for kind,info in row.items():assert len(programs[role][kind])==info['bytes']and hashlib.sha256(programs[role][kind]).hexdigest()==info['sha256']
sole=current['src/backends/metal/sole_interleaved_mlp_shader.h'].decode().split('R"SOLEMLP(',1)[1].split(')SOLEMLP"',1)[0].encode();coop=current['src/backends/metal/cooperative_sole_prefill_shader.h'].decode().split('R"COOPMLP(',1)[1].split(')COOPMLP"',1)[0].encode();dual=current['src/backends/metal/sole_dual_projection_shader.h'].decode().split('R"SOLEDUAL(',1)[1].split(')SOLEDUAL"',1)[0].encode()
for role in roles:
 assert(d/('native-'+role+'-sole_projection_words.bin.metal')).read_bytes()==(sole+coop if role=='compat67'else b'')
 assert(d/('native-'+role+'-sole_dual_projection_words.bin.metal')).read_bytes()==(sole+coop+dual if role=='compat67'else b'')
 lines=(d/('native-'+role+'-sole_projection_words.bin.prefill-pipelines.tsv')).read_text().splitlines();assert len(lines)==(4 if role=='compat67'else 0)
 for line in lines:
  name,cooperative,width,threads,memory,limit=line.split('\\t');assert int(cooperative)==1 and int(width)==32 and int(threads)>=256 and int(memory)<=int(limit)==32768
import struct
raw=(d/'native-on-primary_simd_metadata_words.bin').read_bytes();assert raw==(d/'native-fault-primary_simd_metadata_words.bin').read_bytes()and all(not(d/('native-'+role+'-primary_simd_metadata_words.bin')).read_bytes()for role in ['off','primary','compat67']);position=0;records=0;words=0
for inner in [128,256,512]:
 for pattern in range(3):
  n=260*inner
  for prefill in [0,1]:
   for start,count in [(0,n),(1,1),(127,3),(inner-1,7),(3*inner+125,517),(n-1,1)]:
    assert struct.unpack('6Q',raw[position:position+48])==(260,inner,pattern,prefill,start,count);position+=48;values=struct.unpack(str(count)+'I',raw[position:position+count*4]);position+=count*4
    for j,value in enumerate(values):
     i=start+j;row,k=divmod(i,inner);block=row//4*(inner//128)+k//128;want=i&65535 if pattern==0 else 0x3f00+(i*13)%128
     if pattern==2 and block%3==1:want=i*19&65535
     if want&0x7f80==0x7f80:want=0x8000
     assert value==want<<16
    records+=1;words+=count
assert records==108 and words==1407282 and position==len(raw)
m=load(d/'metadata-word-comparison.json');assert m['record_count']==108 and m['bytes']==len(raw)and m['sha256']==hashlib.sha256(raw).hexdigest()
# Classify the sole controller OUTPUT identity change, retaining actual exit1.
identity=load(d/'native-driver-identities-before-dispatch.json')['files'];workprefix='/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-primary-simd-metadata/';changed=[]
for path,digest in identity.items():
 assert path.startswith(workprefix);relative=path[len(workprefix):];assert(d/relative).is_file()
 if sha(d/relative)!=digest:changed.append(relative)
assert changed==['native-results.json']and sha(d/'failed-first-native/native-results.json')==identity[workprefix+'native-results.json'];audit=load(d/'native-independent-audit.json');assert audit['observed_controller_exit_code']==1 and audit['native_tests_pass']==50 and audit['sole_frozen_mismatch_is_controller_OUTPUT_native_results']
'''
# Original guard and learned result checks are reused, with only archived paths.
v=(b/'work/worldsense-opt/verify_learned_ws68.py').read_text();a=v.index("for name,info in load(w/'learned-guard-adaptation-proof.json').items():");z=v.index("assert all(load(w/'api-correctness-result.json').values())",a)
guard=v[a:z].replace("original=(old/name).read_text()","original=batch([ref+':'+prefix+name])[0].decode()").replace('(w/name)','(d/name)').replace("load(w/","load(d/")
t+=guard
core=v[z:v.index("assert load(w/'native-independent-audit.json')",z)]
core=core.replace("w.parent/'worldsense-primary-blocks/matrix-selection.json'","d/'frozen-native-guards/matrix-selection.json'").replace("w.parent/'worldsense-prefix-cache/state-guard.json'","d/'frozen-native-guards/state-guard.json'").replace("w.parent/'worldsense-prefix-cache/prefix-model-guard.json'","d/'frozen-native-guards/prefix-model-guard.json'").replace("w.parent/'worldsense-request-residency/vision-grid-off.json'","d/'frozen-native-guards/vision-grid-original.json'").replace('load(w/','load(d/').replace('(w/row[','(d/row[')
core=core.replace(";d=json.loads(line.split(' ',1)[1]);enabled=",";route_data=json.loads(line.split(' ',1)[1]);enabled=").replace("assert d['enabled']==enabled and len(d['checks'])==34","assert route_data['enabled']==enabled and len(route_data['checks'])==34").replace("for v in d['checks']","for v in route_data['checks']")
t+='def canonical_learned_checks():\n'+__import__('textwrap').indent(core,' ')+ '\ncanonical_learned_checks()\n'
# Fixed golden/metric/threshold trees compared directly against the immutable seed.
t+='''assert len(load(d/'flag-dependency-proof.json'))==28 and load(d/'admission-complete.json')['all_expected_config_rejections']
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
for rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
 prior=batch([ref+':'+prefix+rel])[0];assert prior is not None
 assert ast.dump(Normalize().visit(ast.parse(prior)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse((d/rel).read_text().replace('worldsense-primary-simd-metadata','worldsense-cooperative-prefill'))),include_attributes=False),rel
'''
# WS66's raw/functional/actualWS10 layout has the same 32-call cohort design; all math unchanged.
v=(b/'work/worldsense-opt/verify_archive_ws66.py').read_text();tail=v[v.index("gold=load(d/'oracles/worldsense-golden.json')"):]
tail=tail.replace('source_files=1723','source_files=1726').replace('ws66-archive-independent-audit','ws68-archive-independent-audit')
tail=tail.replace("source_overlays=len(recon['overlay_files']),", "source_overlays=len(recon['overlay_files']),new_metadata_records=108,new_metadata_words=1407282,native_controller_exit_code=1,native_CTest_pass=50,")
t+=tail
(b/'work/worldsense-opt/verify_archive_ws68.py').write_text(t)
print('Prepared independent canonical WS68 source/native/learned/32-call original-gate verifier; no inference.')
