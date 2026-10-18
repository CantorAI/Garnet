from pathlib import Path
import ast
p=Path('work/worldsense-opt')
t=(p/'archive_ws64.py').read_text().replace('worldsense-direct-interleaved-prefill','worldsense-sole-interleaved-text').replace('worldsense-m5-direct-interleaved-prefill','worldsense-m5-sole-interleaved-text').replace("['off','primary','on','fault','restore']","['off','primary','on','fault','restore','direct64']").replace("'.safetensors'}","'.safetensors','.gz'}").replace("'*ws64*.py'","'*ws65*.py'").replace('candidate_source_files=1721','candidate_source_files=1722');ast.parse(t);(p/'archive_ws65.py').write_text(t)
t=(p/'verify_archive_ws64.py').read_text().replace('worldsense-m5-direct-interleaved-prefill','worldsense-m5-sole-interleaved-text').replace('len(current)==1721','len(current)==1722').replace("ref='40780f3cb39d334c400820903d30d15526c51945';prefix='results/20261018/worldsense-m5-sole-interleaved-mlp/'","ref='a6924be03a4b36a3472677d17f0d16a4f8bdfb5a';prefix='results/20261018/worldsense-m5-direct-interleaved-prefill/'").replace('len(sm)==1720','len(sm)==1721')
a="sorted(['cmake/GarnetMetal.cmake','runtime/metal_serving.py','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/direct_interleaved_prefill_shader.h','src/entry/metal_main.cpp','test2026/metal/sole_interleaved_test.cpp'])";assert t.count(a)==1;t=t.replace(a,"sorted(['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/sole_text_test.cpp'])")
t=t.replace("len(load(d/'native-results.json'))==5","len(load(d/'native-results.json'))==6").replace("['primary','on','fault','restore']","['primary','on','fault','restore','direct64']")
start=t.index("for name,item in load(d/'read-only-prefill-reader-audit.json')");end=t.index('guards=load(',start)
t=t[:start]+'''assert (d/'actual-direct64-sole-program.metal').read_bytes()==old_sole+fragment
assert current['src/backends/metal/sole_interleaved_mlp_shader.h']==seed['src/backends/metal/sole_interleaved_mlp_shader.h']
assert current['src/backends/metal/direct_interleaved_prefill_shader.h']==seed['src/backends/metal/direct_interleaved_prefill_shader.h']
newtest=current['test2026/metal/sole_text_test.cpp'].decode();proof=load(d/'new-native-test-template-proof.json');assert hashlib.sha256(newtest.encode()).hexdigest()==proof['current_sha256']
for op in reversed(proof['operations']):assert newtest.count(op['after'])==op['count'];newtest=newtest.replace(op['after'],op['before'])
assert newtest.encode()==seed['test2026/metal/sole_interleaved_test.cpp'] and hashlib.sha256(newtest.encode()).hexdigest()==proof['seed_sha256']
import gzip,struct
proof=load(d/'sole-text-word-comparison.json');assert proof['record_count']==42 and proof['all_six_builds_complete_FP32_words_identical']
roles=['off','primary','on','fault','restore','direct64'];reference=d/'native-off-sole-text-projection-words.bin.gz'
expected=[(rows,2048,columns,passid)for columns in [2048,1024,151936]for rows in [1,2,3,16,32,64,128]for passid in [0,1]]
with gzip.open(reference,'rb')as f:
 complete=hashlib.sha256();total=0
 for item,shape in zip(proof['records'],expected):
  header=f.read(32);assert struct.unpack('4Q',header)==shape and list(shape)==item['shape'];data=f.read(shape[0]*shape[2]*4);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'];complete.update(header);complete.update(data);total+=32+len(data)
 assert not f.read(1) and total==proof['uncompressed_bytes'] and complete.hexdigest()==proof['uncompressed_sha256']
for role in roles:
 path=d/('native-'+role+'-sole-text-projection-words.bin.gz');assert sha(path)==proof['archives'][role]['sha256']
 with gzip.open(path,'rb')as f,gzip.open(reference,'rb')as g:
  while True:
   a=f.read(4*1024*1024);b=g.read(4*1024*1024);assert a==b
   if not a:break
 log=(d/('native-'+role+'.log')).read_text();assert '100% tests passed, 0 tests failed out of 9'in log
''' +t[end:]
a="text=text.replace('source1721','source1720').replace('native8tests_all5builds','native8tests_all4builds').replace('native8tests_each5builds','native8tests_each4builds')";assert t.count(a)==1;t=t.replace(a,"text=text.replace('source1722','source1721').replace('all6builds','all5builds').replace('each6builds','each5builds')")
a="text.replace('worldsense-direct-interleaved-prefill','worldsense-sole-interleaved')";assert t.count(a)==1;t=t.replace(a,"text.replace('worldsense-sole-interleaved-text','worldsense-direct-interleaved-prefill')")
a=".replace('worldsense-direct-interleaved-prefill','worldsense-sole-interleaved')";assert t.count(a)==1;t=t.replace(a,".replace('worldsense-sole-interleaved-text','worldsense-direct-interleaved-prefill')")
t=t.replace('len(sole)==84','len(sole)==197').replace('1056964608','1720451072').replace('in[(6144,2048),(2048,6144)]','in[(6144,2048),(2048,6144),(2048,2048),(1024,2048),(151936,2048)]').replace("['off','primary','on','restore']","['off','primary','on','restore','direct64']").replace("len(load(d/'flag-dependency-proof.json'))==21","len(load(d/'flag-dependency-proof.json'))==23").replace('twentyone_expected_config_rejections','twentythree_expected_config_rejections').replace('source_files=1721','source_files=1722').replace('original186_new28_complete_native_records_and_scalar_three_tile_learned_words_states_tokens_vision_exact','original186_MLP28_new42_complete_native_records_and_scalar_three_tile_learned_words_states_tokens_vision_exact').replace('ws64-archive-independent-audit.json','ws65-archive-independent-audit.json');ast.parse(t);(p/'verify_archive_ws65.py').write_text(t)
print('WS65 complete lossless archive and independent reconstruction/original-oracle verifier prepared')
