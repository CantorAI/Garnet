from pathlib import Path
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i'); p=b/'work/worldsense-opt/verify_archive_ws66.py';t=p.read_text()
t=t.replace('worldsense-m5-sole-dual-decode','worldsense-m5-cooperative-prefill').replace('ws66-archive','ws67-archive')
t=t.replace("len(current)==1723", "len(current)==1724").replace("ref='a6924be03a4b36a3472677d17f0d16a4f8bdfb5a';prefix='results/20261018/worldsense-m5-direct-interleaved-prefill/'", "ref='a13a77a5880103fed1c2ef44b5354ba22a2acb63';prefix='results/20261018/worldsense-m5-sole-dual-decode/'").replace('len(sm)==1721','len(sm)==1723')
a=t.index('delta=sorted(');z=t.index("for rel,ops in load(d/'source-transformations.json').items():",a)
t=t[:a]+'''delta=sorted(k for k,v in current.items()if seed.get(k)!=v);assert delta==sorted(['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/cooperative_sole_prefill_shader.h','src/entry/metal_main.cpp','test2026/metal/sole_interleaved_test.cpp'])
'''+t[z:]
t=t.replace("if k not in load(d/'source-transformations.json')", "if k not in load(d/'source-transformations.json')") # New file excluded naturally from seed iteration.
t=t.replace("len(load(d/'native-results.json'))==6", "len(load(d/'native-results.json'))==7")
t=t.replace("['primary','on','fault','restore','direct64']", "['primary','on','fault','restore','direct64','dual66']")
t=t.replace("['off','primary','on','fault','restore','direct64']", "['off','primary','on','fault','restore','direct64','dual66']")
t=t.replace("['off','primary','on','restore','direct64']", "['off','primary','on','restore','direct64','dual66']")
a=t.index("assert(d/'actual-on-sole-program.metal')");z=t.index("dual_fragment=current[",a)
t=t[:a]+'''cooperative=current['src/backends/metal/cooperative_sole_prefill_shader.h'].decode().split('R"COOPMLP(',1)[1].split(')COOPMLP"',1)[0].encode()
assert(d/'actual-on-sole-program.metal').read_bytes()==(d/'actual-fault-sole-program.metal').read_bytes()==old_sole+cooperative
assert(d/'actual-restore-sole-program.metal').read_bytes()==old_sole
for role in ['direct64','dual66']:assert(d/('actual-'+role+'-sole-program.metal')).read_bytes()==old_sole+fragment
for rel in ['src/backends/metal/sole_interleaved_mlp_shader.h','src/backends/metal/direct_interleaved_prefill_shader.h','src/backends/metal/sole_dual_projection_shader.h','src/backends/metal/compiled_graph.cpp','runtime/metal_serving.py']:
 assert current[rel]==seed[rel]
for name,proof in load(d/'cooperative-kernel-inverse-proof.json').items():
 original=proof['original_source'];new=proof['new_source'];assert original in fragment.decode()and new in cooperative.decode()
 assert hashlib.sha256(original.encode()).hexdigest()==proof['original_sha256']and hashlib.sha256(new.encode()).hexdigest()==proof['new_sha256']
 for op in reversed(proof['operations']):assert new.count(op['after'])==op['count'];new=new.replace(op['after'],op['before'])
 assert new==original
assert 'return ulong2((i/4)%tileK,(i/(tileK*4))*4+i%4);'in cooperative.decode()and 'return k*(tileN+4)+column;'in cooperative.decode()
cap=load(d/'actual-prefill-pipeline-capabilities.json')['roles'];assert set(cap)=={'off','primary','on','fault','restore','direct64','dual66'}
for role,rows in cap.items():
 assert len(rows)==(4 if role in ['on','fault','direct64','dual66']else 0)
 for row in rows:
  assert row['cooperative']==(role in ['on','fault'])and row['thread_width']==32 and row['max_threads']>=256 and row['static_threadgroup_bytes']<=row['device_max_threadgroup_bytes']==32768
  expected=dict(base32x32=8704,wide16_64x64=25600,wide32_64x32=20992,wordprobe=17408)if row['cooperative']else dict(base32x32=8192,wide16_64x64=24576,wide32_64x32=20480,wordprobe=16384)
  assert row['static_threadgroup_bytes']==expected[row['name']]
'''+t[z:]
a=t.index(" assert (d/('actual-'+role+'-dual-program.metal'))");z=t.index(' import re',a)
t=t[:a]+''' expected=(old_sole+(cooperative if role in ['on','fault']else fragment)+dual_fragment)if role in ['on','fault','dual66']else b''
 assert (d/('actual-'+role+'-dual-program.metal')).read_bytes()==expected
'''+t[z:]
t=t.replace("len(resident)==12", "len(resident)==14").replace("enabled=row['mode']=='on'", "enabled=row['mode']in ['on','dual66']")
t=t.replace("text.replace('source1723','source1721').replace('all6builds','all5builds').replace('each6builds','each5builds')", "text.replace('source1724','source1723').replace('all7builds','all6builds').replace('each7builds','each6builds')")
t=t.replace("text.replace('worldsense-sole-dual-decode','worldsense-direct-interleaved-prefill')", "text.replace('worldsense-cooperative-prefill','worldsense-sole-dual-decode')")
t=t.replace("len(load(d/'flag-dependency-proof.json'))==23 and load(d/'admission-complete.json')['twentythree_expected_config_rejections']", "len(load(d/'flag-dependency-proof.json'))==25 and load(d/'admission-complete.json')['twentyfive_expected_config_rejections']")
t=t.replace(".replace('worldsense-sole-dual-decode','worldsense-direct-interleaved-prefill')", ".replace('worldsense-cooperative-prefill','worldsense-sole-dual-decode')")
t=t.replace("rawcalls=[];parents=[];metrics={};wall={};maxima={}", "rawcalls=[];parents=[];metrics={};wall={};maxima={};primarymetrics={};primarywall={};primarymetal={};primaryfoot={}")
t=t.replace("('functional-correctness-only',d/'functional-smoke',['candidate-1'],2)", "('functional-correctness-only',d/'functional-smoke',['candidate-1'],2),('primary-speed',d/'primary-speed',['control-1','candidate-1'],5)")
t=t.replace("path=d/'production-speed/analyze.py'if stage=='production-speed'else d/'analyze_resource.py'", "path=d/stage/'analyze.py'if stage in ['production-speed','primary-speed']else d/'analyze_resource.py'")
t=t.replace("ns['native_route'](c,label=='candidate-1')", "ns['native_route'](c,label=='candidate-1',True)if stage=='primary-speed'else ns['native_route'](c,label=='candidate-1')")
a=t.index("  if stage=='production-speed':");z=t.index('assert len(rawcalls)',a)
t=t[:a]+'''  if stage in ['production-speed','primary-speed']:
   target=metrics if stage=='production-speed'else primarymetrics;walls=wall if stage=='production-speed'else primarywall;target[label]={}
   for model in ['Qwen3-1.7B','Qwen3.5-4B']:
    cs=[c for i,c in calls if not i['warmup']and c['model_id']==model];assert len(cs)==4;target[label][model]=dict(samples=4,ttft=statistics.median(c['ttft_seconds']for c in cs),total=statistics.median(c['total_seconds']for c in cs),decode_tps=statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in cs))
   walls[label]=statistics.median(i['native_call_seconds']for i in data['items']if not i['warmup'])
   if stage=='primary-speed':
    primarymetal[label]=max(c['timings']['memory_after_decode']['metal_allocated_bytes']for _,c in calls);primaryfoot[label]=max(c['timings']['memory_after_decode']['task_physical_footprint_bytes']for _,c in calls)
'''+t[z:]
t=t.replace('len(rawcalls)==32 and len(parents)==16','len(rawcalls)==52 and len(parents)==26')
a=t.index('assert not(gain>=-.02');z=t.index('calls=list(csv.DictReader',a)
t=t[:a]+'''assert gain>=-.02 and all(row['ttft_change']<=.02 and row['total_change']<=.02 and row['decode_gain']>=-.02 for row in comparisons)and speed['passed']==True and speed['status']=='passed_screen_requires_fresh_full_confirmation'and speed['sampled_max_post_request_metal_bytes']['control-1']is None
primary=load(d/'primary-speed/analysis.json');assert primarymetrics==primary['metrics']and primarywall==primary['native_skill_wall_p50']and primarymetal==primary['sampled_max_post_request_metal_bytes']and primaryfoot==primary['sampled_max_footprint_bytes']
pc=[dict(model=model,ttft_change=primarymetrics['candidate-1'][model]['ttft']/primarymetrics['control-1'][model]['ttft']-1,total_change=primarymetrics['candidate-1'][model]['total']/primarymetrics['control-1'][model]['total']-1,decode_gain=primarymetrics['candidate-1'][model]['decode_tps']/primarymetrics['control-1'][model]['decode_tps']-1)for model in sorted(primarymetrics['control-1'])];pgain=1-primarywall['candidate-1']/primarywall['control-1'];vgain=1-primarymetrics['candidate-1']['Qwen3.5-4B']['ttft']/primarymetrics['control-1']['Qwen3.5-4B']['ttft'];mpreserved=primarymetal['candidate-1']/primarymetal['control-1']-1<=.02 and primaryfoot['candidate-1']/primaryfoot['control-1']-1<=.02
assert pc==primary['comparisons']and pgain==primary['native_skill_wall_gain']and vgain==primary['observed_VLM_TTFT_gain']and mpreserved==primary['matched_memory_preserved']
assert not(mpreserved and vgain>=.05 and pgain>=-.02 and all(v['ttft_change']<=.02 and v['total_change']<=.02 and v['decode_gain']>=-.02 for v in pc))and not primary['passed']and primary['status']=='rejected'
'''+t[z:]
t=t.replace('len(calls)==32 and len(pr)==16 and len(mem)==22','len(calls)==52 and len(pr)==26 and len(mem)==42')
t=t.replace("stage=='production-speed'and not item['warmup']", "stage in ['production-speed','primary-speed']and not item['warmup']")
t=t.replace('source_files=1723','source_files=1724').replace('all32_pipeline','all52_pipeline').replace('calls=32,requests=16,memory_rows=22','calls=52,requests=26,memory_rows=42')
(b/'work/worldsense-opt/verify_archive_ws67.py').write_text(t)
print('Prepared independent WS67 canonical verifier: full original guards and two distinct speed cohorts, no new inference.')
