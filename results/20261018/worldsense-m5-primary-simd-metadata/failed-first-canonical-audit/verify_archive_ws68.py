"""Independent canonical source reconstruction, native words, original goldens and decision."""
from pathlib import Path
import json,hashlib,subprocess,csv,statistics,ast,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');repo=Path('/Users/shawn/CantorAI/Garnet');d=repo/'results/20261018/worldsense-m5-primary-simd-metadata';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=load(d/'file-manifest.json');assert set(manifest)=={str(p.relative_to(d))for p in d.rglob('*')if p.is_file()and p.name!='file-manifest.json'}and all(sha(d/k)==v for k,v in manifest.items())
def batch(refs):
 raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(refs)+'\n').encode());offset=0;out=[]
 for _ in refs:
  end=raw.index(b'\n',offset);head=raw[offset:end].split();offset=end+1
  if head[-1]==b'missing':value=None
  else:assert head[1]==b'blob';size=int(head[2]);value=raw[offset:offset+size];offset+=size+1
  out.append(value)
 assert offset==len(raw);return out
def reconstruct(label):
 recon=load(d/(label+'source-reconstruction.json'));m=load(d/recon['source_manifest']);original=batch([recon['base_commit']+':'+rel for rel in m]);result={}
 for (rel,digest),value in zip(m.items(),original):
  if rel in recon['overlay_files']:value=(d/recon['overlay_root']/rel).read_bytes()
  assert value is not None and hashlib.sha256(value).hexdigest()==digest;result[rel]=value
 assert len(result)==recon['source_files'];return result,recon
current,recon=reconstruct('');legacy,lr=reconstruct('legacy-');assert len(current)==1726 and len(legacy)==1706
ref='a26bf3059642478dcecff9b0687a0c9ac25959d3';prefix='results/20261018/worldsense-m5-cooperative-prefill/'
sm,sr=[json.loads(v)for v in batch([ref+':'+prefix+'correctness-source-manifest.json',ref+':'+prefix+'source-reconstruction.json'])];assert sm==load(d/'seed-source-manifest.json')and len(sm)==1724
seed=dict(zip(sm,batch([(ref+':'+prefix+'source/'+rel)if rel in sr['overlay_files']else sr['base_commit']+':'+rel for rel in sm])));assert all(hashlib.sha256(seed[k]).hexdigest()==v for k,v in sm.items())
delta=sorted(k for k,v in current.items()if seed.get(k)!=v);assert delta==sorted(['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/primary_simd_metadata_shader.h','src/entry/metal_main.cpp','test2026/metal/primary_simd_metadata_test.cpp'])
for rel,ops in load(d/'source-transformations.json').items():
 text=current[rel].decode()
 for op in reversed(ops):
  if op.get('append'):assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text.encode()==seed[rel],rel
assert all(current[k]==v for k,v in seed.items()if k not in load(d/'source-transformations.json'))
roles=['off','primary','on','fault','compat67']
assert load(d/'native-results.json')=={role:0 for role in roles}
for role in roles:
 text=(d/('native-'+role+'.log')).read_text();assert sorted(int(i)for i in __import__('re').findall(r'\d+/10\s+Test\s+#\s*(\d+):[^\n]*Passed',text))==list(range(1,11))
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
  name,cooperative,width,threads,memory,limit=line.split('\t');assert int(cooperative)==1 and int(width)==32 and int(threads)>=256 and int(memory)<=int(limit)==32768
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
for name,info in load(d/'learned-guard-adaptation-proof.json').items():
 original=batch([ref+':'+prefix+name])[0].decode();text=(d/name).read_text();assert hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']and hashlib.sha256(text.encode()).hexdigest()==info['current_sha256']
 for op in reversed(info['operations']):assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text.replace('worldsense-primary-simd-metadata','worldsense-cooperative-prefill')==original
assert all(load(d/'api-correctness-result.json').values())
corpus=load(d/'corpus-GPU-bits.json');t=corpus['totals'];assert t['packed_matrices']==201 and t['raw_small_matrices']==245 and t['HOST_raw_FP32_words_checked']==1724383232 and t['retained_primary_and_small_raw_bytes']==9850930176
assert t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600 and all(v==4200857600 for v in t['GPU_prefill_tile_FP32_words_checked'].values())and set(t['GPU_prefill_tile_FP32_words_checked'])=={'32x32','64x64','64x32'}
sole=[v for v in corpus['matrices']if v['GPU_sole_original_FP32_words_checked']];assert len(sole)==84 and sum(v['GPU_sole_original_FP32_words_checked']for v in sole)==1056964608
assert t['GPU_sole_prefill_tile_FP32_words_checked']=={'32x32':1056964608,'64x64':1056964608,'64x32':1056964608}
for row in sole:assert row['model']=='Qwen3-1.7B'and row['GPU_sole_original_FP32_words_checked']==row['rows']*row['inner']and row['GPU_sole_prefill_tile_FP32_words_checked']=={key:row['rows']*row['inner']for key in ['32x32','64x64','64x32']}
new=load(d/'simd-metadata-corpus.json');assert new['status']=='both_exact_serving_helpers_all201_original_words_exact'and new['totals']==dict(matrices=201,skipped_raw=245,decode_helper_original_FP32_words_checked=4200857600,prefill_helper_original_FP32_words_checked=4200857600)
expected={(row['model'],row['tensor']):(row['rows'],row['inner'])for row in corpus['matrices']if row['packed']};assert len(new['matrices'])==201 and {(row['model'],row['tensor']):(row['rows'],row['inner'])for row in new['matrices']}==expected
for row in new['matrices']:assert row['owner_accounting_restored']and row['serving_counters_unchanged']and row['decode_helper_original_FP32_words_checked']==row['prefill_helper_original_FP32_words_checked']==row['rows']*row['inner']
assert load(d/'direct-upload-corpus.json')['totals']==dict(weights=546,FP32_words_checked=334609152)
paired=load(d/'dual-pair-corpus.json');assert paired['totals']==dict(pairs=28,matrices=56,original_FP32_words_checked=704643072)
names={row['tensor']for row in load(d/'frozen-native-guards/matrix-selection.json')['matrices']if row['model']=='Qwen3-1.7B'and row['tensor'].endswith(('.mlp.gate_proj.weight','.mlp.up_proj.weight'))};assert names=={row[key]for row in paired['pairs']for key in ['gate','up']}
assert all(row['same_decode_vector_helper']and row['owner_accounting_restored']and row['serving_counters_unchanged']and row['original_FP32_words_checked']==25165824 for row in paired['pairs'])
resident=load(d/'resident-results.json');assert len(resident)==10 and {row['mode']for row in resident}=={'off','primary','on','compat67','profile-compat67'}and all(row['returncode']==0 for row in resident)
for row in resident:
 line=next(line for line in(d/row['log']).read_text().splitlines()if line.startswith('dual_projection_generic_resident_routes '));d=json.loads(line.split(' ',1)[1]);enabled=row['mode']=='compat67';assert d['enabled']==enabled and len(d['checks'])==34 and all(v==dict(encoded_pairs=int(enabled),represented_projections=2*int(enabled))for v in d['checks'])
vlm=load(d/'frozen-native-guards/state-guard.json')['records'][1];prefix={v['prompt_index']:v for v in load(d/'frozen-native-guards/prefix-model-guard.json')['records']};records={role:load(d/('api-state-progress-'+role+'.json'))['records']for role in ['off','primary','on','compat67']}
for role,rows in records.items():
 assert len(rows)==5
 for i,row in enumerate(rows):
  ref=vlm if i<2 else prefix[[0,1,0][i-2]];c=row['call'];assert row['states']==ref['states']and c['generated_token_ids']==ref['generated_token_ids'];baseline=records['off'][i]['call'];assert all(c[key]==baseline[key]for key in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
  t=c['timings'];enabled=role!='off';full=role=='compat67';assert ('primary_simd_metadata'in c['backend'])==(role=='on')
  for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:assert(cap in c['backend'])==enabled
  for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_full_isolated_library','cached_matrix_metadata','sole_interleaved_mlp','sole_direct_interleaved_prefill','sole_dual_projection_decode','sole_cooperative_prefill']:assert(cap in c['backend'])==full
  assert t['lazy_default_rehydrated_bytes']==0 and(t['lazy_default_retired_bytes']>0)==enabled and(t['lazy_default_planned_slots']>0)==enabled
  stats=t['primary_storage_after'];delta=t['primary_encoded_delta'];sole=t['sole_storage_after'];sd=t['sole_encoded_delta'];dd=t['dual_projection_delta'];db=t['dual_projection_before'];da=t['dual_projection_after'];assert set(dd)==set(db)==set(da)=={'encoded_pairs','represented_projections'}and dd=={k:da[k]-db[k]for k in dd}and all(v['represented_projections']==2*v['encoded_pairs']for v in [dd,db,da])
  if enabled and i<2:assert stats['active_weights']==201 and stats['original_bytes']==8401715200 and stats['payload_bytes']+stats['metadata_bytes']==6402163712 and stats['metadata_bytes']==32819200 and all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])and delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']
  else:assert all(v==0 for v in delta.values())
  if full and i>=2:assert sole['active_weights']==84 and sole['allocated_bytes']==2113929216 and sole['scratch_bytes']==0 and sole['restore_encoded']==0 and sd['decode_encoded']>0 and dd['encoded_pairs']>=28*(c['generated_tokens']-1)>0
  else:assert all(v==0 for v in sole.values())and all(v==0 for v in sd.values())and all(v==0 for v in dd.values())
for role in ['off','on','compat67']:assert load(d/('vision-grid-'+role+'.json'))==load(d/'frozen-native-guards/vision-grid-original.json')
assert len(load(d/'flag-dependency-proof.json'))==28 and load(d/'admission-complete.json')['all_expected_config_rejections']
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
for rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
 prior=batch([ref+':'+prefix+rel])[0];assert prior is not None
 assert ast.dump(Normalize().visit(ast.parse(prior)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse((d/rel).read_text().replace('worldsense-primary-simd-metadata','worldsense-cooperative-prefill'))),include_attributes=False),rel
gold=load(d/'oracles/worldsense-golden.json');refs={(i['observation_id'],phase):(i,c)for i in gold['items']if i['policy']=='candidate'for phase,c in i['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'}
rawcalls=[];parents=[];metrics={};wall={};maxima={}
for stage,folder,labels,count in [('resource',d,['control-1','candidate-1'],2),('production-speed',d/'production-speed',['control-1','candidate-1'],5),('functional-correctness-only',d/'functional-smoke',['candidate-1'],2)]:
 path=d/'production-speed/analyze.py'if stage=='production-speed'else d/'analyze_resource.py';fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'archived route','exec'),ns)
 for label in labels:
  data=load(folder/(label+'.json'));assert data['valid']and data['assignments_restored']and len(data['items'])==count and not data['search_plans'];calls=[]
  for i,item in enumerate(data['items']):
   assert(item['repetition'],item['warmup'],item['policy'])==(i,i==0,'default');parents.append((stage,label,i,item))
   for phase,c in item['result']['model_response'].items():
    if not isinstance(c,dict)or c.get('status')!='ok':continue
    ri,refcall=refs[(item['observation_id'],phase)];stable=lambda x:{k:v for k,v in x.items()if k not in ['model_response','model_response_json']};assert stable(item['result'])==stable(ri['result'])
    assert all(c[k]==refcall[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])and c['timings']['input_tokens']==refcall['timings']['input_tokens']and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256'];assert ns['native_route'](c,label=='candidate-1')
    calls.append((item,c));rawcalls.append((stage,label,i,phase,item,c))
  assert len(calls)==count*2
  if stage=='resource':maxima[label]={key:max(c['timings']['memory_after_decode'][key]for _,c in calls)for key in ['metal_allocated_bytes','task_physical_footprint_bytes']}
  if stage=='production-speed':
   metrics[label]={}
   for model in ['Qwen3-1.7B','Qwen3.5-4B']:
    cs=[c for i,c in calls if not i['warmup']and c['model_id']==model];assert len(cs)==4;metrics[label][model]=dict(samples=4,ttft=statistics.median(c['ttft_seconds']for c in cs),total=statistics.median(c['total_seconds']for c in cs),decode_tps=statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in cs))
   wall[label]=statistics.median(i['native_call_seconds']for i in data['items']if not i['warmup'])
assert len(rawcalls)==32 and len(parents)==16
resource=load(d/'resource-analysis.json');assert maxima==resource['sampled_maximum_bytes'];reductions={k:1-maxima['candidate-1'][k]/maxima['control-1'][k]for k in maxima['control-1']};assert reductions==resource['reduction']and reductions['metal_allocated_bytes']>=.10 and reductions['task_physical_footprint_bytes']>=.05 and resource['passed']
speed=load(d/'production-speed/analysis.json');assert metrics==speed['metrics']and wall==speed['native_skill_wall_p50']
comparisons=[dict(model=model,ttft_change=metrics['candidate-1'][model]['ttft']/metrics['control-1'][model]['ttft']-1,total_change=metrics['candidate-1'][model]['total']/metrics['control-1'][model]['total']-1,decode_gain=metrics['candidate-1'][model]['decode_tps']/metrics['control-1'][model]['decode_tps']-1)for model in sorted(metrics['control-1'])];assert comparisons==speed['comparisons'];gain=1-wall['candidate-1']/wall['control-1'];assert gain==speed['native_skill_wall_gain']
assert not(gain>=-.02 and all(row['ttft_change']<=.02 and row['total_change']<=.02 and row['decode_gain']>=-.02 for row in comparisons))and speed['passed']==False and speed['status']=='rejected'and speed['sampled_max_post_request_metal_bytes']['control-1']is None
calls=list(csv.DictReader((d/'calls.csv').open()));pr=list(csv.DictReader((d/'skill-requests.csv').open()));mem=list(csv.DictReader((d/'process-memory-samples.csv').open()));assert len(calls)==32 and len(pr)==16 and len(mem)==22
lookup={(row['stage'],row['worker'],int(row['repetition']),row['phase']):row for row in calls}
for stage,label,i,phase,item,c in rawcalls:
 row=lookup[(stage,label,i,phase)];assert float(row['ttft_seconds'])==c['ttft_seconds']and float(row['total_seconds'])==c['total_seconds']and int(row['generated_tokens'])==c['generated_tokens']and float(row['derived_decode_tokens_per_second'])==(c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])and row['eligible_for_speed']==str(stage=='production-speed'and not item['warmup'])
inputs=load(d/'frozen-input-identities.json');assert sha(d/inputs['frame_relative'])==inputs['image_sha256']and sha(d/inputs['serialized_skill_relative'])==inputs['recorded_skill_sha256']
assert load(d/'decision.json')['decision']=='valid-reject'and not load(d/'decision.json')['promotion']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_files=len(manifest),source_files=1726,source_overlays=len(recon['overlay_files']),new_metadata_records=108,new_metadata_words=1407282,native_controller_exit_code=1,native_CTest_pass=50,legacy_files=1706,legacy_overlays=len(lr['overlay_files']),original_guard_and_all_five_metric_gate_ASTs_inverse_exact=True,original186_MLP28_new36_complete_native_records_scalar_three_tile_paired_vector_learned_words_states_tokens_vision_exact=True,all32_pipeline_goldens_native_routes_CSV_resource_speed_and_rejection_recomputed=True,calls=32,requests=16,memory_rows=22,decision='valid-reject',promotion=False)
(b/'work/worldsense-opt/ws68-archive-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
