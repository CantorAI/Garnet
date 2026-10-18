"""Independent canonical source reconstruction, native words, original goldens and decision."""
from pathlib import Path
import json,hashlib,subprocess,csv,statistics,ast,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');repo=Path('/Users/shawn/CantorAI/Garnet');d=repo/'results/20261018/worldsense-m5-packed-request-residency';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
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
current,recon=reconstruct('');legacy,legacy_recon=reconstruct('legacy-');assert len(current)==1721 and len(legacy)==1706
seed_ref='290f5742ba51c41197f24179cfee4876e96e789a';seed_prefix='results/20261017/worldsense-m5-request-weight-residency/'
seed,seed_recon=[json.loads(v)for v in batch([seed_ref+':'+seed_prefix+'correctness-source-manifest.json',seed_ref+':'+seed_prefix+'source-reconstruction.json'])]
assert seed==load(d/'seed-source-manifest.json')and len(seed)==1720
seed_source=dict(zip(seed,batch([(seed_ref+':'+seed_prefix+'source/'+rel)if rel in seed_recon['overlay_files']else seed_recon['base_commit']+':'+rel for rel in seed])))
assert all(hashlib.sha256(seed_source[k]).hexdigest()==v for k,v in seed.items())
delta=sorted(k for k,v in current.items()if seed_source.get(k)!=v)
assert delta==sorted(['cmake/GarnetMetal.cmake','runtime/metal_serving.py','src/backends/metal/decoder_session.cpp','src/backends/metal/decoder_session.h','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/packed_request_residency_test.cpp'])
source=load(d/'correctness-source-manifest.json')
transforms=load(d/'source-transformations.json')
for rel,ops in transforms.items():
 text=current[rel].decode()
 for op in reversed(ops):
  if op.get('append'):assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text==seed_source[rel].decode(),rel
assert all(source[k]==v for k,v in seed.items()if k not in transforms)
context=load(d/'packed-context-adaptation-proof.json');assert context['original']in(seed_source['src/backends/metal/metal_context.mm'].decode())and context['original']in(current['src/backends/metal/metal_context.mm'].decode())and context['new']in(current['src/backends/metal/metal_context.mm'].decode())
text=context['new']
for op in reversed(context['operations']):assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
assert text==context['renamed_original']and '[impl_->packedRequestResidencySet requestResidency]'in context['new']and ' packedRequestResidency]'not in context['new']
assert 'constexpr uint64_t kPackedRequestResidencyLimit=128*1024;'in context['new']and 'constexpr uint64_t kPackedRequestResidencyLimit=uint64_t(8)*1024*1024*1024;'in context['new']
assert 'owner->primaryPayload,owner->primaryMetadata'in context['new']and 'seen.size()>2048'in context['new']and 'packedRequestResidencySet.allocationCount'in context['new']
session=load(d/'packed-session-adaptation-proof.json');assert session['original']in(seed_source['src/backends/metal/decoder_session.cpp'].decode())and session['original']in(current['src/backends/metal/decoder_session.cpp'].decode())and session['new']in(current['src/backends/metal/decoder_session.cpp'].decode())
roles=['off','primary','on','fault','compat56','compat56-fault']
assert load(d/'native-results.json')=={role:0 for role in roles}
import re,struct,copy
for role in roles:
 log=(d/('native-'+role+'.log')).read_text();assert sorted(int(v)for v in re.findall(r'\d+/9\s+Test\s+#\s*(\d+):[^\n]*Passed',log))==list(range(1,10))and re.search(r'100% tests passed(?:, 0 tests failed)? out of 9',log)
 cache=(d/'build-identities'/role/'CMakeCache.txt').read_text();assert ('GARNET_METAL_PACKED_REQUEST_WEIGHT_RESIDENCY:BOOL=ON'in cache)==(role in ['on','fault'])
 assert ('GARNET_METAL_PACKED_REQUEST_RESIDENCY_TEST_FAULTS:BOOL=ON'in cache)==(role=='fault')
 assert ('GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY:BOOL=ON'in cache)==role.startswith('compat56')
 assert ('GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS:BOOL=ON'in cache)==(role=='compat56-fault')
raw=(d/'native-off-primary_projection_words.bin').read_bytes();assert raw==batch([seed_ref+':'+seed_prefix+'native-off-projection-words.bin'])[0]
assert all((d/('native-'+role+'-primary_projection_words.bin')).read_bytes()==raw for role in roles)
proof=load(d/'native-word-comparison.json');assert proof['records']==186 and proof['bytes']==len(raw)and proof['sha256']==hashlib.sha256(raw).hexdigest()
reference=None
for role in roles:
 data=(d/('native-'+role+'-packed_request_projection_words.bin')).read_bytes()
 if reference is None:reference=data
 assert data==reference;position=0
 for pattern,value in [(0,128.),(1,-1.25)]:
  assert struct.unpack('4Q',data[position:position+32])==(260,128,pattern,260);position+=32;assert data[position:position+1040]==struct.pack('<f',value)*260;position+=1040
 assert position==len(data)and(d/('native-'+role+'-packed_request_projection_words.bin.leased.bin')).read_bytes()==(data if role in ['on','fault']else b'')
packed=load(d/'packed-native-word-comparison.json');assert packed['records']==2 and packed['words_per_reference']==520 and packed['bytes']==len(reference)and packed['sha256']==hashlib.sha256(reference).hexdigest()
programs={role:{kind:(d/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()for kind in ['main','prefill']}for role in roles}
assert all(programs[role]==programs['primary']for role in ['primary','on','fault'])
for role,previous in [('off','off'),('primary','primary'),('compat56','on'),('compat56-fault','fault')]:
 for kind in ['main','prefill']:assert programs[role][kind]==batch([seed_ref+':'+seed_prefix+'actual-'+previous+'-'+kind+'-program.metal'])[0]
assert all(not programs[role]['prefill']for role in ['off','primary','on','fault'])
for role,groups in load(d/'actual-program-proof.json')['programs'].items():
 for kind,info in groups.items():assert len(programs[role][kind])==info['bytes']and hashlib.sha256(programs[role][kind]).hexdigest()==info['sha256']
for name,info in load(d/'learned-guard-adaptation-proof.json').items():
 original=batch([seed_ref+':'+seed_prefix+name])[0].decode();value=(d/name).read_text();assert hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']and hashlib.sha256(value.encode()).hexdigest()==info['current_sha256']
 for op in reversed(info['operations']):assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
 assert value.replace('worldsense-packed-request-residency','worldsense-request-weight-residency')==original
def canonical_learned_checks():
 terminal=load(d/'learned-controller-terminal.json');assert terminal['success']and terminal['normal_restored']and not terminal['promotion']
 journal=load(d/'learned-controller-journal.json');assert len(journal)==19 and all(v['returncode']==0 and v['ended_utc']>=v['started_utc']for v in journal)
 assert all(load(d/'api-correctness-result.json').values())
 corpus=load(d/'corpus-GPU-bits.json');t=corpus['totals']
 assert t['packed_matrices']==201 and t['raw_small_matrices']==245 and t['HOST_raw_FP32_words_checked']==1724383232 and t['retained_primary_and_small_raw_bytes']==9850930176
 assert t['original_bytes']==11850481664 and t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600
 assert t['GPU_prefill_tile_FP32_words_checked']=={k:4200857600 for k in ['32x32','64x32','64x64']}
 assert len(corpus['matrices'])==446
 selection=load(d/'frozen-native-guards/matrix-selection.json')['matrices'];assert {(v['model'],v['tensor'],v['rows'],v['inner'])for v in selection}=={(v['model'],v['tensor'],v['rows'],v['inner'])for v in corpus['matrices']}
 for row in corpus['matrices']:
  assert row['owner_accounting_restored'];count=row['rows']*row['inner']
  if row['packed']:assert row['GPU_original_FP32_words_checked']==row['GPU_vector_original_FP32_words_checked']==count and row['GPU_prefill_tile_FP32_words_checked']=={k:count for k in ['32x32','64x32','64x64']}
  else:assert row['HOST_raw_FP32_words_checked']==count
 leased=load(d/'packed-lease-corpus.json');lt=leased['totals'];assert leased['status']=='all201_two_original_readers_exact_under_actual_packed_lease'
 assert lt['matrices']==201 and lt['skipped_raw']==245 and lt['scalar_original_FP32_words']==lt['vector_original_FP32_words']==4200857600
 assert len(leased['matrices'])==201 and {(v['model'],v['tensor'],v['rows'],v['inner'])for v in leased['matrices']}=={(v['model'],v['tensor'],v['rows'],v['inner'])for v in corpus['matrices']if v['packed']}
 def inactive(q):
  assert not q['scope_open'] and not q['attached'] and not q['requested']
  assert q['active_bytes']==q['global_used_bytes']==q['set_allocation_count']==q['owner_count']==0
 total_alloc=0
 for index,row in enumerate(leased['matrices'],1):
  assert row['owner_accounting_restored']and row['serving_counters_unchanged']and row['scalar_original_FP32_words']==row['vector_original_FP32_words']==row['rows']*row['inner']
  q=row['active_lease'];e=row['after_cleanup'];assert q['calls']==q['admitted']==index and q['ends']==index-1 and q['fallback']==0
  assert q['last_count']==q['attempted_count']==q['set_allocation_count']==2 and q['owner_count']==1 and q['scope_open']and q['attached']and q['requested']
  assert q['active_bytes']==q['global_used_bytes']==q['last_bytes']==q['attempted_bytes']>0
  inactive(e);assert e['calls']==e['admitted']==e['ends']==index and e['fallback']==0;total_alloc+=q['last_bytes']
 assert total_alloc==lt['sum_individual_actual_allocation_bytes']
 assert load(d/'direct-upload-corpus.json')['totals']==dict(weights=546,FP32_words_checked=334609152)
 resident=load(d/'resident-results.json');assert len(resident)==8 and {v['mode']for v in resident}=={'off','primary','on','compat56'}and all(v['returncode']==0 for v in resident)
 private_steps=0
 for row in resident:
  lines=(d/row['log']).read_text().splitlines();steps=[json.loads(line.split(' ',1)[1])for line in lines if line.startswith('private_kv_step ')]
  assert len(steps)==8 and all(v['exact_output_and_state']for v in steps)and any(v.startswith('private_kv_reference PASS ')for v in lines);private_steps+=len(steps)
 for role in ['off','on','fault','compat56']:assert 'Four-input trained linear generic-resident parity/host-private isolation/inactive/malformed/shared-after-source-close/rebuild PASS '+role in(d/('trained-linear-'+role+'.log')).read_text()
 vlm=load(d/'frozen-native-guards/state-guard.json')['records'][1];prefix={v['prompt_index']:v for v in load(d/'frozen-native-guards/prefix-model-guard.json')['records']}
 records={role:load(d/('api-state-progress-'+role+'.json'))['records']for role in ['off','primary','on','compat56']}
 for role,rows in records.items():
  assert len(rows)==5
  for i,row in enumerate(rows):
   ref=vlm if i<2 else prefix[[0,1,0][i-2]];c=row['call'];t=c['timings'];assert row['states']==ref['states']and c['generated_token_ids']==ref['generated_token_ids']
   baseline=records['off'][i]['call'];assert all(c[k]==baseline[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
   enabled=role!='off';compat=role=='compat56'
   for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:assert(cap in c['backend'])==enabled
   for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_full_isolated_library','cached_matrix_metadata','request_scope_weight_residency']:assert(cap in c['backend'])==compat
   assert('packed_request_weight_residency'in c['backend'])==(role=='on')
   assert t['lazy_default_rehydrated_bytes']==0 and(t['lazy_default_retired_bytes']>0)==enabled and(t['lazy_default_planned_slots']>0)==enabled
   p=t['primary_storage_after'];delta=t['primary_encoded_delta']
   if enabled and i<2:
    assert p['active_weights']==201 and p['original_bytes']==8401715200 and p['payload_bytes']+p['metadata_bytes']==6402163712 and p['metadata_bytes']==32819200
    assert all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])and delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']
   else:assert all(v==0 for v in delta.values())
   if role=='on':
    q=t['packed_request_weight_residency'];inactive(q);eligible=i<2;assert t['packed_request_eligible']==eligible and q['limit_bytes']==8589934592
    if eligible:
     assert q['calls']==q['admitted']==q['ends']==[1,3][i]and q['fallback']==q['coverage_fallbacks']==0 and q['reason']==1
     assert q['last_count']==q['attempted_count']==402 and q['last_bytes']==q['attempted_bytes']==p['allocated_bytes']
     assert q['coverage_checks']>0 and q['coverage_cache_hits']>0 and 33<q['last_coverage_graphs']<=256
    else:assert all(q[k]==0 for k in ['calls','ends','admitted','fallback','coverage_checks','coverage_cache_hits'])
   if compat:
    q=t['request_weight_residency'];assert q['calls']==q['ends']and q['active_bytes']==q['global_used_bytes']==0 and not q['scope_open']and q['limit_bytes']==4294967296
    if i<2:assert q['reason']==7 and q['admitted']==0 and q['fallback']==q['calls']
    else:assert q['reason']==1 and q['last_count']==197 and q['last_bytes']==3440902144 and q['coverage_fallbacks']==0
 for role in ['off','on','compat56']:assert load(d/('vision-grid-'+role+'.json'))==load(d/'frozen-native-guards/vision-grid-original.json')
 assert load(d/'request-session-guard-compat56.json')['all_cases_pass']
 assert load(d/'request-full-session-guard-compat56.json')['real197masters_global4GiB_contention_close_readmission_reset']
 layer=load(d/'packed-layer-session-guard.json');assert layer['all_cases_pass']and len(layer['records'])==10
 for row in layer['records']:inactive(row['stats'])
 full=load(d/'packed-full-session-guard.json');assert full['all_cases_pass']and len(full['records'])==8 and full['actual_owner_count']==201 and full['actual_allocation_count']==402 and full['logical_packed_bytes']==6402163712
 inv=load(d/'packed-full-graph-inventory.json');buffers={v['id']:v for v in inv['buffers']};refs=[v for v in inv['references']if v['representation']in ['primary_bf16_payload','primary_bf16_metadata']];ids={v['buffer_id']for v in refs}
 assert len(ids)==402 and sum(buffers[i]['allocated_bytes']for i in ids)==full['actual_allocation_bytes']and sum(buffers[i]['length_bytes']for i in ids)==6402163712
 recovered=next(v for v in full['records']if v['case']=='same_model_after_image_errors_original32layers_states_tokens_exact');assert recovered['states']==vlm['states']and recovered['call']['generated_token_ids']==vlm['generated_token_ids'];inactive(recovered['call']['timings']['packed_request_weight_residency'])
 assert load(d/'native-independent-audit.json')['native_tests_pass']==54

canonical_learned_checks()
full=load(d/'packed-full-session-guard.json');actual=full['actual_allocation_bytes'];assert actual==6405324800
whole=next(v for v in full['records']if v['case']=='whole_actual_201owners_402allocations_nested_and_global_budget');active=whole['active'];contender=whole['contender']
assert active['scope_open']and active['attached']and active['requested']and active['active_bytes']==active['global_used_bytes']==active['last_bytes']==actual and active['owner_count']==201 and active['set_allocation_count']==active['last_count']==402
assert contender['reason']==5 and contender['active_bytes']==0 and contender['global_used_bytes']==actual
thread=next(v for v in full['records']if v['case']=='cross_thread_same_runtime_budget_contention')['stats'];assert thread['reason']==5 and thread['active_bytes']==0 and thread['global_used_bytes']==actual
errors=[v for v in full['records']if v['case']=='native_image_error_outer_finally_cleanup'];assert len(errors)==2
for row in errors:
 assert row['error']['status']=='error';q=row['after'];assert q['calls']==q['admitted']==q['ends']and q['fallback']==0 and q['active_bytes']==q['global_used_bytes']==q['owner_count']==q['set_allocation_count']==0 and not q['scope_open']and not q['attached']and not q['requested']
assert load(d/'native-independent-audit.json')['observed_exit_code']==0
assert len(load(d/'flag-dependency-proof.json'))==12 and load(d/'admission-complete.json')['all_expected_config_rejections']
for name,info in load(d/'flag-dependency-proof.json').items():assert info['returncode']!=0 and info['expected_diagnostic']in' '.join((d/('invalid-'+name+'-config.log')).read_text().split())
metric_ref='b600cf8de6e99e11d35f6727bc3c3cacfc1d5418';metric_prefix='results/20261018/worldsense-m5-primary-simd-metadata/'
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
for rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
 original=batch([metric_ref+':'+metric_prefix+rel])[0];assert original is not None
 assert ast.dump(Normalize().visit(ast.parse(original)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse((d/rel).read_text().replace('worldsense-packed-request-residency','worldsense-primary-simd-metadata'))),include_attributes=False),rel
# Replay the original offline adversarial audit in a separate namespace, only
# against archived calls/oracle. No file writes or new model calls are executed.
adversarial=(d/'work/audit_routes_ws69.py').read_text();body=adversarial[adversarial.index("for rel in ['analyze_resource.py'"):adversarial.index("p=w/'acceptance-gate-audit.json'")]
body=body.replace("b/'work/worldsense-isolated-primary/production-speed/control-1.json'","w/'oracles/legacy-native-route-control.json'")
namespace=dict(w=d,b=b,load=load,results={},ast=ast,copy=copy,json=json);exec(compile(body,'archived adversarial audit','exec'),namespace)
assert namespace['results']==load(d/'acceptance-gate-audit.json')['native_route_audit']and sum(v['malformed_copies_rejected']for v in namespace['results'].values())==8149
# Every bundled external input is byte checked; omitted external dependencies
# retain terminal-local rehash records and must be supplied/revalidated for reuse.
resolution=load(d/'external-input-resolution.json')
for original,info in resolution.items():
 assert Path(original).name!='worldsense-http.py';target=info['bundled']
 if target is None:continue
 if target.startswith('reconstructed-candidate:'):assert hashlib.sha256(current[target.split(':',1)[1]]).hexdigest()==info['sha256']
 else:assert sha(d/target)==info['sha256']
for filename,key in [('native-input-identities-before-dispatch.json','files'),('learned-identities-before.json','external'),('driver-identities-before-dispatch.json','files'),('outer-method-verifier-identities-before-resource.json','files')]:
 for path,digest in load(d/filename)[key].items():assert resolution[path]['sha256']==digest
assert len(load(d/'build-host-dependencies-before-dispatch.json')['files'])==1877
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
cleanup=list(csv.DictReader((d/'post-cleanup-memory-samples.csv').open()));leases=list(csv.DictReader((d/'packed-request-lifecycle.csv').open()));assert len(cleanup)==len(leases)==18
lease_rows={(v['stage'],v['worker'],int(v['repetition']),v['phase']):v for v in leases};cleanup_rows={(v['stage'],v['worker'],int(v['repetition']),v['phase']):v for v in cleanup}
for stage,label,index,phase,item,call in rawcalls:
 t=call['timings'];folder=d/'production-speed'if stage=='production-speed'else d/'functional-smoke'if stage=='functional-correctness-only'else d
 assert load(folder/(label+'.json'))['request']==load(folder/'cases.json')
 cohort=next(v for v in load(folder/'cohort-manifest.json')if v['label']==label);environment=load((d/'production-speed/environment.json')if stage=='production-speed'else d/'environment.json');enabled=label=='candidate-1'
 assert cohort['client_exit']==0 and cohort['library_sha256']==environment['libraries']['on'if enabled else'off']
 assert cohort['serving_sha256']==(load(d/'correctness-source-manifest.json')['runtime/metal_serving.py']if enabled or stage!='production-speed'else environment['control_serving_sha256'])
 if'packed_request_weight_residency'not in t:continue
 key=(stage,label,index,phase);row=lease_rows[key];assert row['packed_request_eligible']==str(t['packed_request_eligible'])
 for name,value in t['packed_request_weight_residency'].items():assert row[name]==str(value)
 for name,value in t['memory_after_packed_request_cleanup'].items():assert cleanup_rows[key][name]==str(value)
scope=load(d/'export-scope.json');assert scope['post_cleanup_rows']==18 and scope['memory_rows']==22 and scope['calls']==32
assert load(d/'restoration-verification.json')['benchmark_valid_request_http404']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_files=len(manifest),source_files=1721,source_overlays=len(recon['overlay_files']),legacy_files=1706,legacy_overlays=len(legacy_recon['overlay_files']),native_CTest_pass=54,original186_complete_projection_streams_and_new2records520words_exact=True,original_programs_and_raw_policy_preserved=True,original_guard_and_all_five_metric_gate_ASTs_inverse_exact=True,original_five_full_learned_readers_and_two_leased_readers_states_tokens_vision_lifecycle_exact=True,actual4B_owners=201,actual4B_allocations=402,actual4B_allocation_bytes=actual,offline_malformed_copies_rejected=8149,all32_pipeline_goldens_native_routes_CSV_resource_speed_and_rejection_recomputed=True,calls=32,requests=16,memory_rows=22,post_cleanup_rows=18,external_input_resolutions=len(resolution),decision='valid-reject',promotion=False)
(b/'work/worldsense-opt/ws69-archive-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
