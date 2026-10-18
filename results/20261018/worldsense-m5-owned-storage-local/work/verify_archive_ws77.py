"""Independent canonical source reconstruction, native words, original goldens and decision."""
from pathlib import Path
import json,hashlib,subprocess,csv,statistics,ast,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');repo=Path('/Users/shawn/CantorAI/Garnet');d=repo/'results/20261018/worldsense-m5-owned-storage-local';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
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
current,recon=reconstruct('');legacy,legacy_recon=reconstruct('legacy-');assert len(current)==1722 and len(legacy)==1706
seed_ref='e15c3f4325682098bd4c7b4a17da62254e1e36c5';seed_prefix='results/20261018/worldsense-m5-minimal-selective-defaults/'
seed,seed_recon=[json.loads(v)for v in batch([seed_ref+':'+seed_prefix+'correctness-source-manifest.json',seed_ref+':'+seed_prefix+'source-reconstruction.json'])]
assert seed==load(d/'seed-source-manifest.json')and len(seed)==1721
seed_source=dict(zip(seed,batch([(seed_ref+':'+seed_prefix+'source/'+rel)if rel in seed_recon['overlay_files']else seed_recon['base_commit']+':'+rel for rel in seed])))
assert all(hashlib.sha256(seed_source[k]).hexdigest()==v for k,v in seed.items())
changed=['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/owned_storage_local_test.cpp']
assert sorted(k for k in current if current[k]!=seed_source.get(k))==changed
assert set(current)-set(seed_source)=={changed[-1]} and not set(seed_source)-set(current)
operations=load(d/'source-transformations.json')
for rel in {op['path']for op in operations}:
 text=current[rel].decode()
 for op in reversed([op for op in operations if op['path']==rel]):
  if op['append']:assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text.encode()==seed_source[rel]
assert load(d/'source-inverse-proof.json')['changed_or_added']==changed

assert len(recon['overlay_files'])==17 and len(legacy_recon['overlay_files'])==9
roles=['off','primary','on','fault','compat56','compat56-fault','control','raw-borrow'];assert load(d/'native-results.json')=={role:0 for role in roles}
import re,struct,copy
for role in roles:
 log=(d/('native-'+role+'.log')).read_text();assert sorted(int(v)for v in re.findall(r'\d+/10\s+Test\s+#\s*(\d+):[^\n]*Passed',log))==list(range(1,11))and re.search(r'100% tests passed(?:, 0 tests failed)? out of 10',log)
 cache=(d/'build-identities'/role/'CMakeCache.txt').read_text();assert ('GARNET_METAL_PRIMARY_CORE_ISOLATED_LIBRARY:BOOL=ON'in cache) is False
 assert ('GARNET_METAL_REQUEST_SCOPE_WEIGHT_RESIDENCY:BOOL=ON'in cache)==role.startswith('compat56')
 assert ('GARNET_METAL_REQUEST_RESIDENCY_TEST_FAULTS:BOOL=ON'in cache)==(role=='compat56-fault')
 flags=dict(re.findall(r'^GARNET_METAL_([A-Z0-9_]+):BOOL=(ON|OFF)$',cache,re.M));profile={'control':'on','raw-borrow':'off'}.get(role,role);selective=profile in ['primary','on','fault'];compat=profile.startswith('compat56');vector=profile in ['on','fault']or compat
 assert flags['OWNED_STORAGE_LOCAL']==('ON'if role in ['primary','on','fault','raw-borrow']else'OFF')
 assert flags['PRIMARY_ONLY_LAZY_DEFAULTS']==('ON'if selective else'OFF')
 for key in ['PRIMARY_BF16_BLOCKS','WIDE_PRIMARY_STORAGE','LAZY_RESIDENT_DEFAULTS']:assert flags[key]==('OFF'if profile=='off'else'ON')
 for key in ['PRIMARY_VECTOR_PREFILL','PRIMARY_PREFILL_ISOLATED_LIBRARY','PRIMARY_FULL_ISOLATED_LIBRARY']:assert flags[key]==('ON'if vector else'OFF')
 for key in ['PRIMARY_VECTOR_PREFILL_K32_ONLY','DIRECT_FLOAT_WEIGHT_UPLOAD','CACHED_MATRIX_METADATA','REQUEST_SCOPE_WEIGHT_RESIDENCY']:assert flags[key]==('ON'if compat else'OFF')
 assert flags['PRIMARY_TEST_FAULTS']==('ON'if role in ['fault','compat56-fault']else'OFF')

raw=(d/'native-off-primary_projection_words.bin').read_bytes();assert raw==batch([seed_ref+':'+seed_prefix+'native-off-primary_projection_words.bin'])[0]
assert all((d/('native-'+role+'-primary_projection_words.bin')).read_bytes()==raw for role in roles)
proof=load(d/'native-word-comparison.json');assert proof['records']==186 and proof['bytes']==len(raw)and proof['sha256']==hashlib.sha256(raw).hexdigest()
reference=None;cases=[(rows,inner,columns,0)for inner in [128,256]for columns in [260,1024]for rows in [1,2,16,32]]+[(rows,128,100352,0)for rows in [1,2]]+[(rows,128,260,1)for rows in [1,2]]
for role in roles:
 data=(d/('native-'+role+'-primary_core_projection_words.bin')).read_bytes()
 if reference is None:reference=data
 assert data==reference;position=0;words=0
 for rows,inner,columns,raw_mode in cases:
  count=rows*columns;assert struct.unpack('<6Q',data[position:position+48])==(rows,inner,columns,raw_mode,int(rows<=2 and columns<100000),count);position+=48
  expected=b''.join(struct.pack('<f',inner*(row+1)/16)*columns for row in range(rows));assert data[position:position+count*4]==expected;position+=count*4;words+=count
 assert position==len(data)and words==432804
 summary=load(d/('native-'+role+'-primary_core_projection_words.bin.summary.json'));assert summary['complete_records']==20 and summary['complete_FP32_words']==432804 and summary['specialized_body_cases']==10 and summary['core_enabled'] is False
 assert [summary[k]for k in ['core_decode_delta','core_prefill_delta','core_head_delta']]==[0,0,0]
proof=load(d/'core-native-word-comparison.json');assert proof['records']==20 and proof['words_per_build']==432804 and proof['bytes']==len(reference)and proof['sha256']==hashlib.sha256(reference).hexdigest()
# Independently recompute all new lifetime stream headers, words and EOF.
reference_lifetime=None
for role in roles:
 data=(d/('native-'+role+'-owned_storage_projection_words.bin')).read_bytes();offset=0;records=words=0
 for raw_mode in [False,True]:
  for mode in range(5):
   shapes=[(rows,columns)for rows in [1,2,16,32,64]for columns in [260,1024]]+[(rows,100352)for rows in [1,2]]
   for rows,columns in shapes:
    count=rows*columns;assert struct.unpack_from('<6Q',data,offset)==(rows,128,columns,int(raw_mode),mode,count);offset+=48
    expected=b''.join(struct.pack('<f',8*(row+1))*columns for row in range(rows));assert data[offset:offset+count*4]==expected;offset+=count*4;records+=1;words+=count
 assert offset==len(data)and records==120 and words==4487160
 if reference_lifetime is None:reference_lifetime=data
 assert reference_lifetime==data
 summary=load(d/('native-'+role+'-owned_storage_projection_words.bin.summary.json'));packed=role not in ['off','raw-borrow']
 assert summary['complete_records']==records and summary['complete_FP32_words']==words and summary['owned_storage_local_enabled']==(role in ['primary','on','fault','raw-borrow'])and summary['packed_cases']==(60 if packed else 0)and summary['raw_cases']==(60 if packed else 120)and summary['invalid_cases_rejected']==72 and summary['prepared_cases']==40 and summary['live_logical_accounting_restored']
host=load(d/'host-codegen-review.json');assert host['selected_local_ownership_removed_both_raw_and_packed']
for role,functions in host['details'].items():
 for kind,record in functions.items():
  p=d/(role+'-'+kind+'.arm64.txt');assert sha(p)==record['disassembly_sha256'];lines=p.read_text().splitlines();owner=next(i for i,line in enumerate(lines)if re.search(r'fmov\s+x\d+, d0',line));last=next(i for i in range(owner+1,len(lines))if '_objc_msgSend'in lines[i]);window=lines[owner:last]
  assert window==record['selection_window'];enabled=role in ['raw-borrow','on'];assert(sum('_objc_retain'in x for x in window),sum('_objc_release'in x for x in window))==((0,0)if enabled else((1,0)if role=='off'else(2,1)))
  assert any('0xf8290108'in x for x in lines[max(0,owner-12):owner]);release=next(i for i in range(last+1,len(lines))if '__shared_weak_count14__release_weakEv'in lines[i]);assert any('0xf8e90108'in x for x in lines[release-10:release])
  if kind=='encode':assert any('_objc_msgSend$endEncoding'in x for x in lines[last:release])
programs={role:{kind:(d/('actual-'+role+'-'+kind+'-program.metal')).read_bytes()for kind in ['main','prefill','core']}for role in roles}
for role in roles:
 previous={'control':'on','raw-borrow':'off'}.get(role,role)
 for kind in ['main','prefill','core']:assert programs[role][kind]==batch([seed_ref+':'+seed_prefix+'actual-'+previous+'-'+kind+'-program.metal'])[0]
full_program=current['src/backends/metal/primary_isolation_shader.h'].decode().split('R"IPRIMARY(',1)[1].split(')IPRIMARY"',1)[0]+current['src/backends/metal/primary_block_shader.h'].decode().split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]+current['src/backends/metal/primary_vector_prefill_shader.h'].decode().split('R"PVEC(',1)[1].split(')PVEC"',1)[0]
for role in ['on','fault','control']:assert programs[role]['main']==programs['off']['main']and programs[role]['prefill']==full_program.encode()==programs['compat56']['prefill']
assert all(not programs[role]['core']for role in roles)
for role,groups in load(d/'actual-program-proof.json')['programs'].items():
 for kind,info in groups.items():assert len(programs[role][kind])==info['bytes']and hashlib.sha256(programs[role][kind]).hexdigest()==info['sha256']
for name,info in load(d/'learned-guard-adaptation-proof.json').items():
 original=batch([seed_ref+':'+seed_prefix+name])[0].decode();value=(d/name).read_text();assert hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']and hashlib.sha256(value.encode()).hexdigest()==info['current_sha256']
 for op in reversed(info['operations']):assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
 assert value.replace('worldsense-owned-storage-local','worldsense-minimal-selective-defaults')==original
# Recheck the complete immutable parent numerical-guard chain.
for child_ref,child_prefix,child_slug,parent_ref,parent_prefix,parent_slug in [
 (seed_ref,seed_prefix,'worldsense-minimal-selective-defaults','4c811ec4e305317efac622177d4623b1cf8c83a4','results/20261018/worldsense-m5-minimal-full-prefill/','worldsense-minimal-full-prefill'),
 ('4c811ec4e305317efac622177d4623b1cf8c83a4','results/20261018/worldsense-m5-minimal-full-prefill/','worldsense-minimal-full-prefill','80787c104fdafe7eb4671cc6c7b390bfcce2fb1f','results/20261018/worldsense-m5-primary-core-isolation/','worldsense-primary-core-isolation'),
 ('80787c104fdafe7eb4671cc6c7b390bfcce2fb1f','results/20261018/worldsense-m5-primary-core-isolation/','worldsense-primary-core-isolation','290f5742ba51c41197f24179cfee4876e96e789a','results/20261017/worldsense-m5-request-weight-residency/','worldsense-request-weight-residency')]:
 parent_proof=json.loads(batch([child_ref+':'+child_prefix+'learned-guard-adaptation-proof.json'])[0])
 for name,info in parent_proof.items():
  value=batch([child_ref+':'+child_prefix+name])[0].decode();original=batch([parent_ref+':'+parent_prefix+name])[0].decode()
  assert hashlib.sha256(value.encode()).hexdigest()==info['current_sha256']and hashlib.sha256(original.encode()).hexdigest()==info['seed_sha256']
  for op in reversed(info['operations']):assert value.count(op['after'])==op['count'];value=value.replace(op['after'],op['before'])
  assert value.replace(child_slug,parent_slug)==original
def canonical_learned_checks():
 terminal=load(d/'learned-controller-terminal.json');assert terminal['success']and terminal['normal_restored']and not terminal['promotion'];journal=load(d/'learned-controller-journal.json');assert len(journal)==18 and all(v['returncode']==0 and v['ended_utc']>=v['started_utc']for v in journal)
 assert all(load(d/'api-correctness-result.json').values());corpus=load(d/'corpus-GPU-bits.json');t=corpus['totals']
 assert t['packed_matrices']==201 and t['raw_small_matrices']==245 and t['HOST_raw_FP32_words_checked']==1724383232 and t['retained_primary_and_small_raw_bytes']==9850930176 and t['original_bytes']==11850481664
 assert t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600 and t['GPU_prefill_tile_FP32_words_checked']=={k:4200857600 for k in ['32x32','64x32','64x64']}
 selection=load(d/'frozen-native-guards/matrix-selection.json')['matrices'];assert len(corpus['matrices'])==446 and {(v['model'],v['tensor'],v['rows'],v['inner'])for v in selection}=={(v['model'],v['tensor'],v['rows'],v['inner'])for v in corpus['matrices']}
 for row in corpus['matrices']:
  assert row['owner_accounting_restored'];count=row['rows']*row['inner']
  if row['packed']:assert row['GPU_original_FP32_words_checked']==row['GPU_vector_original_FP32_words_checked']==count and row['GPU_prefill_tile_FP32_words_checked']=={k:count for k in ['32x32','64x32','64x64']}
  else:assert row['HOST_raw_FP32_words_checked']==count
 assert load(d/'direct-upload-corpus.json')['totals']==dict(weights=546,FP32_words_checked=334609152)
 resident=load(d/'resident-results.json');assert len(resident)==8 and {v['mode']for v in resident}=={'off','primary','on','compat56'}and all(v['returncode']==0 for v in resident);private_steps=0
 for row in resident:
  lines=(d/row['log']).read_text().splitlines();steps=[json.loads(v.split(' ',1)[1])for v in lines if v.startswith('private_kv_step ')];assert len(steps)==8 and all(v['exact_output_and_state']for v in steps)and any(v.startswith('private_kv_reference PASS ')for v in lines);private_steps+=len(steps)
 for role in ['off','on','fault','compat56']:assert 'Four-input trained linear generic-resident parity/host-private isolation/inactive/malformed/shared-after-source-close/rebuild PASS '+role in(d/('trained-linear-'+role+'.log')).read_text()
 vlm=load(d/'frozen-native-guards/state-guard.json')['records'][1];prefix={v['prompt_index']:v for v in load(d/'frozen-native-guards/prefix-model-guard.json')['records']};records={role:load(d/('api-state-progress-'+role+'.json'))['records']for role in ['off','primary','on','compat56','control','raw-borrow']};core_routes=[]
 for role,rows in records.items():
  assert len(rows)==5
  for index,row in enumerate(rows):
   reference=vlm if index<2 else prefix[[0,1,0][index-2]];call=row['call'];timings=call['timings'];assert row['states']==reference['states']and call['generated_token_ids']==reference['generated_token_ids'];baseline=records['off'][index]['call'];assert all(call[k]==baseline[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
   profile={'control':'on','raw-borrow':'off'}.get(role,role);enabled=profile!='off';compat=profile=='compat56';isolated=False;vector=profile in ['on','compat56']
   assert ('owned_storage_local'in call['backend'])==(role in ['primary','on','raw-borrow'])
   for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage','lazy_resident_defaults']:assert(cap in call['backend'])==enabled
   for cap in ['primary_vector_prefill','primary_prefill_isolated_library','primary_full_isolated_library']:assert(cap in call['backend'])==vector
   for cap in ['cached_matrix_metadata','request_scope_weight_residency']:assert(cap in call['backend'])==compat
   assert('primary_core_isolated_library'in call['backend'])==isolated
   selective=profile in ['primary','on'];retirement=enabled and (not selective or index<2)
   assert ('primary_only_lazy_defaults'in call['backend'])==selective
   assert timings['lazy_default_rehydrated_bytes']==0 and(timings['lazy_default_retired_bytes']>0)==retirement and(timings['lazy_default_planned_slots']>0)==retirement
   stats=timings['primary_storage_after'];before=timings['primary_storage_before'];delta=timings['primary_encoded_delta'];derived={}
   for key,oldkey in [('core_library_decode_encoded','decode_encoded'),('core_library_prefill_encoded','prefill_encoded'),('core_library_head_encoded','head_encoded')]:
    assert type(before[key])is int and type(stats[key])is int and 0<=before[key]<=stats[key];derived[oldkey]=stats[key]-before[key];assert derived[oldkey]==(delta[oldkey]if isolated else 0)
    if not isolated:assert before[key]==stats[key]==0
   core_routes.append(dict(role=role,index=index,model=call['model_id'],core_deltas=derived))
   if enabled and index<2:
    assert stats['active_weights']==201 and stats['original_bytes']==8401715200 and stats['payload_bytes']+stats['metadata_bytes']==6402163712 and stats['metadata_bytes']==32819200
    assert all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])and delta['vector_decode_encoded']==delta['decode_encoded']and delta['vector_head_encoded']==delta['head_encoded']
   else:assert all(v==0 for v in delta.values())and all(v==0 for v in derived.values())
   if compat:
    q=timings['request_weight_residency'];assert q['calls']==q['ends']and q['active_bytes']==q['global_used_bytes']==0 and not q['scope_open']and q['limit_bytes']==4294967296
    if index<2:assert q['reason']==7 and q['admitted']==0 and q['fallback']==q['calls']
    else:assert q['reason']==1 and q['last_count']==197 and q['last_bytes']==3440902144 and q['coverage_fallbacks']==0
 for role in ['off','on','compat56']:assert load(d/('vision-grid-'+role+'.json'))==load(d/'frozen-native-guards/vision-grid-original.json')
 assert load(d/'request-session-guard-compat56.json')['all_cases_pass']and load(d/'request-full-session-guard-compat56.json')['real197masters_global4GiB_contention_close_readmission_reset']
 assert load(d/'native-independent-audit.json')['native_tests_pass']==80

canonical_learned_checks()
assert load(d/'native-independent-audit.json')['observed_exit_code']==0
assert len(load(d/'flag-dependency-proof.json'))==18 and load(d/'admission-complete.json')['all_expected_config_rejections']
for name,info in load(d/'flag-dependency-proof.json').items():assert info['returncode']!=0 and info['expected_diagnostic']in' '.join((d/('invalid-'+name+'-config.log')).read_text().split())
metric_ref='b600cf8de6e99e11d35f6727bc3c3cacfc1d5418';metric_prefix='results/20261018/worldsense-m5-primary-simd-metadata/'
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,node):return None if node.name=='native_route'else self.generic_visit(node)
for rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
 original=batch([metric_ref+':'+metric_prefix+rel])[0];assert original is not None
 assert ast.dump(Normalize().visit(ast.parse(original)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse((d/rel).read_text().replace('worldsense-owned-storage-local','worldsense-primary-simd-metadata'))),include_attributes=False),rel
# Replay the original offline adversarial audit in a separate namespace, only
# against archived calls/oracle. No file writes or new model calls are executed.
adversarial=(d/'work/audit_routes_ws77.py').read_text();body=adversarial[adversarial.index("for rel in ['analyze_resource.py'"):adversarial.index("p=w/'acceptance-gate-audit.json'")]
body=body.replace("b/'work/worldsense-isolated-primary/production-speed/control-1.json'","w/'oracles/legacy-native-route-control.json'")
namespace=dict(w=d,b=b,load=load,results={},ast=ast,copy=copy,json=json);exec(compile(body,'archived adversarial audit','exec'),namespace)
assert namespace['results']==load(d/'acceptance-gate-audit.json')['native_route_audit']and sum(v['malformed_copies_rejected']for v in namespace['results'].values())==22867
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
for path,digest in load(d/'host-codegen-review.json')['inputs'].items():assert resolution[path]['sha256']==digest
assert len(load(d/'build-host-dependencies-before-dispatch.json')['files'])>1800
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
core_rows=list(csv.DictReader((d/'core-dispatch-deltas.csv').open()));assert len(core_rows)==32
lookup={(v['stage'],v['worker'],int(v['repetition']),v['phase']):v for v in core_rows};available=0;positive_vlm=0
for stage,label,index,phase,item,call in rawcalls:
 t=call['timings'];folder=d/'production-speed'if stage=='production-speed'else d/'functional-smoke'if stage=='functional-correctness-only'else d
 assert load(folder/(label+'.json'))['request']==load(folder/'cases.json')
 cohort=next(v for v in load(folder/'cohort-manifest.json')if v['label']==label);environment=load((d/'production-speed/environment.json')if stage=='production-speed'else d/'environment.json');enabled=label=='candidate-1'
 assert cohort['client_exit']==0 and cohort['library_sha256']==environment['libraries']['on'if enabled else'off']
 assert cohort['serving_sha256']==(load(d/'correctness-source-manifest.json')['runtime/metal_serving.py']if enabled or stage!='production-speed'else environment['control_serving_sha256'])
 row=lookup[(stage,label,index,phase)];present='primary_storage_before'in t;assert row['counter_telemetry_available']==str(present)
 if present:available+=1
 for key,oldkey in [('core_library_decode_encoded','decode_encoded'),('core_library_prefill_encoded','prefill_encoded'),('core_library_head_encoded','head_encoded')]:
  if not present:assert row[key]=='';continue
  actual=t['primary_storage_after'][key]-t['primary_storage_before'][key];assert int(row[key])==actual==0
 if present and enabled and call['model_id']=='Qwen3.5-4B':
  positive_vlm+=1
  assert all(t['primary_encoded_delta'][key]>0 for key in ['decode_encoded','prefill_encoded','head_encoded'])
assert available==22 and positive_vlm==9
scope=load(d/'export-scope.json');assert scope['core_delta_rows']==32 and scope['counter_available_rows']==22 and scope['memory_rows']==22 and scope['calls']==32
assert load(d/'restoration-verification.json')['benchmark_valid_request_http404']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_files=len(manifest),source_files=1722,source_overlays=len(recon['overlay_files']),legacy_files=1706,legacy_overlays=len(legacy_recon['overlay_files']),native_CTest_pass=80,original186_complete_projection_streams_and_new20records432804words_exact=True,new120records4487160words_each_eight_roles_exact=True,actual_HOST_selection_ownership_removed_with_owner_kept=True,exact_four_path_inverse_to74=True,original_programs_and_runtime_shader_math_CUDA_engine_test_bytes_preserved=True,original_two_program_FULL_partition_exact_and_CORE_getter_empty=True,original_guard77_to74_to71_to70_to56_and_all_five_metric_gate_ASTs_inverse_exact=True,original_five_full_learned_readers_in_actual_candidate_states_tokens_vision_lifecycle_exact=True,offline_malformed_copies_rejected=22867,all32_pipeline_goldens_native_routes_CSV_resource_speed_and_rejection_recomputed=True,calls=32,requests=16,memory_rows=22,core_delta_rows=32,core_counter_available_rows=22,actual_candidate_VLM_route_records=9,positive_CORE_VLM_calls=0,external_input_resolutions=len(resolution),decision='valid-reject',promotion=False)
(b/'work/worldsense-opt/ws77-archive-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
