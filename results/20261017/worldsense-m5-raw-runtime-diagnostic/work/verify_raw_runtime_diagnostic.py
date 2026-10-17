from pathlib import Path
import json,hashlib,subprocess,csv,statistics,re
r=Path('/Users/shawn/CantorAI/Garnet');d=r/'results/20261017/worldsense-m5-raw-runtime-diagnostic';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=load(d/'file-manifest.json');assert set(manifest)=={str(p.relative_to(d))for p in d.rglob('*')if p.is_file()and p.name!='file-manifest.json'}and all(sha(d/k)==h for k,h in manifest.items());p=load(d/'protocol.json');assert p['experiment']=='WS-M5-55'and p['kind']=='diagnostic_only'and not p['promote']and p['calls']==40
reconstruction=load(d/'source-reconstruction.json');source_bytes={}
for role,info in reconstruction.items():
 m=load(d/info['manifest']);raw=subprocess.check_output(['git','-C',str(r),'cat-file','--batch'],input=('\n'.join(info['base_commit']+':'+k for k in m)+'\n').encode());at=0;source_bytes[role]={}
 for rel,h in m.items():
  end=raw.index(b'\n',at);header=raw[at:end].split();at=end+1
  if header[-1]==b'missing':original=None
  else:size=int(header[2]);original=raw[at:at+size];at+=size+1
  overlay=d/'source'/role/rel;data=overlay.read_bytes()if overlay.exists()else original;assert data is not None and hashlib.sha256(data).hexdigest()==h;source_bytes[role][rel]=data
 assert at==len(raw)and len(m)==info['source_files'];assert {str(q.relative_to(d/'source'/role))for q in(d/'source'/role).rglob('*')if q.is_file()}==set(info['overlay_files'])
assert len(source_bytes['old'])==1706 and len(source_bytes['newraw'])==1718
ref='59a882e9765d4c1f61cd6bc117aa620dd06abd16'
for local,gitrel in [('starting-source-manifest.json','correctness-source-manifest.json'),('control-source-before-dispatch.json','production-speed/control-source-before-dispatch.json')]:
 assert load(d/local)==json.loads(subprocess.check_output(['git','-C',str(r),'show',ref+':results/20261017/worldsense-m5-isolated-primary/'+gitrel]))
flags=load(d/'build-flag-comparison.json');a,b=flags['old'],flags['newraw'];assert all(a[k]==b[k]for k in a.keys()&b.keys())and not(a.keys()-b.keys())and all(b[k]=='OFF'for k in b.keys()-a.keys())
x=(d/'selected-constructor-old.txt').read_text();y=(d/'selected-constructor-newraw.txt').read_text();assert re.sub(r'\s+',' ',x).strip()==re.sub(r'\s+',' ',y).strip();assert re.findall(r'R"([A-Za-z0-9_]*)\(([\s\S]*?)\)\1"',x)==re.findall(r'R"([A-Za-z0-9_]*)\(([\s\S]*?)\)\1"',y);assert source_bytes['old']['src/backends/metal/body_shape_shader.h']==source_bytes['newraw']['src/backends/metal/body_shape_shader.h']
provenance=load(d/'reused-correctness-provenance.json');env=load(d/'environment.json');arch54=json.loads(subprocess.check_output(['git','-C',str(r),'show',ref+':results/20261017/worldsense-m5-isolated-primary/environment.json']));assert env['libraries']['on']==provenance['raw54_library_sha256']==arch54['libraries']['off'];assert env['libraries']['off']=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
cohort=load(d/'cohort-manifest.json');assert [v['label']for v in cohort]==p['schedule']==['control-1','candidate-1','candidate-2','control-2'];gold=json.loads(subprocess.check_output(['git','-C',str(r),'show',reconstruction['old']['base_commit']+':results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json']));refs={(i['observation_id'],phase):(i,c)for i in gold['items']if i['policy']=='candidate'for phase,c in i['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'};expected={};parents={};metrics={};walls={}
for worker in cohort:
 label=worker['label'];new=label.startswith('candidate');f=label+'.json';data=load(d/f);assert data['valid']and data['assignments_restored']and data['executor']=='XLang3'and data['request']==load(d/'cases.json')and len(data['items'])==5 and not data['search_plans'];assert [(i['repetition'],i['warmup'],i['policy'])for i in data['items']]==[(i,i==0,'default')for i in range(5)];assert worker['client_exit']==0 and worker['library_sha256']==env['libraries']['on'if new else'off'];warm={m:[]for m in ['Qwen3.5-4B','Qwen3-1.7B']}
 assert worker['serving_sha256']==hashlib.sha256(source_bytes['newraw'if new else'old']['runtime/metal_serving.py']).hexdigest()
 for idx,item in enumerate(data['items']):
  parents[(f,str(idx))]=item
  for phase,c in item['result']['model_response'].items():
   if not isinstance(c,dict)or'model_id'not in c:continue
   ri,refc=refs[(item['observation_id'],phase)];assert c['status']=='ok'and all(c[k]==refc[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration']);assert c['timings']['input_tokens']==refc['timings']['input_tokens']and all(item[k]==ri[k]for k in ['image_sha256','skill_sha256']);stable=lambda i:{k:v for k,v in i['result'].items()if k not in ['model_response','model_response_json']};assert stable(item)==stable(ri)
   t=c['timings'];assert t['serving_source']==str(Path(env['new_source'if new else'old_source'])/'runtime/metal_serving.py');assert 'dense_prefix_cache'in c['backend'];assert not any(cap in c['backend']for cap in ['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','primary_vector_prefill','primary_prefill_isolated_library','primary_duo_prefill','primary_minimal_prefill_library','direct_float_weight_upload','primary_vector_prefill_k32_only','cached_matrix_metadata','primary_full_isolated_library','lazy_resident_defaults','wide_primary_storage','wide_checkpoint_storage'])
   if new:
    assert all(v==0 for key in ['primary_storage_before','primary_storage_after','primary_encoded_delta']for v in t[key].values());assert all(t[k]==0 for k in ['lazy_default_retired_bytes','lazy_default_planned_slots','lazy_default_rehydrated_bytes'])and t['memory_after_decode']['task_info_status']==0
   else:assert all(k not in t for k in ['primary_storage_before','primary_storage_after','primary_encoded_delta','memory_after_decode'])
   identity=(f,str(idx),phase,c['model_id']);assert identity not in expected;expected[identity]=(item,c)
   if not item['warmup']:warm[c['model_id']].append(c)
 assert load(d/('validation-'+label+'.json'))['data_sha256']==sha(d/f);metrics[label]={}
 for model,cs in warm.items():
  assert len(cs)==4;metrics[label][model]={'samples':4,'ttft':statistics.median(c['ttft_seconds']for c in cs),'total':statistics.median(c['total_seconds']for c in cs),'decode_tps':statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in cs)}
 walls[label]=statistics.median(i['native_call_seconds']for i in data['items']if not i['warmup'])
assert len(expected)==40 and len(parents)==20
indicators=[]
for i in [1,2]:
 a,b='control-'+str(i),'candidate-'+str(i);report=load(d/('analysis-'+a+'-'+b+'.json'));assert report['metrics']=={a:metrics[a],b:metrics[b]};assert report['diagnostic_only']and not report['promote'];wall_gain=1-walls[b]/walls[a];assert report['native_skill_wall_gain']==wall_gain;preserved=wall_gain>=-.02
 for model in metrics[a]:
  aa,bb=metrics[a][model],metrics[b][model];compare=next(v for v in report['comparisons']if v['model']==model);changes={'ttft_change':bb['ttft']/aa['ttft']-1,'total_change':bb['total']/aa['total']-1,'decode_gain':bb['decode_tps']/aa['decode_tps']-1};assert all(compare[k]==v for k,v in changes.items());preserved=preserved and changes['ttft_change']<=.02 and changes['total_change']<=.02 and changes['decode_gain']>=-.02
 assert report['preserved_within_2percent']==preserved and report['sampled_max_post_request_metal_bytes'][a]is None and report['sampled_metal_reduction']is None;indicators.append(preserved)
for filename in ['calls.csv','execution-routes.csv']:
 rows=list(csv.DictReader((d/filename).open()));assert {tuple(row[k]for k in ['artifact','sample_index','phase','model_id'])for row in rows}==set(expected)and len(rows)==40
 for row in rows:
  item,c=expected[tuple(row[k]for k in ['artifact','sample_index','phase','model_id'])];assert row['artifact_sha256']==sha(d/row['artifact'])and row['warmup']==str(item['warmup'])and row['repetition']==str(item['repetition'])
  if filename=='calls.csv':
   assert all(float(row[k])==c[k]for k in ['ttft_seconds','total_seconds','generated_tokens']);assert float(row['derived_decode_tokens_per_second'])==(c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds']);assert float(row['post_decode_residual_seconds'])==c['total_seconds']-c['ttft_seconds']-c['timings']['decode_seconds'];assert row['eligible_for_diagnostic_warm']==str(not item['warmup'])
  else:
   assert row['backend']==c['backend']and row['serving_source']==c['timings']['serving_source']
   for k in ['primary_storage_before','primary_storage_after','primary_encoded_delta','memory_after_decode']:assert row[k+'_available']==str(k in c['timings'])and all(row[k+'_'+kk]==str(vv)for kk,vv in c['timings'].get(k,{}).items())
rows=list(csv.DictReader((d/'skill-requests.csv').open()));assert len(rows)==20 and {tuple(row[k]for k in ['artifact','sample_index'])for row in rows}==set(parents)
for row in rows:assert float(row['native_call_seconds'])==parents[(row['artifact'],row['sample_index'])]['native_call_seconds']and row['artifact_sha256']==sha(d/row['artifact'])
post=load(d/'posthoc-observations.json')
for worker in cohort:
 label=worker['label'];size=int(re.search(r'page size of (\d+) bytes',worker['vm_stat_before']).group(1));parse=lambda text:{k:int(v)for k,v in re.findall(r'^([^:\n]+):\s+(\d+)\.',text,re.M)};before,after=parse(worker['vm_stat_before']),parse(worker['vm_stat_after']);record=post['system_observations'][label]
 assert record['global_swapin_bytes']==(after['Swapins']-before['Swapins'])*size and record['global_swapout_bytes']==(after['Swapouts']-before['Swapouts'])*size
 assert post['workers'][label]['Skill_wall']==walls[label]
 for model in metrics[label]:assert all(post['workers'][label][model][k]==metrics[label][model][k]for k in ['ttft','total','decode_tps'])
terminal=load(d/'terminal.json');assert terminal['controller_session']==50693 and terminal['terminal_exit_code']==0 and not terminal['promote']and terminal['pair_2percent_indicators']==indicators==[False,True];assert load(d/'dependencies-after-diagnostic.json')['verified_files']==len(load(d/'dependencies-before-dispatch.json')['files'])==5298 and load(d/'dependencies-after-diagnostic.json')['all_aliases_and_bytes_unchanged'];assert load(d/'restoration-verification.json')['benchmark_valid_request_http404']
print(json.dumps({'SHA_files':len(manifest),'source_files':{k:len(v)for k,v in source_bytes.items()},'calls':40,'parents':20,'independent_pair_indicators':indicators,'all_original_outputs_native_valid':True,'metric_CSV_source_identity_diagnostic_decision_consistent':True,'promote':False,'scope':'Independent artifact/source/call/output/metric/system-counter audit; not performance confirmation, live rerun or original-build environment proof.'}))
