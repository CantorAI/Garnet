"""Independent WS41 archive coverage/output/metric audit; CPU only."""
from pathlib import Path
import csv,hashlib,json,statistics,subprocess,sys
r=Path('/Users/shawn/CantorAI/Garnet')
d=Path(sys.argv[1])if len(sys.argv)>1 else r/'results/20261017/worldsense-m5-primary-narrow-vocab'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=load(d/'file-manifest.json')
assert set(manifest)=={str(p.relative_to(d))for p in d.rglob('*')if p.is_file()and p.name!='file-manifest.json'}
assert all(sha(d/rel)==h for rel,h in manifest.items())
reconstruction=load(d/'source-reconstruction.json');closure=load(d/reconstruction['source_manifest']);base=reconstruction['base_commit']
refs=[base+':'+rel for rel in closure]
raw=subprocess.check_output(['git','-C',str(r),'cat-file','--batch'],input=('\n'.join(refs)+'\n').encode())
offset=0
for rel,digest in closure.items():
    end=raw.index(b'\n',offset);header=raw[offset:end].split();offset=end+1
    if header[-1]==b'missing':original=None
    else:
        assert header[1]==b'blob';size=int(header[2]);original=raw[offset:offset+size];offset+=size+1
    overlay=d/'source'/rel
    actual=overlay.read_bytes()if overlay.exists()else original
    assert actual is not None and hashlib.sha256(actual).hexdigest()==digest,rel
assert offset==len(raw)
assert closure==load(d/'correctness-source-manifest.json')
assert sorted(reconstruction['overlay_files'])==sorted(load(d/'implementation-audit.json')['changed_files'])
assert len(reconstruction['overlay_files'])==7
assert all(load(d/'api-correctness-result.json').values())
assert all(v==0 for v in load(d/'native-results.json').values())
assert all(v['returncode']==0 for v in load(d/'resident-results.json'))
assert (d/'native-off-projection-words.bin').read_bytes()==(d/'native-on-projection-words.bin').read_bytes()
assert load(d/'native-word-comparison.json')['records']==186
corpus=load(d/'corpus-GPU-bits.json');t=corpus['totals']
assert t['packed_matrices']==397 and t['raw_fallback_matrices']==49 and t['raw_vocabulary_owners']==1
assert t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==5610143744
assert t['retained_primary_and_small_raw_bytes']==9182203392
assert len(corpus['matrices'])==446 and all(m['owner_accounting_restored']for m in corpus['matrices'])
excluded=[m for m in corpus['matrices']if not m['semantic_primary_allowed']]
assert len(excluded)==1 and excluded[0]['model']=='Qwen3-1.7B'and excluded[0]['tensor']=='model.embed_tokens.weight'and not excluded[0]['packed']
assert excluded[0]['GPU_original_FP32_words_checked']==0,'Raw vocabulary is not primary-reader-covered'
off,on=(load(d/('api-state-progress-'+mode+'.json'))['records']for mode in ['off','on'])
assert len(off)==len(on)==5
for a,b in zip(off,on):
    assert a['model_id']==b['model_id']and a['states']==b['states']
    assert all(a['call'][k]==b['call'][k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])

gold=json.loads(subprocess.check_output(['git','-C',str(r),'show',base+':results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json']))
refs={(i['observation_id'],phase):(i,c)for i in gold['items']if i['policy']=='candidate'for phase,c in i['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'}
query_refs={q['query']:q for q in gold['search_plans']if q['policy']=='candidate'}
storage=load(d/'prospective-storage-counters.json')['native_owner_tuples']
expected=[];parents=[];stages={};outcomes={};raw_calls={};raw_requests={}
for stage,folder in [('resource',d),('speed',d/'speed'),('production-speed',d/'production-speed'),('confirmation',d/'confirmation')]:
    data={};calls={}
    for p in sorted(folder.glob('*.json')):
        if not p.name.startswith(('control-','candidate-')):continue
        dataset=load(p)
        if not isinstance(dataset,dict)or'items'not in dataset or'search_plans'not in dataset:continue
        label=dataset['label'];data[label]=dataset;calls[label]=[]
        assert dataset['valid']and dataset['assignments_restored']and dataset['request']==load(folder/'cases.json')
        env=load((d if stage=='confirmation'else folder)/'environment.json');worker=next(m for m in load(folder/'cohort-manifest.json')if m['label']==label)
        enabled=label.startswith('candidate')
        assert worker['library_sha256']==env['libraries']['on'if enabled else'off']
        assert worker['serving_sha256']==(env['control_serving_sha256']if stage=='production-speed'and not enabled else closure['runtime/metal_serving.py'])
        for category,items in [('frame',dataset['items']),('query',dataset['search_plans'])]:
            for index,item in enumerate(items):
                artifact=str(p.relative_to(d));parents.append((artifact,category,str(index)));raw_requests[(artifact,category,str(index))]=item
                responses=item['result'].get('model_response',{})if category=='frame'else{'query_plan':item['result']['benchmark_model_response']}
                for phase,c in responses.items():
                    if not isinstance(c,dict)or'model_id'not in c:continue
                    assert c['status']=='ok'
                    if category=='frame':
                        ri,ref=refs[(item['observation_id'],phase)]
                        assert all(item[k]==ri[k]for k in ['image_sha256','skill_sha256'])
                        stable=lambda i:{k:v for k,v in i['result'].items()if k not in ['model_response','model_response_json']}
                        assert stable(item)==stable(ri)
                    else:
                        ri=query_refs[item['query']];ref=ri['result']['benchmark_model_response']
                        stable=lambda i:{k:v for k,v in i['result'].items()if k!='benchmark_model_response'}
                        assert stable(item)==stable(ri)
                    assert all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
                    assert c['timings']['input_tokens']==ref['timings']['input_tokens']
                    tt=c['timings'];caps=['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','primary_raw_narrow_vocab']
                    assert all((cap in c['backend'])==enabled for cap in caps)
                    source=env['control_source']if stage=='production-speed'and not enabled else'/Users/shawn/CantorAI/out/research/worldsense-primary-narrow-vocab/source'
                    assert tt['serving_source']==str(Path(source)/'runtime/metal_serving.py')
                    if stage=='production-speed'and not enabled:
                        assert 'dense_prefix_cache'in c['backend']and all(k not in tt for k in ['primary_storage_after','primary_encoded_delta','memory_after_decode'])
                    else:
                        assert tt['lazy_default_retired_bytes']==tt['lazy_default_rehydrated_bytes']==0 and tt['memory_after_decode']['task_info_status']==0
                        owner=tt['primary_storage_after'];delta=tt['primary_encoded_delta']
                        if not enabled:assert all(v==0 for v in owner.values())and all(v==0 for v in delta.values())
                        else:
                            assert [owner['original_bytes'],owner['payload_bytes']+owner['metadata_bytes'],owner['metadata_bytes']]==storage[str(owner['active_weights'])]
                            assert 0<owner['allocated_bytes']<owner['original_bytes']and delta['prefill_encoded']>0
                            assert (delta['head_encoded']==0)if c['model_id']=='Qwen3-1.7B'else(delta['head_encoded']>0)
                            assert delta['vector_decode_encoded']==delta['decode_encoded']>0 and delta['vector_head_encoded']==delta['head_encoded']
                    identity=(artifact,category,str(index),phase,c['model_id']);calls[label].append((item,c));expected.append(identity);raw_calls[identity]=(item,c,stage)
    if not data:continue
    stages[stage]=sum(len(v)for v in calls.values())
    assert load(d/('dependencies-after-'+stage+'.json'))['all_aliases_and_file_bytes_unchanged']
    if stage=='resource':
        report=load(d/'resource-analysis.json');sampled={label:{field:max(c['timings']['memory_after_decode'][field]for _,c in values)for field in ['metal_allocated_bytes','task_physical_footprint_bytes']}for label,values in calls.items()}
        assert stages[stage]==8 and report['sampled_maximum_bytes']==sampled
        metal=1-sampled['candidate-1']['metal_allocated_bytes']/sampled['control-1']['metal_allocated_bytes'];foot=1-sampled['candidate-1']['task_physical_footprint_bytes']/sampled['control-1']['task_physical_footprint_bytes']
        assert report['passed']==(metal>=.10 and foot>=.05);outcomes[stage]=report['passed'];continue
    metrics={};walls={}
    for label,values in calls.items():
        metrics[label]={};walls[label]=statistics.median(i['native_call_seconds']for i in data[label]['items']if not i['warmup'])
        for model in ['Qwen3-1.7B','Qwen3.5-4B']:
            cc=[c for i,c in values if not i['warmup']and c['model_id']==model]
            assert len(cc)==(3 if stage=='confirmation'else 4)
            metrics[label][model]=dict(samples=len(cc),ttft=statistics.median(c['ttft_seconds']for c in cc),total=statistics.median(c['total_seconds']for c in cc),decode_tps=statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in cc))
    pairs=[('control-full-'+str(i),'candidate-full-'+str(i))for i in [1,2]]if stage=='confirmation'else[('control-1','candidate-1')]
    for control,candidate in pairs:
        report=load(folder/('analysis-'+control+'-'+candidate+'.json'if stage=='confirmation'else'analysis.json'))
        assert report['metrics']=={control:metrics[control],candidate:metrics[candidate]}
        passed=1-walls[candidate]/walls[control]>=-.02
        for model in metrics[control]:
            a,b=metrics[control][model],metrics[candidate][model]
            passed=passed and b['ttft']/a['ttft']-1<=.02 and b['total']/a['total']-1<=.02 and b['decode_tps']/a['decode_tps']-1>=-.02
        if stage!='production-speed':
            memory={label:max(c['timings']['memory_after_decode']['metal_allocated_bytes']for _,c in calls[label])for label in [control,candidate]}
            passed=passed and 1-memory[candidate]/memory[control]>=.10
        else:assert report['sampled_max_post_request_metal_bytes'][control]is None and report['sampled_metal_reduction']is None
        assert report['passed']==passed
        outcomes[stage+(':'+control if stage=='confirmation'else'')]=passed
for name in ['calls.csv','execution-routes.csv']:
    with(d/name).open()as f:rows=list(csv.DictReader(f))
    assert sorted(tuple(row[k]for k in ['artifact','category','sample_index','phase','model_id'])for row in rows)==sorted(expected)
    assert all(row['artifact_sha256']==sha(d/row['artifact'])for row in rows)
    for row in rows:
        identity=tuple(row[k]for k in ['artifact','category','sample_index','phase','model_id']);item,c,stage=raw_calls[identity]
        assert row['warmup']==str(item['warmup'])and row['repetition']==str(item['repetition'])and row['stage']==stage
        if name=='calls.csv':
            assert all(float(row[k])==c[k]for k in ['ttft_seconds','total_seconds','generated_tokens'])
            assert float(row['derived_decode_tokens_per_second'])==(c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])
            assert row['status']==c['status']and row['input_prompt_sha256']==c['input_prompt_sha256']and int(row['input_tokens'])==c['timings']['input_tokens']
            assert row['eligible_for_speed']==str(stage!='resource'and not item['warmup'])
        else:
            assert row['backend']==c['backend']and row['serving_source']==c['timings']['serving_source']
            for field in ['primary_storage_after','primary_encoded_delta','memory_after_decode']:
                assert row[field+'_available']==str(field in c['timings'])
                assert all(row[field+'_'+k]==str(v)for k,v in c['timings'].get(field,{}).items())
with(d/'skill-requests.csv').open()as f:rows=list(csv.DictReader(f))
assert sorted(tuple(row[k]for k in ['artifact','category','sample_index'])for row in rows)==sorted(parents)
for row in rows:
    item=raw_requests[tuple(row[k]for k in ['artifact','category','sample_index'])]
    assert row['artifact_sha256']==sha(d/row['artifact'])and row['warmup']==str(item['warmup'])
    assert row['native_call_seconds']==str(item.get('native_call_seconds',''))
assert load(d/'export-scope.json')['stage_calls']==stages
decision=load(d/'decision.json');assert decision['stage_outcomes']==outcomes
assert decision['prototype_confirmed']==(len(expected)==96 and all(outcomes.values())and len(outcomes)==5)
assert not decision['promote'],'Canonical production acceptance is a separate required archive'
print(json.dumps(dict(files=len(manifest),source_files=len(closure),source_overlay=7,benchmark_calls=len(expected),native_requests=len(parents),stage_outcomes=outcomes,exact_outputs=True,decision_consistent=True,scope='Independent artifact/source/call/parent-request/output/metric audit; original native tests retain their recorded numerical scope.')))
