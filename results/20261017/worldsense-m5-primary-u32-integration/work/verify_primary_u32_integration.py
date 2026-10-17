"""CPU-only independent integration archive audit; never a new timing result."""
from pathlib import Path
import csv, hashlib, json, statistics, subprocess, sys

r = Path('/Users/shawn/CantorAI/Garnet')
d = Path(sys.argv[1]) if len(sys.argv)>1 else r/'results/20261017/worldsense-m5-primary-u32-integration'
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest = load(d/'file-manifest.json')
assert set(manifest)=={str(p.relative_to(d)) for p in d.rglob('*') if p.is_file() and p.name!='file-manifest.json'}
assert all(sha(d/rel)==h for rel,h in manifest.items())
for role, record in load(d/'source-reconstruction.json').items():
    closure=load(d/record['source_manifest'])
    assert len(closure)==record['frozen_source_files']
    for rel, expected in closure.items():
        overlay=d/record['overlay_directory']/rel
        raw=overlay.read_bytes() if overlay.exists() else subprocess.check_output(['git','-C',str(r),'show',record['base_commit']+':'+rel])
        assert hashlib.sha256(raw).hexdigest()==expected, (role,rel)
assert all(load(d/'api-correctness-result.json').values())
assert all(v==0 for v in load(d/'native-results.json').values())
assert all(v['returncode']==0 for v in load(d/'resident-results.json'))
assert (d/'native-off-projection-words.bin').read_bytes()==(d/'native-on-projection-words.bin').read_bytes()
assert load(d/'native-word-comparison.json')['all_FP32_words_identical']
assert load(d/'dependencies-after-production.json')['all_aliases_and_file_bytes_unchanged']
gold=load(r/'results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json')
refs={(i['observation_id'],phase):(i,c) for i in gold['items'] if i['policy']=='candidate' for phase,c in i['result']['model_response'].items() if isinstance(c,dict) and c.get('status')=='ok'}
refq={q['query']:q for q in gold['search_plans'] if q['policy']=='candidate'}
expected=[];datasets={};metrics={};walls={}
for p in sorted((d/'production').glob('*.json')):
    if not p.name.startswith(('control-full-','candidate-full-')):continue
    data=load(p)
    if 'items' not in data or 'search_plans' not in data:continue
    label=data['label'];datasets[label]=data;calls=[]
    assert data['valid'] and data['assignments_restored'] and data['request']==load(d/'production/cases.json')
    assert [(i['repetition'],i['warmup']) for i in data['items']]==[(repetition,repetition==0) for repetition in range(2) for _ in range(3)]
    assert [(i['repetition'],i['warmup']) for i in data['search_plans']]==[(0,True),(1,False)]
    for category,items in [('frame',data['items']),('query',data['search_plans'])]:
        for index,item in enumerate(items):
            responses=item['result']['model_response'] if category=='frame' else {'query_plan':item['result']['benchmark_model_response']}
            for phase,c in responses.items():
                if not isinstance(c,dict) or 'model_id' not in c:continue
                assert c['status']=='ok'
                if category=='frame':
                    ri,ref=refs[(item['observation_id'],phase)]
                    assert all(item[k]==ri[k] for k in ['image_sha256','skill_sha256'])
                    stable=lambda i:{k:v for k,v in i['result'].items() if k not in ['model_response','model_response_json']}
                    assert stable(item)==stable(ri)
                else:
                    ri=refq[item['query']];ref=ri['result']['benchmark_model_response']
                    stable=lambda i:{k:v for k,v in i['result'].items() if k!='benchmark_model_response'}
                    assert stable(item)==stable(ri)
                assert all(c[k]==ref[k] for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
                assert c['timings']['input_tokens']==ref['timings']['input_tokens']
                is_candidate=label.startswith('candidate')
                caps=['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata']
                assert all((cap in c['backend'])==is_candidate for cap in caps)
                if not is_candidate:assert all(k not in c['timings'] for k in ['primary_storage_after','primary_encoded_delta','memory_after_decode'])
                calls.append((item,c));expected.append((str(p.relative_to(d)),category,str(index),phase,c['model_id']))
    assert len(calls)==12
    metrics[label]={}
    for model in ['Qwen3.5-4B','Qwen3-1.7B']:
        cc=[c for item,c in calls if not item['warmup'] and c['model_id']==model];assert len(cc)==3
        metrics[label][model]=dict(samples=3,ttft=statistics.median(c['ttft_seconds'] for c in cc),total=statistics.median(c['total_seconds'] for c in cc),decode_tps=statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds']) for c in cc))
    walls[label]=statistics.median(i['native_call_seconds'] for i in data['items'] if not i['warmup'])
assert len(datasets)==4 and len(expected)==48
for name in ['calls.csv','execution-routes.csv']:
    with (d/name).open() as f: rows=list(csv.DictReader(f))
    actual=[tuple(row[k] for k in ['artifact','category','sample_index','phase','model_id']) for row in rows]
    assert sorted(actual)==sorted(expected) and all(row['artifact_sha256']==sha(d/row['artifact']) for row in rows)
pair_passes=[]
for pair in [1,2]:
    control,candidate='control-full-'+str(pair),'candidate-full-'+str(pair)
    report=load(d/('production/analysis-'+control+'-'+candidate+'.json'))
    assert report['metrics']=={control:metrics[control],candidate:metrics[candidate]}
    assert report['sampled_max_post_request_metal_bytes'][control] is None
    assert report['sampled_metal_reduction'] is None
    passed=1-walls[candidate]/walls[control]>=-.02
    for model in metrics[control]:
        a,b=metrics[control][model],metrics[candidate][model]
        passed=passed and b['ttft']/a['ttft']<=1.02 and b['total']/a['total']<=1.02 and b['decode_tps']/a['decode_tps']>=.98
    assert report['passed']==passed
    pair_passes.append(passed)
decision=load(d/'decision.json')
assert decision['production_pairs_passed']==pair_passes
assert decision['promote']==all(pair_passes)
print(json.dumps(dict(files=len(manifest),source_closures=['canonical','control'],raw_calls=48,exact_outputs=True,production_pair_passes=pair_passes,decision_consistent=True,scope='Independent archive hash/source/coverage/golden/metric audit; original native proofs remain their documented scope.')))
