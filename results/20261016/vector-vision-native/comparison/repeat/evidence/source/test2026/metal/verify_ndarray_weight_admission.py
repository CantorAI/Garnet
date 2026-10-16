"""Audit frozen CPU eligibility provenance and counters; raw weights are omitted.

This verifies captured original-C++ roundtrips, not an independent rerun of
the original weight arithmetic on this host. Full model checkpoints/raw
matrices must be supplied to rerun that calculation.
"""
from pathlib import Path
import hashlib,json,sys
base=Path(sys.argv[1]).resolve();reference=Path(sys.argv[2]).resolve()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((base/'evidence/manifest.json').read_text())
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base)
    assert p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
archive=json.loads((base/'archive-verification.json').read_text())
assert archive['archive_sha256']=='66d403a090e6275e9a623459b33b9a9ffd30de19e9811cc3ce84a36da74f8e4e'
trials=('ndarray-weight-admission-20261016','ndarray-weight-admission-cpp-20261016')
assert set(proof['jobs'])==set(trials) and all(v['exit_code']==0 for v in proof['jobs'].values())
folders=[base/'evidence/logs'/t for t in trials]
for t,p in zip(trials,folders):assert json.loads((p/'exit.json').read_text())==proof['jobs'][t]
first,second=folders
meta=json.loads((first/'metadata.json').read_text());cpp=json.loads((second/'metadata.json').read_text())
for m,p in ((meta,first),(cpp,second)):
    assert not m['full_goal_met']
    for name,sha in m['source_sha256'].items():assert digest(p/'source-snapshot'/Path(name).name)==sha
assert digest(first/'source-snapshot/exact_half_codec.h')==digest(second/'source-snapshot/exact_half_codec.h')=='e8800f4983bb2786a7e376ca7784a7817176a5e6dd2d46c7e353116f99b2a4c9'
assert digest(first/'report.json')==cpp['prior_report_sha256']
stages=json.loads((second/'statuses.json').read_text());assert [v['name'] for v in stages]==['compile','audit']
assert all(v['exit_code']==0 for v in stages)
compile_command=stages[0]['command'];assert compile_command[compile_command.index('-framework')+1]=='Foundation'
assert not any(v in compile_command for v in ('Metal','MetalPerformanceShaders'))
assert proof['omitted']['logs/'+trials[1]+'/weight_admission_cpu']['sha256']==cpp['binary_sha256']
records=[json.loads(line) for line in (second/'audit.log').read_text().splitlines() if line.startswith('{')]
assert records==json.loads((second/'report.json').read_text())['records'] and records[-1]['kind']=='complete'
assert len(records)==6 and records[-1]['matrices']==5
native_proof=json.loads((reference/'evidence/manifest.json').read_text())
assert json.loads((reference/'archive-verification.json').read_text())['archive_sha256']==meta['baseline_archive_sha256']=='7ed9d48e71a3723cf84e62fc95170bbee52cf614d90ef8fd7c93df338dc92c33'
assert native_proof['jobs']['ndarray-native-repaired-20261015']['exit_code']==0
report=json.loads((first/'report.json').read_text());assert len(report['models'])==3 and not report['full_goal_met']
models=[];counter_checks=0;non_common=[];all_cases=[]
for model,counts in zip(report['models'],((168,168,168),(144,143,144),(186,182,185))):
    name=model['model'];cases=model['cases'];header=json.loads((first/(name+'-header.json')).read_text())
    expected={n:v for n,v in header.items() if n.startswith(('model.layers.','model.language_model.layers.')) and v.get('dtype')=='BF16' and len(v.get('shape',[]))==2 and not(n.startswith('model.layers.') and n.endswith('.self_attn.k_proj.weight'))}
    assert len(cases)==len(expected) and {c['tensor'] for c in cases}==set(expected)
    assert (len(cases),sum(c['global_half_ready'] for c in cases),sum(c['exact_half_ready'] for c in cases))==counts
    assert model['matrices']==counts[0] and model['global_half_ready']==counts[1] and model['exact_half_ready']==counts[2]
    assert proof['omitted']['models/'+name+'/model.safetensors']['sha256']==model['checkpoint_sha256']
    for c in cases:
        item=expected[c['tensor']];n,k=item['shape'];assert (n,k)==(c['rows'],c['inner']) and k%8==0
        assert item['data_offsets'][1]-item['data_offsets'][0]==c['original_bytes']==n*k*2
        assert c['full_f32_cache_bytes']==n*k*4 and c['storage_cap_admitted']==(n*k*2<=16*1024*1024)
        assert c['global_half_ready']==(c['storage_cap_admitted'] and c['common']['admitted'])
        assert c['exact_half_ready']==(c['storage_cap_admitted'] and (c['common']['admitted'] or c['rowwise']['admitted']))
        for test in (c['common'],c['rowwise']):
            if test is not None:
                assert test['bitwise_roundtrip']==test['admitted']
                assert test['admitted']==(test['unrepresentable_values']==test['unrepresentable_rows']==0)
        if not c['global_half_ready']:non_common.append((name,c))
    for key,predicate in (('all_f32_cache_bytes',lambda c:True),('non_common_f32_cache_bytes',lambda c:not c['global_half_ready']),('no_exact_f32_cache_bytes',lambda c:not c['exact_half_ready'])):
        assert model[key]==sum(c['full_f32_cache_bytes'] for c in cases if predicate(c))
    assert model['non_common']==[c['tensor'] for c in cases if not c['global_half_ready']]
    assert model['no_exact_half']==[c['tensor'] for c in cases if not c['exact_half_ready']]
    exports=json.loads((first/'assets'/name/'manifest.json').read_text());assert exports['checkpoint_sha256']==model['checkpoint_sha256']
    assert exports['cases']==[c for c in cases if not c['global_half_ready']]
    actual=[]
    for batch in (1,2):
        label=name+'-b'+str(batch)
        rel='evidence/logs/ndarray-native-repaired-20261015/jobs/'+label+'-attempt0/'+label+'-control-0/output.log'
        p=reference/rel;identity=native_proof['files'][rel];assert p.stat().st_size==identity['bytes'] and digest(p)==identity['sha256']
        prefix='matched_frame_result ' if name=='Qwen3.5-0.8B' else 'matched_result '
        rows=[json.loads(line.split(' ',1)[1]) for line in p.read_text().splitlines() if line.startswith(prefix)];assert len(rows)==3
        for row in rows:
            chunks=sum(2<batch*c<=128 or batch*c==160 for c in row['prefill_chunk_plan']);count=counts[1]*chunks
            observed=row['prefill_profile'].get('global_half_alpha_dispatches',0);assert count==observed
            actual.append(dict(batch=batch,request=row['request'],expected_global_half_calls=count,actual_global_half_calls=observed));counter_checks+=1
    assert actual==model['baseline_counter_checks']
    all_cases.extend((name,c) for c in cases)
    models.append({k:v for k,v in model.items() if k not in ('cases','baseline_counter_checks')})
assert len(all_cases)==498 and len(non_common)==5 and counter_checks==18
assert len({(r['model'],r['tensor']) for r in records[:-1]})==5
total=0
for name,c in non_common:
    p='logs/'+trials[0]+'/assets/'+name+'/'+c['name']+'/raw.bin'
    assert proof['omitted'][p]==dict(bytes=c['original_bytes'],sha256=c['raw_sha256'])
    r=next(r for r in records[:-1] if (r['model'],r['tensor'])==(name,c['tensor']))
    assert r['raw_sha256']==c['raw_sha256'] and r['global_half_ready']==c['global_half_ready'] and r['exact_half_ready']==c['exact_half_ready']
    assert len(r['bad_rows'])==c['rowwise']['unrepresentable_rows']
    expected_values=(c['rows']-len(r['bad_rows']))*c['inner'];assert r['bitwise_roundtrip_values']==expected_values;total+=expected_values
    if r['bad_rows']:
        assert name=='Qwen3.5-0.8B' and c['raw_sha256']=='8904cdb2c2e3aa5fc158d074e3b84fb5165a6637f2bd38f67fb93b29406fe0dc'
        assert [r['row'] for r in r['bad_rows']]==[397,423,438,453,470,476,479,2445,2471,2486,2501,2518,2524,2527]
        assert all(r['max_bf16_exponent']==4 and r['requested_scale']==138 for r in r['bad_rows'])
assert records[-1]['bitwise_roundtrip_values']==total==19548160
assert models[-1]['no_exact_f32_cache_bytes']==24*2**20
result=dict(verified_files=len(proof['files']),models=models,body_matrices=498,original_cpp_crosschecked_matrices=5,captured_bitwise_roundtrip_values=total,
    actual_baseline_counter_checks=counter_checks,archive_sha256=archive['archive_sha256'],scope=__doc__,full_goal_met=False)
Path(sys.argv[3]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k not in ('models','scope')}))
