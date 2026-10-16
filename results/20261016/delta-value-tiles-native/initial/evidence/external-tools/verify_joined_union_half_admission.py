"""Audit frozen full-layer CPU provenance/counters; omitted weight arithmetic is not rerun."""
from pathlib import Path
import hashlib,json,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
first=root/'logs/joined-union-half-admission-20261016';second=root/'logs/joined-union-half-admission-cpp-20261016'
assert set(proof['jobs'])=={first.name,second.name} and all(s['exit_code']==0 for s in proof['jobs'].values())
for folder in (first,second):
    assert json.loads((folder/'exit.json').read_text())==proof['jobs'][folder.name]
    meta=json.loads((folder/'metadata.json').read_text())
    for name,sha in meta['source_sha256'].items():assert digest(folder/'source-snapshot'/Path(name).name)==sha
    assert digest(folder/'source-snapshot/exact_half_codec.h')=='e8800f4983bb2786a7e376ca7784a7817176a5e6dd2d46c7e353116f99b2a4c9'
prior=first/'source-snapshot/report.json';assert digest(prior)=='f9505c75246b7ac43fbf1c28987104dd1e5d4c7101ae265e984af20ba246df97'
old={m['model']:m for m in json.loads(prior.read_text())['models']}
cppmeta=json.loads((second/'metadata.json').read_text());assert cppmeta['prior_report_sha256']==digest(first/'report.json')
stages=json.loads((second/'statuses.json').read_text());assert [s['name'] for s in stages]==['compile','audit'] and all(s['exit_code']==0 for s in stages)
assert '-framework' in stages[0]['command'] and 'Foundation' in stages[0]['command'] and not any(x in stages[0]['command'] for x in ('Metal','MetalPerformanceShaders'))
assert proof['omitted']['logs/'+second.name+'/joined_union_half_admission_cpu']['sha256']==cppmeta['binary_sha256']
records=[json.loads(line) for line in (second/'audit.log').read_text().splitlines() if line.startswith('{')];assert records==json.loads((second/'report.json').read_text())['records'] and len(records)==77
report=json.loads((first/'report.json').read_text());assert len(report['models'])==3;models=[];rejected=restored=completeValues=0
for model,counts in zip(report['models'],((28,17,28),(24,13,23),(24,0,24))):
    name=model['model'];pairs=model['pairs'];assert (len(pairs),model['old_equal_scale_pairs'],model['union_pairs'])==counts
    assert [p['layer'] for p in pairs]==list(range(counts[0]));header=json.loads((first/(name+'-header.json')).read_text());oldCases={c['tensor']:c for c in old[name]['cases']}
    assert old[name]['checkpoint_sha256']==model['checkpoint_sha256']==proof['omitted']['models/'+name+'/model.safetensors']['sha256']
    for pair in pairs:
        n,k=pair['rows'],pair['inner'];assert pair['original_half_weight_payload_bytes']==pair['joined_half_weight_payload_bytes']==n*k*4
        assert len(pair['sides'])==len(pair['checks'])==2
        assert pair['union_scale']==min(s['independent_common_scale'] for s in pair['sides'])
        original=all(s['independent_common_ready'] for s in pair['sides']) and len({s['independent_common_scale'] for s in pair['sides']})==1
        assert pair['prior_equal_scale_eligible']==original and pair['union_admitted']==all(c['admitted'] for c in pair['checks'])
        cpp=next(r for r in records[:-1] if (r['model'],r['layer'])==(name,pair['layer']));assert cpp['union_scale']==pair['union_scale'] and cpp['admitted']==pair['union_admitted']
        assert len(cpp['sides'])==2 and cpp['bitwise_roundtrip_values']==sum(s['encoded_roundtrip_values'] for s in cpp['sides'])
        restored+=cpp['bitwise_roundtrip_values']
        for side,check,c in zip(pair['sides'],pair['checks'],cpp['sides']):
            item=header[side['tensor']];previous=oldCases[side['tensor']]
            assert item['shape']==[n,k] and item['dtype']=='BF16' and side['original_bytes']==n*k*2
            assert side['raw_sha256']==previous['raw_sha256']==c['raw_sha256'] and c['tensor']==side['tensor']
            assert (side['independent_common_scale'],side['independent_common_ready'])==(previous['common_scale'],previous['global_half_ready'])
            assert c['rejected_values']==check['unrepresentable_values'] and c['encoded_roundtrip_values']+c['rejected_values']==n*k
            assert check['admitted']==check['bitwise_roundtrip']==(c['rejected_values']==0);rejected+=c['rejected_values']
        if pair['union_admitted']:completeValues+=n*k*2
    assert model['newly_eligible_layers']==[p['layer'] for p in pairs if p['union_admitted'] and not p['prior_equal_scale_eligible']]
    assert model['fallback_layers']==([8] if name=='Qwen2.5-Coder-0.5B-Instruct' else [])
    models.append(dict(model=name,pairs=counts[0],old_equal_scale_pairs=counts[1],union_pairs=counts[2],fallback_layers=model['fallback_layers']))
last=records[-1];assert (last['pairs'],last['matrices'],last['admitted_pairs'])==(76,152,75)
assert last['bitwise_roundtrip_values']==restored==561512447 and rejected==1
result=dict(verified_files=len(proof['files']),models=models,pairs=76,matrices=152,union_admitted_pairs=75,
 captured_original_cpp_roundtrip_values=restored,rejected_weight_values=rejected,complete_admitted_pair_values=completeValues,
 scope='Captured full CPU eligibility/original-codec provenance and counters; raw checkpoint arithmetic omitted here, no GPU/native performance admission.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
