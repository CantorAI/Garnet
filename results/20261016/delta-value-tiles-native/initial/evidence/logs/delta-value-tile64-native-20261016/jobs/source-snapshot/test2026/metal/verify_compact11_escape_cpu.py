"""Verify captured CPU histogram provenance and recompute all byte budgets, without omitted weights."""
from pathlib import Path
import hashlib,json,math,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base)
    assert p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
trial='compact11-escape-cpu-20261016';folder=root/'logs'/trial
assert proof['jobs'][trial]==json.loads((folder/'exit.json').read_text()) and proof['jobs'][trial]['exit_code']==0
meta=json.loads((folder/'metadata.json').read_text());report=json.loads((folder/'report.json').read_text())
assert not meta['full_goal_met'] and not report['full_goal_met'] and 'Apple M4' in proof['hardware']['machdep.cpu.brand_string']
for name,sha in meta['source_sha256'].items():assert digest(folder/'source-snapshot'/Path(name).name)==sha
assert len(report['models'])==3;all_cases=eligible=values=0;models=[]
for model in report['models']:
    name=model['model'];manifest=json.loads((root/'logs/decoder-codec-gpu-20261013/jobs'/(name+'-assets')/'manifest.json').read_text())
    assert proof['omitted']['models/'+name+'/model.safetensors']['sha256']==model['checkpoint_sha256']==manifest['checkpoint_sha256']
    assert len(model['cases'])==len(manifest['cases'])==(15 if name=='Qwen3.5-0.8B' else 7)
    for c,r in zip(manifest['cases'],model['cases']):
        all_cases+=1;n,k=c['rows'],c['inner'];assert (n,k,c['tensor'],c['sha256']['raw.bin'])==(r['rows'],r['inner'],r['tensor'],r['raw_sha256'])
        path='logs/decoder-codec-gpu-20261013/jobs/'+name+'-assets/'+c['name']+'/raw.bin'
        assert proof['omitted'][path]==dict(bytes=n*k*2,sha256=r['raw_sha256'])
        special=c['tensor'].startswith('model.layers.') and c['tensor'].endswith('.self_attn.k_proj.weight')
        assert r['eligible']==(not special and 256<n<=8192 and k%128==0 and n*k>=1048576)
        if not r['eligible']:continue
        eligible+=1;spans=r['exponent_span_histogram'];escapes=r['escape_count_histogram']
        assert len(spans)==256 and len(escapes)==513 and all(isinstance(x,int) and x>=0 for x in spans+escapes)
        assert r['blocks']==((n+3)//4)*k//128==sum(spans)
        assert r['compact_blocks']==sum(spans[:16])==sum(escapes)
        assert r['raw_fallback_blocks']==sum(spans[16:])
        assert r['exact_expression_roundtrip_values']==sum(escapes)*512;values+=r['exact_expression_roundtrip_values']
        assert r['retained_compact_payload_metadata_bytes']==sum(escapes)*772
        assert r['logical_raw_fallback_read_bytes']==sum(spans[16:])*1024
        assert r['hypothetical_ideal_packet_bytes']==sum(v*min(772,708+e) for e,v in enumerate(escapes))
        assert r['hypothetical_ranked_aligned_packet_bytes']==sum(v*min(772,((736+e+15)//16)*16+4) for e,v in enumerate(escapes))
    totals=model['totals']
    for key in ('blocks','compact_blocks','raw_fallback_blocks','exact_expression_roundtrip_values','retained_compact_payload_metadata_bytes',
                'hypothetical_ideal_packet_bytes','hypothetical_ranked_aligned_packet_bytes','logical_raw_fallback_read_bytes'):
        assert totals[key]==sum(c.get(key,0) for c in model['cases'])
    assert totals['eligible_matrices']==sum(c['eligible'] for c in model['cases'])
    denominator=totals['retained_compact_payload_metadata_bytes']+totals['logical_raw_fallback_read_bytes']
    for kind in ('ideal','ranked_aligned'):
        ratio=(totals['hypothetical_'+kind+'_packet_bytes']+totals['logical_raw_fallback_read_bytes'])/denominator
        assert math.isfinite(ratio) and ratio==totals[kind+'_packet_fraction']
    models.append(dict(model=name,totals=totals))
assert all_cases==29 and eligible==20 and values==64268800
result=dict(verified_files=len(proof['files']),fixture_matrices=all_cases,eligible_matrices=eligible,captured_exact_expression_values=values,
    models=models,scope=__doc__,full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='models'}))
