"""Verify actual component checks/timings, identities, exits and profile sizes."""
import hashlib,json,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]).resolve();e=r/'evidence';files=json.loads((e/'files.json').read_text());successful=[]
for name,item in files.items():
    raw=(r/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],name
for folder in sorted((e/'logs').glob('state-zero-gpu-probe*')):
    code=json.loads((folder/'exit.json').read_text())['exit_code'];job=folder/'jobs'
    metadata=json.loads((job/'metadata.json').read_text())
    for name,expected in metadata['source_sha256'].items():assert hashlib.sha256((job/'source-snapshot'/name).read_bytes()).hexdigest()==expected
    if code:
        assert folder.name=='state-zero-gpu-probe-20261013' and code==1
        assert 'expected expression' in (job/'compile.log').read_text();continue
    statuses=json.loads((job/'statuses.json').read_text());assert len(statuses)==2 and all(s['exit_code']==0 for s in statuses)
    if 'conditioned' in folder.name:
        assert statuses[-1]['command'][-1]=='gpu-conditioned'
        assert 'GPU writes' in metadata['input_conditioning']
    assert json.loads((job/'gpu-monitor.json').read_text())
    rows=[json.loads(v) for v in (job/'gpu.log').read_text().splitlines() if v.startswith('{')]
    checks=[v for v in rows if v['kind']=='check'];assert len(checks)==63
    profiles=metadata['profiles'];assert len(profiles)==7
    for model,entry in metadata['configs'].items():
        c=entry['config'];expected=[]
        for kind in c.get('layer_types',['full_attention']*c['num_hidden_layers']):
            if kind=='linear_attention':
                channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim']
                expected.extend([channels*c['linear_conv_kernel_dim'],c['linear_num_value_heads']*c['linear_key_head_dim']*c['linear_value_head_dim']])
            else:
                width=c['num_key_value_heads']*256*c.get('head_dim',c['hidden_size']//c['num_attention_heads']);expected.extend([width,width])
        assert len(expected)==2*c['num_hidden_layers']
        for batch in (1,2):assert next(p for p in profiles if p['name']==model+'-b'+str(batch))['columns']==expected
    for p in profiles:
        assert p['logical_bytes']==p['batch']*sum(p['columns'])*4
        checked=[c for c in checks if c['profile']==p['name']];assert len(checked)==9
        assert {(c['policy'],c['generation']) for c in checked}=={(p,g) for p in range(3) for g in range(3)}
        assert all(c['bytes']==p['logical_bytes'] and c['cpu_gpu_positive_zero'] and c['padding_preserved'] for c in checked)
        timings=[v for v in rows if v['kind']=='timing' and v['profile']==p['name']];assert len(timings)==12
        assert all(t['samples']==8 and t['wall_ms_median']>0 for t in timings)
        medians={str(policy):statistics.median(t['wall_ms_median'] for t in timings if t['policy']==policy) for policy in range(3)}
        value=next(v for v in json.loads((job/'summary.json').read_text())['profiles'] if v['profile']==p['name'])
        assert value['wall_ms_median']==medians and abs(value['gpu_over_direct_cpu']-medians['1']/medians['2'])<1e-12
    successful.append(folder.name)
assert successful
print(json.dumps(dict(files=len(files),successful_runs=successful,checks=63*len(successful),scope='Reset component only; no full-model speed claim')))
