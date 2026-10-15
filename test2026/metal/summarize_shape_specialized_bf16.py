"""Separate unweighted trained body/head timing pools and excluded setup costs."""
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text());profiles=x['profiles'];pools=[];setup=[]
for trial in sorted({p['trial'] for p in profiles}):
    rows=[p for p in profiles if p['trial']==trial]
    native=[p for p in rows if p['pair']=='native']
    setup.append(dict(trial=trial,configurations=len(native),two_pipeline_setup_sum_ms=sum(p['pipeline_specialization_ms'] for p in native),two_pipeline_setup_max_ms=max(p['pipeline_specialization_ms'] for p in native)))
    for model in sorted({p['model'] for p in rows}):
        for batch in (1,2):
            for part in ('body','head'):
                for pair in ('native','dimensions'):
                    selected=[p for p in rows if p['model']==model and p['batch']==batch and p['pair']==pair and not p['tensor'].startswith('synthetic') and (p['columns']>=100000)==(part=='head')]
                    assert selected
                    pools.append(dict(trial=trial,model=model,batch=batch,part=part,pair=pair,matrix_count=len(selected),fractions={k:sum(p['means']['candidate'][k] for p in selected)/sum(p['means']['control'][k] for p in selected) for k in ('encode_ms','wall_ms','gpu_ms')}))
result=dict(scope='Unweighted component pools, no inference gain. Native pair compares full specialization with original; dimensions pair compares full specialization with dimension-only. Setup builds two pipelines per matrix/batch, excluded from projection clocks; compiler caches unflushed.',pools=pools,pipeline_setup=setup)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(pools=len(pools),setup=setup)))
