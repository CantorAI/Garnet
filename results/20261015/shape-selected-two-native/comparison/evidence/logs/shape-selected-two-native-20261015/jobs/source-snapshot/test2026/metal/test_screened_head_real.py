"""Actual three trained heads through the new native context, before serving timing."""
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();build=Path(sys.argv[3]).resolve();out.mkdir(exist_ok=False);reports=[]
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    assets=root/'logs/screened-head-after-vision-20261015/jobs'/(model+'-assets');meta=json.loads((assets/'manifest.json').read_text())
    assert digest(assets/'raw.bin')==meta['raw_sha256'] and digest(root/'models'/model/'model.safetensors')==meta['checkpoint_sha256']
    for q in meta['queries']:
        for suffix,key in (('.input.bin','input_sha256'),('.reference.bin','reference_sha256')):assert digest(assets/(q['name']+suffix))==q[key]
    command=[str(build/'garnet_screened_head_real_test'),str(assets)]
    with (out/(model+'.log')).open('w') as stream:r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    assert r.returncode==0,model
    values=[json.loads(v) for v in (out/(model+'.log')).read_text().splitlines() if v.startswith('{')]
    assert len(values)==17 and {v['query'] for v in values}=={q['name'] for q in meta['queries']}
    assert all(v['raw_bits_exact'] and v['retained_bits_exact'] and v['every_rejection_checked'] and v['greedy_exact'] and v['dispatches']==1 for v in values)
    reports.append(dict(model=model,command=command,manifest=meta,checks=values))
(out/'summary.json').write_bytes((json.dumps(dict(profiles=reports,query_checks=51,row_checks=78),indent=2)+'\n').encode());print('All 51 actual trained context-head queries passed original GPU bits and greedy checks',flush=True)
