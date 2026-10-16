"""Parse preserved native measurements; check exact short-answer token parity."""
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);records=[];statuses=[]
models=('Qwen3.5-0.8B','Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct')
for path in sorted(root.rglob('output.log')):
    relative=path.parent.relative_to(root).as_posix();results=[];process=None
    status_path=path.parent/'exit.json'
    if status_path.exists():statuses.append(dict(folder=relative,**json.loads(status_path.read_text(encoding='utf-8'))))
    for line in path.read_text(encoding='utf-8').splitlines():
        marker,separator,payload=line.partition(' ')
        if marker in ('metal_image_result','metal_text_result','metal_three_resident_result'):
            value=json.loads(payload);value['folder']=relative
            value['model']=value.get('model',next((name for name in models if name in relative),
                'Qwen3.5-0.8B' if 'image' in relative else 'Qwen2.5-Coder-0.5B-Instruct' if 'coder' in relative else 'Qwen3-0.6B'))
            results.append(value)
        elif marker in ('metal_image_process','metal_text_process','metal_three_resident_process'):process=json.loads(payload)
    for result in results:result['process']=process;records.append(result)
comparisons=[]
for batch in (1,2):
    for model in models:
        group=[v for v in records if any(v['folder'].endswith('-b'+str(batch)+'-'+suffix) for suffix in ('control','optimized','repeat')) and v['model']==model and v['batch']==batch]
        if not group:continue
        control=next(v for v in group if v['request']==1 and v['folder'].endswith('control'))
        for value in group:
            if value['token_ids']!=control['token_ids'] or value['finished']!=control['finished']:
                raise ValueError('Output parity failed: '+value['folder'])
        comparisons.append(dict(model=model,batch=batch,exact_tokens=True,requests_checked=len(group),finished=control['finished']))
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
    group=[v for v in records if 'extended' in v['folder'] and v['model']==model]
    if group:
        for value in group:
            if value['token_ids']!=group[0]['token_ids'] or value['finished']!=group[0]['finished']:
                raise ValueError('Extended output parity failed: '+value['folder'])
        comparisons.append(dict(model=model,extended=True,exact_tokens=True,requests_checked=len(group),finished=group[0]['finished']))
for model in models:
    baseline=[v for v in records if 'baseline' in v['folder'] and v['model']==model and v['request']==1]
    current=[v for v in records if v['folder'].endswith('-b1-optimized') and v['model']==model and v['request']==1]
    if baseline and current:
        assert baseline[0]['token_ids']==current[0]['token_ids'] and baseline[0]['finished']==current[0]['finished'],'Original build output parity failed'
        comparisons.append(dict(model=model,original_build_parity=True,exact_tokens=True))
    combined=[v for v in records if v.get('models_loaded')==3 and v['model']==model]
    if combined and current:
        assert all(v['token_ids']==current[0]['token_ids'] and v['finished']==current[0]['finished'] for v in combined),'Resident model switching changed output'
        comparisons.append(dict(model=model,all_three_resident=True,exact_tokens=True,requests_checked=len(combined)))
output=dict(records=records,comparisons=comparisons,statuses=statuses)
(root/'benchmarks.json').write_bytes((json.dumps(output,indent=2)+'\n').encode('utf-8'))
for value in records:
    if value['request']==1:
        print(json.dumps(dict(folder=value['folder'],model=value['model'],ttft=value['ttft_loaded_seconds'],total=value['total_loaded_seconds'],
            tps=value['decode_steps_per_second'],tokens=value['output_tokens'],finished=value['finished'],
            rss_gib=value['process']['peak_rss_bytes']/2**30 if value['process'] else None,
            token_sha256=hashlib.sha256(json.dumps(value['token_ids']).encode()).hexdigest())))
print('Exact token comparisons:',len(comparisons))
