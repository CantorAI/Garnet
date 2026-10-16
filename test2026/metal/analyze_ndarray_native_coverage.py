"""Header-derived route/payload calculation; not a measurement of unique GPU memory or RSS."""
from pathlib import Path
import hashlib,json,math,sys
repo=Path(__file__).resolve().parents[2]
old=repo/'results/20261015/shape-selected-two-native/comparison'
proof=json.loads((old/'evidence/manifest.json').read_text())
component=repo/'results/20261015/ndarray-prefill/repaired/evidence/logs/ndarray-prefill-repaired-20261015/jobs/metadata.json'
meta=json.loads(component.read_text());models=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    name='evidence/checkpoint-headers/'+model+'.json';p=old/name;info=proof['files'][name]
    assert p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
    checkpoint=proof['omitted']['models/'+model+'/model.safetensors']['sha256']
    assert checkpoint==meta['assets'][model]['checkpoint_sha256']
    header=json.loads(p.read_text());tensors=[]
    for name,item in header.items():
        if not(name.startswith(('model.layers.','model.language_model.layers.')) and item.get('dtype')=='BF16' and len(item.get('shape',[]))==2):continue
        if name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight'):continue
        values=math.prod(item['shape']);lo,hi=item['data_offsets'];assert hi-lo==values*2
        tensors.append(dict(name=name,shape=item['shape'],values=values,original_bytes=values*2,expanded_cache_bytes=values*4))
    assert len(tensors)<=256
    profiles=[]
    for batch in (1,2):
        rows=(80*batch if model=='Qwen3.5-0.8B' else (24 if batch==1 else 44) if model=='Qwen3-0.6B' else 26*batch)
        enabled=rows==80
        profiles.append(dict(batch=batch,physical_rows=rows,expected_prefill_dispatches=len(tensors) if enabled else 0,
            expected_decode_dispatches=0,logical_weight_cache_bytes=sum(t['expanded_cache_bytes'] for t in tensors) if enabled else 0))
    models.append(dict(model=model,checkpoint_sha256=checkpoint,header_sha256=info['sha256'],eligible_layer_matrices=len(tensors),
        cache_entry_cap=256,all_expanded_weight_bytes=sum(t['expanded_cache_bytes'] for t in tensors),tensors=tensors,profiles=profiles))
result=dict(models=models,source_header_archive_sha256=json.loads((old/'archive-verification.json').read_text())['archive_sha256'],
    component_metadata_sha256=digest(component),scope='Arithmetic over byte-verified frozen checkpoint headers matching trained component checkpoints. Logical exact-F32 weight payload only; ignores allocator/MPS repacking and view/cache ownership overhead. Does not prove actual routes, stable per-request keys, measured RSS, unique GPU memory, speed or full goal.',full_goal_met=False)
Path(sys.argv[1]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(models=[{k:v for k,v in m.items() if k in ('model','eligible_layer_matrices','all_expanded_weight_bytes','profiles')} for m in models],full_goal_met=False)))
