"""Independent CPU FP32 causal attention oracle; no native implementation import.

Generate after primary timing jobs finish. Fixtures cover padded chunks,
ragged/inactive rows and the actual 2B/4B query:KV ratio and head dimensions.
"""
import json,sys
from pathlib import Path
import numpy as np
import torch
root=Path(sys.argv[1]);wide='--wide' in sys.argv[2:]
if wide:assert root.is_dir() and not (root/'attention-wide-fixtures.json').exists()
else:assert not root.exists();root.mkdir(parents=True)
torch.set_num_threads(1);rng=np.random.default_rng(4471);cases=[]
profiles=[(1,1,8,2,32,32),(2,4,8,2,32,128),(4,4,8,2,512,256),
          (8,4,16,4,512,256),(2,16,16,4,128,256),
          (1,4,4,1,33,64),(1,1,8,2,8192,256)]
if wide:profiles=[(1,32,8,2,65,32),(1,64,8,2,129,256),(2,128,16,4,257,256),(8,128,8,2,257,32)]
for index,(batch,tokens,heads,kv,capacity,dim) in enumerate(profiles):
    folder=root if wide else root/str(index)
    if not wide:folder.mkdir()
    query=rng.normal(0,.25,(batch,tokens,heads,dim)).astype(np.float32)
    key=rng.normal(0,.25,(batch,tokens,kv,dim)).astype(np.float32)
    value=rng.normal(0,.25,key.shape).astype(np.float32)
    gate=rng.normal(0,2,query.shape).astype(np.float32)
    old_key=rng.normal(0,.1,(batch,kv,capacity,dim)).astype(np.float32)
    old_value=rng.normal(0,.1,old_key.shape).astype(np.float32)
    active=np.ones((batch,tokens),np.float32);slots=np.zeros_like(active)
    for b in range(batch):
        for t in range(tokens):
            if (b+t)%3==1:active[b,t]=0
            if b==batch-1 and batch>1:active[b,t]=0
        next_slot=capacity-int(active[b].sum()) if b==0 else 3+b
        for t in range(tokens):
            slots[b,t]=next_slot;next_slot+=int(active[b,t])
    next_key=torch.from_numpy(old_key.copy());next_value=torch.from_numpy(old_value.copy())
    expected=torch.zeros(query.shape,dtype=torch.float32)
    # Serial reference writes each token and attends only the visible prefix.
    for t in range(tokens):
        for b in range(batch):
            if not active[b,t]:continue
            slot=int(slots[b,t]);next_key[b,:,slot]=torch.from_numpy(key[b,t]);next_value[b,:,slot]=torch.from_numpy(value[b,t])
            for h in range(heads):
                kh=h//(heads//kv);q=torch.from_numpy(query[b,t,h]);keys=next_key[b,kh,:slot+1]
                scores=keys@q/(dim**.5);probabilities=scores.softmax(-1)
                attended=probabilities@next_value[b,kh,:slot+1]
                expected[b,t,h]=attended*torch.from_numpy(gate[b,t,h]).sigmoid()
    arrays=[query,key,value,gate,old_key,old_value,slots,active]
    outputs=[expected.numpy(),next_key.numpy(),next_value.numpy()]
    def save(arr,name):
        path=folder/(f'attention-wide-case{index}-'+name if wide else name);assert not path.exists()
        arr.astype('<f4').tofile(path);return {'file':str(path.resolve()),'shape':list(arr.shape)}
    cases.append({'config':dict(batch=batch,tokens=tokens,heads=heads,kv=kv,capacity=capacity,dim=dim),
        'inputs':[save(arr,f'input{i}.f32') for i,arr in enumerate(arrays)],
        'outputs':[save(arr,f'expected{i}.f32') for i,arr in enumerate(outputs)]})
(root/('attention-wide-fixtures.json' if wide else 'fixtures.json')).write_text(json.dumps(cases,indent=2));print('ATTENTION_CORE_FIXTURES_PREPARED',len(cases))
