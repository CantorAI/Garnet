"""FP64 oracle for FP32 accumulation plus BF16 output, fixed before GPU execution."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root).resolve()
manifest=root/'indexed-head-fixtures.json';assert root.is_dir() and not manifest.exists()
rng=np.random.default_rng(574991);cases=[]
for number,(rows,hidden,vocab,count) in enumerate([(1,8,64,32),(2,2048,2048,512),(8,2560,4096,512),(1,2048,16384,8192),(1,8,64,32)]):
    query=torch.tensor(rng.normal(0,.6,(rows,hidden)),dtype=torch.float32)
    weight=torch.tensor(rng.normal(0,.03,(vocab,hidden)),dtype=torch.bfloat16)
    indices=rng.integers(0,vocab,(rows,count),dtype=np.int32)
    if rows>1:query[-1].zero_()
    indices[0,-2:]=[-1,vocab+3]
    if number==4:indices.fill(-1)
    q=query.to(torch.bfloat16).double().numpy();dots=np.zeros((rows,count),np.float64);sums=dots.copy()
    for row in range(rows):
        selected=weight[torch.tensor(np.clip(indices[row],0,vocab-1).astype(np.int64))].double().numpy()
        dots[row]=selected@q[row];sums[row]=np.abs(selected*q[row]).sum(-1)
    spec={'config':{'rows':rows,'hidden':hidden,'vocab':vocab,'candidates':count},'files':{}}
    for name,value in [('queries',query.numpy()),('table',weight.view(torch.uint16).numpy()),('indices',indices),('dots',dots),('sum_abs',sums)]:
        path=root/f'indexed-head-case{number}-{name}.bin';assert not path.exists();value.tofile(path);spec['files'][name]=str(path)
    cases.append(spec)
manifest.write_text(json.dumps(cases,indent=2));print('QWEN35_INDEXED_HEAD_FIXTURES_PREPARED',len(cases))
