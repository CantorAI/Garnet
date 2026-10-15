"""Vectorized implementation of the existing dense-reference acceptance rules."""
import json
from pathlib import Path
import numpy as np

def validate(fixture):
    fixture=Path(fixture);f=json.loads(fixture.read_text());a=json.loads(Path(str(fixture)+'.capture.json').read_text())
    assert a['invalid_controls_rejected'] and a['closed_graph_rejected']
    assert all(np.all(np.asarray(v)==0) for v in a['reset'])
    c=f['config'];batch=f['batch'];chunk=f['chunk'];capacity=f['capacity'];dim=c['head_dim'];kv=c['num_key_value_heads'];d=c['hidden_size']
    hidden_width=chunk*d;cache_width=kv*capacity*dim;row_width=hidden_width+2*cache_width;reports=[]
    assert len(a['steps'])==len(f['steps'])==3
    for number,(actual,step) in enumerate(zip(a['steps'],f['steps'])):
        obtained=np.asarray(actual['actual'],np.float64).reshape(batch,row_width)
        expected=np.fromfile(step['expected'],dtype='<f4').astype(np.float64).reshape(batch,row_width)
        resident=np.concatenate([np.asarray(actual[k],np.float64).reshape(batch,-1) for k in ('hidden','key','value')],axis=1)
        explicit=np.asarray(actual['explicit'],np.float64).reshape(batch,row_width)
        assert np.array_equal(resident,explicit),'Resident/explicit mismatch'
        assert np.all(np.isfinite(obtained)),'Nonfinite output'
        h=obtained[:,:hidden_width].reshape(batch,chunk,d);hr=expected[:,:hidden_width].reshape(batch,chunk,d)
        v=obtained[:,hidden_width+cache_width:].reshape(batch,kv*capacity,dim);vr=expected[:,hidden_width+cache_width:].reshape(batch,kv*capacity,dim)
        def vectors(x,y):
            error=np.sqrt(np.sum((x-y)**2,axis=-1));scale=np.sqrt(np.sum(y*y,axis=-1))
            l2=error/(5e-6+5e-5*scale)
            bound=5e-6+5e-5*np.max(np.abs(y),axis=-1,keepdims=True)
            return float(np.max(l2)),float(np.max(np.abs(x-y)/bound))
        hidden_l2,hidden_ratio=vectors(h,hr);value_l2,value_ratio=vectors(v,vr)
        k=obtained[:,hidden_width:hidden_width+cache_width].reshape(batch,kv*capacity,dim)
        kr=expected[:,hidden_width:hidden_width+cache_width].reshape(batch,kv*capacity,dim)
        half=dim//2
        magnitude=np.sqrt(kr[...,:half]**2+kr[...,half:]**2)
        difference=np.sqrt((k[...,:half]-kr[...,:half])**2+(k[...,half:]-kr[...,half:])**2)
        key_ratio=float(np.max(difference/(5e-6+5e-5*magnitude)))
        worst=max(hidden_ratio,value_ratio,key_ratio)
        report=dict(layer=f['layer'],batch=batch,chunk=chunk,step=number,max_abs_error=float(np.max(np.abs(obtained-expected))),hidden_l2_worst_ratio=hidden_l2,value_l2_worst_ratio=value_l2,worst_ratio=worst,key_pair_ratio=key_ratio)
        print('dense vector reference',json.dumps(report),flush=True)
        assert hidden_l2<=1 and value_l2<=1 and worst<=1,report
        reports.append(report)
    print('pretrained vector dense PASS',f['layer'],batch,chunk,flush=True)
    return reports
