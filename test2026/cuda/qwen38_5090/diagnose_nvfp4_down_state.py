from pathlib import Path
import hashlib,json
import numpy as np
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/nvfp4-down-full-qualification';folder=out/'state'
a=json.loads((folder/'original-report.json').read_text(encoding='utf8'));b=json.loads((folder/'joined-report.json').read_text(encoding='utf8'))
assert a['status']=='passed' and b['status']=='failed';rows=[]
for old,new in zip(a['cases'],b['cases']):
 assert old['case']==new['case'] and old['repeat']==new['repeat']
 changed=[{'index':i,'offset':x['offset'],'words':x['words']} for i,(x,y) in enumerate(zip(old['chunks'],new['chunks'])) if x['sha256']!=y['sha256']]
 case=old['case'];repeat=old['repeat'];left=r/('results/joined-program-equivalence/original-'+case+'-'+str(repeat)+'-first.f32');right=folder/('joined-'+case+'-'+str(repeat)+'-first.f32')
 raw_a,raw_b=left.read_bytes(),right.read_bytes();assert hashlib.sha256(raw_a).hexdigest()==old['chunks'][0]['sha256'] and hashlib.sha256(raw_b).hexdigest()==new['chunks'][0]['sha256']
 x=np.frombuffer(raw_a,dtype='<f4');y=np.frombuffer(raw_b,dtype='<f4');assert np.isfinite(x).all() and np.isfinite(y).all()
 bits=x.view('<u4')!=y.view('<u4');indices=np.flatnonzero(bits);hidden=b['program']['hidden_width'];delta=np.abs(x.astype(np.float64)-y.astype(np.float64))
 rows.append({'case':case,'repeat':repeat,'changed_chunks':changed,'total_chunks':len(old['chunks']),'prefix_words':len(x),'prefix_different_words':int(bits.sum()),'prefix_hidden_different_words':int(bits[:hidden].sum()),'prefix_state_different_words':int(bits[hidden:].sum()),'prefix_max_abs':float(delta.max()),'prefix_rms_difference':float(np.sqrt(np.mean(delta**2))),'first_difference_samples':[{'index':int(i),'old':float(x[i]),'candidate':float(y[i]),'abs':float(delta[i])} for i in indices[:8]]})
path=out/'state-difference-diagnostic.json';assert not path.exists();result={'status':'exact_state_gate_failed','inputs_equal':a['inputs']==b['inputs'],'rows':rows,'scope':'CPU first-retained-chunk numerical diagnosis and full chunk-hash difference map. Remaining raw states unavailable here; no attribution, harmlessness, tolerance/quality/performance waiver.'};path.write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}));print(json.dumps([{k:v for k,v in row.items() if k not in ('changed_chunks','first_difference_samples')}|{'changed_chunks_count':len(row['changed_chunks'])} for row in rows]))
