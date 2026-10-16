"""Offline partial Cauchy bounds against actual full native GPU head logits.

Arguments: model.safetensors, captured queries.json, report.json.
Counterfactual weight reads exclude bound metadata, reductions and dispatch.
No pruned GPU implementation or inference-speed improvement is demonstrated.
"""
import hashlib,json,mmap,struct,sys
from pathlib import Path
import numpy as np

def gamma(n,bits):
    value=n*2.0**(-bits);assert value<1;return value/(1-value)

def main():
    checkpoint=Path(sys.argv[1]);query_path=Path(sys.argv[2]);destination=Path(sys.argv[3])
    capture=json.loads(query_path.read_text());batch=capture['batch'];width=capture['width'];samples=[];logits=[];score_digests={}
    for query in capture['queries']:
        path=Path(query['scores_file']);data=path.read_bytes();score_digests[path.name]=hashlib.sha256(data).hexdigest()
        scores=np.asarray(json.loads(data),np.float32).reshape(batch,-1)
        hidden=np.asarray(query['normalized'],np.float32).reshape(batch,width)
        for row in range(batch):
            if not query['active'][row]:continue
            assert int(scores[row].argmax())==query['prediction'][row]
            samples.append(dict(step=query['step'],row=row,prediction=query['prediction'][row],hidden=hidden[row].astype(np.float64)))
            logits.append(scores[row].astype(np.float64))
    h=np.asarray([sample['hidden'] for sample in samples]);reference=np.asarray(logits);queries=len(h)
    assert width%32==0 and queries>0
    base_block=32;blocks=width//base_block;rows=reference.shape[1]
    terms=np.empty((queries,rows,blocks),np.float64);norms=np.empty((rows,blocks),np.float64)
    with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
        length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);base=8+length
        info=header[capture['head_weight']];assert info['dtype']=='BF16' and info['shape']==[rows,width]
        for start in range(0,rows,512):
            count=min(512,rows-start)*width
            bits=np.frombuffer(mapping,dtype='<u2',count=count,offset=base+info['data_offsets'][0]+start*width*2).astype(np.uint32)
            values=(bits<<16).view(np.float32).astype(np.float64).reshape(-1,blocks,base_block)
            assert np.isfinite(values).all()
            norms[start:start+len(values)]=np.sum(values*values,axis=2)*(1+gamma(2*base_block+4,53))
            terms[:,start:start+len(values)]=np.einsum('rbd,qbd->qrb',values,h.reshape(queries,blocks,base_block),optimize=False)
        del values,bits
    norms=np.nextafter(norms,np.inf)
    hidden_norms=np.nextafter(np.sum(h.reshape(queries,blocks,base_block)**2,axis=2)*(1+gamma(2*base_block+4,53)),np.inf)
    total_norms=norms.sum(axis=1)*(1+gamma(blocks+4,53))
    total_hidden=hidden_norms.sum(axis=1)*(1+gamma(blocks+4,53))
    # Deliberately conservative: covers any sum/reduction order within the
    # native FP32 dot. The FP64 calculation has a separate, smaller envelope.
    budget=(gamma(2*width+64,24)+gamma(2*width+64,53)+1e-12)*np.sqrt(total_norms[None,:]*total_hidden[:,None])
    cpu_logits=terms.sum(axis=2);error=np.abs(cpu_logits-reference)
    assert np.all(error<=budget),'Actual GPU logits exceed the assumed numerical envelope'
    assert np.array_equal(cpu_logits.argmax(axis=1),reference.argmax(axis=1)),'FP64 reference changes the winner'
    results=[];seed_rows=min(8192,rows);max_violation=-np.inf
    for block_size in (32,64,128):
        assert width%block_size==0;group=block_size//base_block;number=width//block_size
        grouped_terms=terms.reshape(queries,rows,number,group).sum(axis=3)
        grouped_norms=norms.reshape(rows,number,group).sum(axis=2)*(1+gamma(group+4,53))
        grouped_hidden=hidden_norms.reshape(queries,number,group).sum(axis=2)*(1+gamma(group+4,53))
        for q,sample in enumerate(samples):
            for order_name in ('contiguous','hidden-energy'):
                order=np.arange(number) if order_name=='contiguous' else np.argsort(-grouped_hidden[q],kind='stable')
                for threshold_name in ('fixed-8192-seed','oracle-full-max'):
                    threshold=float(reference[q,:seed_rows].max() if threshold_name=='fixed-8192-seed' else reference[q].max())
                    active=np.ones(rows,dtype=bool);active[:seed_rows]=False
                    partial=np.zeros(rows,np.float64);reads=seed_rows*width;curve=[]
                    for index,block in enumerate(order):
                        reads+=int(active.sum())*block_size;partial+=grouped_terms[q,:,block]
                        remaining=order[index+1:]
                        residual=grouped_norms[:,remaining].sum(axis=1)*(1+gamma(number+4,53))
                        hidden_residual=float(grouped_hidden[q,remaining].sum())*(1+gamma(number+4,53))
                        bound=partial+np.sqrt(residual*hidden_residual)*(1+1e-12)+budget[q]
                        violation=float(np.max(reference[q]-bound));max_violation=max(max_violation,violation)
                        assert violation<=0,'Partial bound underestimates an actual GPU logit'
                        prunable=active&(bound<threshold)
                        assert not prunable[sample['prediction']],'Bound discards the actual winning row'
                        active[prunable]=False
                        curve.append(dict(computed_columns=(index+1)*block_size,remaining_rows=int(active.sum())))
                    results.append(dict(step=sample['step'],row=sample['row'],prediction=sample['prediction'],block_size=block_size,
                        order=order_name,threshold=threshold_name,seed_max=threshold,
                        counterfactual_weight_fraction=reads/(rows*width),counterfactual_weight_bytes=reads*2,
                        includes_full_seed_cost=True,curve=curve))
    aggregates=[]
    for block in (32,64,128):
        for order in ('contiguous','hidden-energy'):
            for threshold in ('fixed-8192-seed','oracle-full-max'):
                selected=[row['counterfactual_weight_fraction'] for row in results if row['block_size']==block and row['order']==order and row['threshold']==threshold]
                aggregates.append(dict(block_size=block,order=order,threshold=threshold,
                    weight_fraction_min=min(selected),weight_fraction_median=float(np.median(selected)),weight_fraction_max=max(selected)))
    report=dict(scope='Offline counterfactual partial bounds on captured queries only; no pruned GPU kernel, generated pruning result or latency gain',
        model=capture['model'],batch=batch,head_weight=capture['head_weight'],head_shape=[rows,width],queries=queries,
        normalized_inputs='Actual native GPU RMS output; separate complete projection verified bit-exact',
        full_gpu_logits_sha256=score_digests,all_actual_gpu_logits_within_fp32_error_envelope=True,
        all_partial_bounds_cover_every_actual_gpu_logit=True,actual_winners_preserved=True,
        maximum_bound_violation=max_violation,fp32_gamma=gamma(2*width+64,24),
        maximum_full_logit_error=float(error.max()),maximum_error_budget_fraction=float(np.max(error/budget)),
        oracle_warning='Uses an unavailable full-head winner threshold; optimistic feasibility only',
        omitted_costs=['bound metadata reads','norm/reduction arithmetic','dispatches','noncontiguous gather overhead','session integration'],
        aggregates=aggregates,results=results)
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','full_gpu_logits_sha256')}),flush=True)

if __name__=='__main__':main()
