"""Trace only changed batch1 benchmark cases, in one resident model session."""
import hashlib,json,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from runtime.cuda_serving import Model
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from qwen35_benchmark_workload import batch_inputs
checkpoint,candidate_path,baseline_path,output=sys.argv[1:5]
candidate=json.load(open(candidate_path));baseline=json.load(open(baseline_path))
def key(r):return r['trial'],tuple(r['images'])
prior={key(r):r for r in baseline}
changed=[r for r in candidate if r['generated_token_ids']!=prior[key(r)]['generated_token_ids']]
assert changed and all(r['batch']==1 for r in changed)
model=Model(checkpoint,candidate[0]['profile'],candidate[0]['model'])
trace=[];embeddings=[];positions=[];rows=[]
original=model.decoder.chunk_hidden
def traced(*args,**kwargs):
    result=original(*args,**kwargs)
    if args[3] if len(args)>3 else kwargs.get('predict',True):
        trace.append(garnet.tensor_to_cpu(model.decoder.head.run_device([model.decoder.last_hidden])).tolist())
    return result
model.decoder.chunk_hidden=traced
prepare=model._prepare
def traced_prepare(*args,**kwargs):
    result=prepare(*args,**kwargs)
    embeddings.append([garnet.tensor_to_cpu(t).tolist() for t in result[0]])
    positions.append(result[1]);return result
model._prepare=traced_prepare
try:
    for r in changed:
        trace.clear();embeddings.clear();positions.clear()
        if r.get('matched_batch_inputs'):
            image=batch_inputs(r['dataset'],r['trial'],r['workload_group'],1,True)[1][0]
        else:image=os.path.join(r['dataset'],str(r['variant_round']),r['images'][0])
        prompt='Describe the main objects in this image in one sentence.'
        actual=model.infer_batch([prompt],[image],r['request_output_limit'])
        actual.update({'style':'changed_trajectory','trial':r['trial'],'images':r['images'],
            'image':image,'image_sha256':hashlib.sha256(open(image,'rb').read()).hexdigest(),
            'normalized_hidden_trace':list(trace),'prefill_embedding_trace':list(embeddings),
            'native_position_trace':list(positions),'owned_memory':model.memory_info(),
            'matches_original_candidate_tokens':actual['generated_token_ids']==r['generated_token_ids'],
            'diagnostic_timing_limitation':'Host traces; not performance evidence'})
        rows.append(actual)
        with open(output,'w') as stream:json.dump(rows,stream,indent=2)
        print('CHANGED_TRAJECTORY_TRACED',r['trial'],r['images'],actual['matches_original_candidate_tokens'],flush=True)
    print('QWEN35_CHANGED_TRAJECTORIES_COMPLETE',len(rows),flush=True)
finally:model.close()
