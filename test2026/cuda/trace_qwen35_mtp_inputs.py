"""Record all target hidden/input rows for an offline MTP feasibility study.

This does not implement speculative serving. Native host copies perturb timing.
"""
import json,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import garnet
from runtime.cuda_serving import Model
from qwen35_benchmark_workload import batch_inputs,input_hashes
checkpoint,cohort,output=sys.argv[1:4]
index=int(sys.argv[4]) if len(sys.argv)>4 else 8
record=json.load(open(cohort))[index];assert record['batch']==1 and record['matched_batch_inputs']
names,images=batch_inputs(record['dataset'],record['trial'],record['workload_group'],1,True)
assert input_hashes(images)==record['input_sha256']
model=Model(checkpoint,record['profile'],record['model']);captured=[]
original=model.decoder.chunk_hidden;width=model.config['text_config']['hidden_size']
def traced(hidden_tokens,active_tokens,positions,predict=True):
    start=model.decoder.slots[0]
    inputs=[garnet.tensor_to_cpu(t).tolist() for t in hidden_tokens]
    result=original(hidden_tokens,active_tokens,positions,predict)
    hidden=garnet.tensor_to_cpu(model.decoder.hidden).tolist()
    assert not isinstance(hidden[0],list),'Expected flat native tensor list'
    for t,flags in enumerate(active_tokens):
        if flags[0]:captured.append({'slot':start,'position':positions[0][t],
            'input_embedding':inputs[t],'target_hidden_pre_final_norm':hidden[t*width:(t+1)*width]});start+=flags[0]
    return result
model.decoder.chunk_hidden=traced
try:
    result=model.infer_batch(['Describe the main objects in this image in one sentence.'],images,record['request_output_limit'])
    assert result['generated_token_ids']==record['generated_token_ids'],'Instrumented target trajectory changed'
    assert result['prompt_token_ids']==record['prompt_token_ids']
    assert len(captured)==model.decoder.slots[0]
    assert [r['slot'] for r in captured]==list(range(len(captured)))
    data={'source_cohort':cohort,'source_index':index,'profile':record['profile'],'model':record['model'],
        'image_sha256':record['input_sha256'],'prompt_token_ids':result['prompt_token_ids'],
        'generated_token_ids':result['generated_token_ids'],'stop_token_ids':result['stop_token_ids'],
        'rows':captured,'scope':'Offline MTP input study only; complete native-input/target pre-final-norm rows, no speculative serving or performance claim'}
    with open(output,'w') as stream:json.dump(data,stream)
    print('QWEN35_MTP_INPUT_TRACE_COMPLETE','rows',len(captured),'output_matches',True,flush=True)
finally:model.close()
