"""Pretrained CUDA smoke/measurement runner. Run only one engine at a time."""
import os,sys,json,time
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runtime.cuda_serving import Model
root,image,output=sys.argv[1:4]
if image=='-':image=''
with open(os.path.join(root,'config.json')) as f:config=json.loads(f.read())
name='Qwen3.5-2B' if config['text_config']['hidden_size']==2048 else 'Qwen3.5-4B'
limit=int(sys.argv[4]) if len(sys.argv)>4 else 8
batch=int(sys.argv[5]) if len(sys.argv)>5 else 1
profile={'batch':batch,'maxInputTokens':256,'maxOutputTokens':128,'imagePixels':65536}
profile['prefillChunk']=int(os.environ.get('GARNET_PREFILL_CHUNK','1'))
profile['cudaGraph']=os.environ.get('GARNET_QWEN35_CUDA_GRAPH','0')=='1'
profile['fusedDelta']=os.environ.get('GARNET_QWEN35_FUSED_DELTA','0')=='1'
profile['fusedAttention']=os.environ.get('GARNET_QWEN35_FUSED_ATTENTION','0')=='1'
profile['projectionWorkspaceBytes']=int(os.environ.get('GARNET_QWEN35_PROJECTION_WORKSPACE_BYTES','0'))
if image and os.environ.get('GARNET_PREBUILD_IMAGE_GRID')=='1':profile['imageGrid']=[12,20]
prompt=os.environ.get('GARNET_BENCHMARK_PROMPT','Describe the main objects in this image in one sentence.' if image else 'Explain what you can do in one sentence.')
print(json.dumps({'event':'load_begin','unix_seconds':time.time(),'model':name,'profile':profile}),flush=True)
start=time.monotonic();model=Model(root,profile,name)
trace=[];embedding_trace=[];position_trace=[]
if os.environ.get('GARNET_TRACE_HIDDEN')=='1':
    # Diagnostic only: host copies perturb timing. Never enable in a cohort.
    import garnet
    method='chunk_hidden' if model.decoder.chunk_size>1 else 'step_hidden'
    original=getattr(model.decoder,method)
    def traced(*args,**kwargs):
        result=original(*args,**kwargs)
        predict=args[3] if len(args)>3 else kwargs.get('predict',True)
        if predict:trace.append(garnet.tensor_to_cpu(model.decoder.head.run_device([model.decoder.last_hidden])).tolist())
        return result
    setattr(model.decoder,method,traced)
    original_prepare=model._prepare
    def traced_prepare(*args,**kwargs):
        result=original_prepare(*args,**kwargs)
        embedding_trace.append([garnet.tensor_to_cpu(t).tolist() for t in result[0]])
        position_trace.append(result[1])
        return result
    model._prepare=traced_prepare
print('MODEL_READY',name,'load_seconds',time.monotonic()-start,flush=True)
print(json.dumps({'event':'load_complete','unix_seconds':time.time()}),flush=True)
try:
    rounds=int(os.environ.get('GARNET_BENCHMARK_ROUNDS','1'));results=[]
    for trial in range(rounds):
        trace.clear()
        embedding_trace.clear();position_trace.clear()
        print(json.dumps({'event':'request_begin','unix_seconds':time.time(),'trial':trial}),flush=True)
        prompts=[prompt if b%2==0 else 'Describe this image briefly.' if image else 'Say hello.' for b in range(batch)]
        result=model.infer_batch(prompts,[image]*batch,limit);result['trial']=trial
        result['io_memory']=json.loads(model.decoder.io.memory_json())
        if os.environ.get('GARNET_BENCHMARK_MEMORY')=='1':result['owned_memory']=model.memory_info()
        if os.environ.get('GARNET_TRACE_HIDDEN')=='1':
            result['normalized_hidden_trace']=list(trace)
            result['prefill_embedding_trace']=list(embedding_trace)
            result['native_position_trace']=list(position_trace)
            result['diagnostic_timing_limitation']='CPU hidden copies and extra norm calls; not a performance cohort'
        results.append(result)
        with open(output,'w') as f:f.write(json.dumps(results,indent=2))
        print(json.dumps(result),flush=True)
        print(json.dumps({'event':'request_complete','unix_seconds':time.time(),'trial':trial}),flush=True)
    print('QWEN35_PRETRAINED_CUDA_COMPLETE',name,flush=True)
finally:model.close()
