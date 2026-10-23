"""Sequential graph-off/graph-on diagnostic on one loaded model/weight set.

Checks exact per-mode replay, records allocator scopes and actual CUDA graph
capture/replay counts. Timing is diagnostic until a full cohort validates it.
"""
import os,sys,json,time
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runtime.cuda_serving import Model
root,image,output=sys.argv[1:4]
chunk=int(sys.argv[4]) if len(sys.argv)>4 else 4
profile={'batch':1,'maxInputTokens':256,'maxOutputTokens':128,'imagePixels':65536,'imageGrid':[12,20],'prefillChunk':chunk,'cudaGraph':False}
profile['fusedDelta']='fused' in sys.argv[5:]
profile['fusedAttention']='attention' in sys.argv[5:]
profile['projectionWorkspaceBytes']=16384 if 'workspace16k' in sys.argv[5:] else 0
context=int(os.environ.get('GARNET_BENCHMARK_CONTEXT','384'))
assert context in (384,512)
profile['maxInputTokens']=context-profile['maxOutputTokens']
model=Model(root,profile,'Qwen3.5-2B');rows=[]
try:
    for graph in [False,True]:
        model.decoder.enable_cuda_graph(graph);model.profile['cudaGraph']=graph
        for trial in range(4):
            result=model.infer('Describe the main objects in this image in one sentence.',image,32)
            result['graph_mode']=graph;result['trial']=trial;result['warmup']=trial<2
            result['profile']=dict(model.profile);result['memory']=model.memory_info()
            rows.append(result)
            with open(output,'w') as f:f.write(json.dumps(rows,indent=2))
            print(json.dumps({'graph':graph,'trial':trial,'total_seconds':result['total_seconds'],
                'prefill_seconds':result['prefill_seconds'],'decode_seconds':result['decode_seconds'],
                'sampled':result['sampled_tokens_including_stop'],'owned_model_bytes':result['memory']['owned_model_bytes']}),flush=True)
        phase=rows[-4:]
        assert all(r['generated_token_ids']==phase[0]['generated_token_ids'] for r in phase)
        if graph:
            assert rows[0]['generated_token_ids']==phase[0]['generated_token_ids']
            counts=phase[-1]['memory']['decoder']['graphs']
            assert all(g['graph_captures']>0 and g['graph_replays']>0 and not g['graph_capture_failures'] for g in counts),counts
    print('QWEN35_CHUNK_GRAPH_PROBE_COMPLETE','chunk',chunk,flush=True)
finally:model.close()
