"""One resident Garnet profile with explicit phases for external memory sampling.

Diagnostic only: host polling and deliberate holds invalidate latency claims.
Run exactly one model engine at a time. Close remains observable before exit.
"""
import os,sys,json,time
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runtime.cuda_serving import Model
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from qwen35_benchmark_workload import batch_inputs,input_hashes
root,dataset,output=sys.argv[1:4]
batch=int(sys.argv[4]) if len(sys.argv)>4 else 1
with open(os.path.join(root,'config.json')) as stream:config=json.loads(stream.read())
name='Qwen3.5-2B' if config['text_config']['hidden_size']==2048 else 'Qwen3.5-4B'
profile={'batch':batch,'maxInputTokens':256,'maxOutputTokens':128,'imagePixels':65536,
    'imageGrid':[12,20],'prefillChunk':4,'cudaGraph':True,'fusedDelta':True}
context=int(os.environ.get('GARNET_BENCHMARK_CONTEXT','384'))
assert context in (384,512)
profile['maxInputTokens']=context-profile['maxOutputTokens']
profile['fusedAttention']=os.environ.get('GARNET_QWEN35_FUSED_ATTENTION','0')=='1'
profile['prefillChunk']=int(os.environ.get('GARNET_PREFILL_CHUNK','4'))
profile['decodeChunk']=int(os.environ.get('GARNET_QWEN35_DECODE_CHUNK',str(profile['prefillChunk'])))
profile['sharedProjectionWeights']=os.environ.get('GARNET_QWEN35_SHARED_PROJECTION_WEIGHTS','0')=='1'
profile['packPrompt']=os.environ.get('GARNET_QWEN35_PACK_PROMPT','0')=='1'
profile['packedDecoderProgram']=os.environ.get('GARNET_QWEN35_PACKED_DECODER_PROGRAM','0')=='1'
profile['residentControls']=os.environ.get('GARNET_QWEN35_RESIDENT_CONTROLS','0')=='1'
profile['serialGraphReuse']=os.environ.get('GARNET_QWEN35_SERIAL_GRAPH_REUSE','0')=='1'
profile['serialTargetGraphReuse']=os.environ.get('GARNET_QWEN35_SERIAL_TARGET_GRAPH_REUSE','0')=='1'
profile['deferPackedStateCommit']=os.environ.get('GARNET_QWEN35_DEFER_PACKED_STATE_COMMIT','0')=='1'
profile['nativePackedPrefixCommit']=os.environ.get('GARNET_QWEN35_NATIVE_PACKED_PREFIX_COMMIT','0')=='1'
profile['sharePackedOutputStorage']=os.environ.get('GARNET_QWEN35_SHARED_OUTPUT_STORAGE','0')=='1'
profile['registerDelta128']=os.environ.get('GARNET_QWEN35_REGISTER_DELTA_128','0')=='1'
profile['flatPackedConcat']=os.environ.get('GARNET_QWEN35_FLAT_PACKED_CONCAT','0')=='1'
profile['speculativeMTP']=int(os.environ.get('GARNET_QWEN35_SPECULATIVE_MTP','0'))
profile['speculativeHeadCandidates']=int(os.environ.get('GARNET_QWEN35_MTP_HEAD_CANDIDATES','0'))
profile['speculativePrefixReplay']=os.environ.get('GARNET_QWEN35_MTP_PREFIX_REPLAY','0')=='1'
profile['projectionWorkspaceBytes']=int(os.environ.get('GARNET_QWEN35_PROJECTION_WORKSPACE_BYTES','0'))
events=[]
def emit(event,**values):
    record=dict(event=event,unix_seconds=time.time(),process_id=os.getpid(),model=name,profile=profile,
        native_build_policy={'zero_aux_streams_requested':os.environ.get('GARNET_TRT_ZERO_AUX_STREAMS')=='1',
            'whole_program_cuda_requested':os.environ.get('GARNET_BENCHMARK_WHOLE_PROGRAM_CUDA')=='1',
            'reuse_static_bindings':os.environ.get('GARNET_TRT_REUSE_STATIC_BINDINGS')=='1',
            'sync_temporary_allocator':os.environ.get('GARNET_TRT_SYNC_TEMPORARY_ALLOCATOR')=='1'},**values)
    events.append(record)
    with open(output,'w') as stream:stream.write(json.dumps(events,indent=2))
    print(json.dumps({'event':event,'unix_seconds':record['unix_seconds'],'process_id':record['process_id']}),flush=True)
images=['frame_0.jpg','frame_10050.jpg','frame_10200.jpg','frame_10350.jpg']
prompt='Describe the main objects in this image in one sentence.'
matched=os.environ.get('GARNET_MATCHED_BATCH_INPUTS','0')=='1'
def infer(trial):
    if matched:names,paths=batch_inputs(dataset,trial//4,trial%4,batch,True)
    else:
        names=[images[(trial+b)%4] for b in range(batch)]
        paths=[os.path.join(dataset,name) for name in names]
    result=model.infer_batch([prompt]*batch,paths,32)
    return result,{'images':names,'input_sha256':input_hashes(paths)}
emit('load_begin',scope='Memory diagnostic; timing is not primary benchmark evidence')
model=Model(root,profile,name)
emit('loaded',owned_memory=model.memory_info())
try:
    actual_whole=all(g.get('qwen35_whole_program_cuda',False) for g in model.decoder.memory_info()['graphs'])
    assert actual_whole==(os.environ.get('GARNET_BENCHMARK_WHOLE_PROGRAM_CUDA')=='1'),'Memory build-policy label differs from actual DLL'
    for trial in range(2):
        result,inputs=infer(trial)
        emit('warmup_complete',trial=trial,sampled_tokens=result['sampled_tokens_including_stop'],**inputs)
    emit('steady_before_begin',owned_memory=model.memory_info());time.sleep(8);emit('steady_before_end')
    emit('inference_begin')
    for trial in range(4):
        result,inputs=infer(trial+2)
        emit('request_complete',trial=trial+2,sampled_tokens=result['sampled_tokens_including_stop'],owned_memory=model.memory_info(),**inputs)
    emit('inference_end')
    emit('steady_after_begin',owned_memory=model.memory_info());time.sleep(8);emit('steady_after_end')
finally:
    model.close();emit('closed_begin');time.sleep(8);emit('closed_end')
emit('complete')
print('QWEN35_GARNET_MEMORY_SESSION_COMPLETE',name,'batch',batch,flush=True)
