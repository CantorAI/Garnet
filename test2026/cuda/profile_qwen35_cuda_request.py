"""One warmed real-model request under CUDA profiler API; diagnostic only.

Run with Nsight Systems capture-range=cudaProfilerApi. Initialization/warmup
are excluded. Trace overhead means these times cannot establish speedup.
"""
import ctypes,hashlib,json,os,sys,time
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runtime.cuda_serving import Model
root,image,output=sys.argv[1:4]
batch=int(sys.argv[4]) if len(sys.argv)>4 else 1
with open(os.path.join(root,'config.json')) as stream:config=json.load(stream)
name='Qwen3.5-2B' if config['text_config']['hidden_size']==2048 else 'Qwen3.5-4B'
profile={'batch':batch,'maxInputTokens':384,'maxOutputTokens':128,'imagePixels':65536,
    'imageGrid':[12,20],'prefillChunk':int(os.environ.get('GARNET_PREFILL_CHUNK','4')),'cudaGraph':True,'fusedDelta':True,
    'fusedAttention':os.environ.get('GARNET_QWEN35_FUSED_ATTENTION','0')=='1'}
profile.update({'decodeChunk':int(os.environ.get('GARNET_QWEN35_DECODE_CHUNK',str(profile['prefillChunk']))),
    'sharedProjectionWeights':os.environ.get('GARNET_QWEN35_SHARED_PROJECTION_WEIGHTS','0')=='1',
    'packedDecoderProgram':os.environ.get('GARNET_QWEN35_PACKED_DECODER_PROGRAM','0')=='1',
    'residentControls':os.environ.get('GARNET_QWEN35_RESIDENT_CONTROLS','0')=='1',
    'serialGraphReuse':os.environ.get('GARNET_QWEN35_SERIAL_GRAPH_REUSE','0')=='1',
    'serialTargetGraphReuse':os.environ.get('GARNET_QWEN35_SERIAL_TARGET_GRAPH_REUSE','0')=='1',
    'deferPackedStateCommit':os.environ.get('GARNET_QWEN35_DEFER_PACKED_STATE_COMMIT','0')=='1',
    'nativePackedPrefixCommit':os.environ.get('GARNET_QWEN35_NATIVE_PACKED_PREFIX_COMMIT','0')=='1',
    'sharePackedOutputStorage':os.environ.get('GARNET_QWEN35_SHARED_OUTPUT_STORAGE','0')=='1',
    'packPrompt':os.environ.get('GARNET_QWEN35_PACK_PROMPT','0')=='1',
    'registerDelta128':os.environ.get('GARNET_QWEN35_REGISTER_DELTA_128','0')=='1',
    'flatPackedConcat':os.environ.get('GARNET_QWEN35_FLAT_PACKED_CONCAT','0')=='1',
    'speculativeMTP':int(os.environ.get('GARNET_QWEN35_SPECULATIVE_MTP','0')),
    'speculativeHeadCandidates':int(os.environ.get('GARNET_QWEN35_MTP_HEAD_CANDIDATES','0')),
    'speculativePrefixReplay':os.environ.get('GARNET_QWEN35_MTP_PREFIX_REPLAY','0')=='1',
    'projectionWorkspaceBytes':int(os.environ.get('GARNET_QWEN35_PROJECTION_WORKSPACE_BYTES','0'))})
runtime=ctypes.CDLL('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64/cudart64_13.dll')
prompt='Describe the main objects in this image in one sentence.'
print(json.dumps({'event':'profile_load_begin','model':name,'profile':profile,'unix_seconds':time.time()}),flush=True)
model=Model(root,profile,name);rows=[]
try:
    for trial in range(3):
        capture=trial==2
        if capture:assert runtime.cudaProfilerStart()==0,'cudaProfilerStart failed'
        try:result=model.infer_batch([prompt]*batch,[image]*batch,32)
        finally:
            if capture:assert runtime.cudaProfilerStop()==0,'cudaProfilerStop failed'
        result.update({'trial':trial,'profiler_capture':capture,'image':image,
            'image_sha256':hashlib.sha256(open(image,'rb').read()).hexdigest(),
            'scope':'Instrumented single repeated image; diagnostic, not matched cohort or quality acceptance'})
        result['owned_memory']=model.memory_info();rows.append(result)
        with open(output,'w') as stream:json.dump(rows,stream,indent=2)
        print(json.dumps({'event':'profile_request_complete','trial':trial,'capture':capture,
            'total_seconds':result['total_seconds'],'sampled_tokens':result['sampled_tokens_including_stop']}),flush=True)
    assert all(r['generated_token_ids']==rows[0]['generated_token_ids'] for r in rows),'Reset/replay mismatch'
    assert all(g['graph_replays']>0 and g['graph_capture_failures']==0 for g in rows[-1]['owned_memory']['decoder']['graphs'])
    print('QWEN35_CUDA_REQUEST_PROFILE_COMPLETE',flush=True)
finally:model.close()
