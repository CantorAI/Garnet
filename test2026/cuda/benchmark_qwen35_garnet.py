"""One resident Garnet model/profile, serial real-image benchmark cohort.

Run engines sequentially. This records raw evidence, never asserts a speedup.
"""
import os,sys,json,time
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runtime.cuda_serving import Model
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from qwen35_benchmark_workload import batch_inputs,input_hashes
root,dataset,output=sys.argv[1:4]
batch=int(sys.argv[4]) if len(sys.argv)>4 else 1
name=sys.argv[5] if len(sys.argv)>5 else 'Qwen3.5-2B'
variants=len(sys.argv)>6 and sys.argv[6]=='variants'
rounds=int(sys.argv[7]) if len(sys.argv)>7 else 7
matched_inputs=os.environ.get('GARNET_MATCHED_BATCH_INPUTS','0')=='1'
assert not matched_inputs or variants
limit=32
profile={'batch':batch,'maxInputTokens':256,'maxOutputTokens':128,'imagePixels':65536,'imageGrid':[12,20]}
context=int(os.environ.get('GARNET_BENCHMARK_CONTEXT','384'))
assert context in (384,512),'Benchmark context must be384 historical or512 matched'
profile['maxInputTokens']=context-profile['maxOutputTokens']
profile['prefillChunk']=int(os.environ.get('GARNET_PREFILL_CHUNK','1'))
profile['decodeChunk']=int(os.environ.get('GARNET_QWEN35_DECODE_CHUNK',str(profile['prefillChunk'])))
profile['sharedProjectionWeights']=os.environ.get('GARNET_QWEN35_SHARED_PROJECTION_WEIGHTS','0')=='1'
profile['packedDecoderProgram']=os.environ.get('GARNET_QWEN35_PACKED_DECODER_PROGRAM','0')=='1'
profile['packPrompt']=os.environ.get('GARNET_QWEN35_PACK_PROMPT','0')=='1'
profile['reuseVerifiedHead']=os.environ.get('GARNET_QWEN35_REUSE_VERIFIED_HEAD','0')=='1'
profile['residentControls']=os.environ.get('GARNET_QWEN35_RESIDENT_CONTROLS','0')=='1'
profile['serialGraphReuse']=os.environ.get('GARNET_QWEN35_SERIAL_GRAPH_REUSE','0')=='1'
profile['serialTargetGraphReuse']=os.environ.get('GARNET_QWEN35_SERIAL_TARGET_GRAPH_REUSE','0')=='1'
profile['deferPackedStateCommit']=os.environ.get('GARNET_QWEN35_DEFER_PACKED_STATE_COMMIT','0')=='1'
profile['nativePackedPrefixCommit']=os.environ.get('GARNET_QWEN35_NATIVE_PACKED_PREFIX_COMMIT','0')=='1'
profile['sharePackedOutputStorage']=os.environ.get('GARNET_QWEN35_SHARED_OUTPUT_STORAGE','0')=='1'
profile['registerDelta128']=os.environ.get('GARNET_QWEN35_REGISTER_DELTA_128','0')=='1'
profile['flatPackedConcat']=os.environ.get('GARNET_QWEN35_FLAT_PACKED_CONCAT','0')=='1'
profile['cudaGraph']=os.environ.get('GARNET_QWEN35_CUDA_GRAPH','0')=='1'
profile['fusedDelta']=os.environ.get('GARNET_QWEN35_FUSED_DELTA','0')=='1'
profile['fusedAttention']=os.environ.get('GARNET_QWEN35_FUSED_ATTENTION','0')=='1'
profile['projectionWorkspaceBytes']=int(os.environ.get('GARNET_QWEN35_PROJECTION_WORKSPACE_BYTES','0'))
profile['speculativeMTP']=int(os.environ.get('GARNET_QWEN35_SPECULATIVE_MTP','0'))
profile['speculativeHeadCandidates']=int(os.environ.get('GARNET_QWEN35_MTP_HEAD_CANDIDATES','0'))
profile['speculativePrefixReplay']=os.environ.get('GARNET_QWEN35_MTP_PREFIX_REPLAY','0')=='1'
images=['frame_0.jpg','frame_10050.jpg','frame_10200.jpg','frame_10350.jpg']
prompt='Describe the main objects in this image in one sentence.'
rows=[]
print(json.dumps({'event':'load_begin','unix_seconds':time.time(),'engine':'garnet','model':name,'profile':profile}),flush=True)
model=Model(root,profile,name)
print(json.dumps({'event':'load_complete','unix_seconds':time.time()}),flush=True)
try:
    actual_whole=all(g.get('qwen35_whole_program_cuda',False) for g in model.decoder.memory_info()['graphs'])
    assert actual_whole==(os.environ.get('GARNET_BENCHMARK_WHOLE_PROGRAM_CUDA')=='1'),'Benchmark build-policy label differs from actual DLL'
    for trial in range(rounds):
        for image_index in range(4):
            names=[images[(image_index+b)%4] for b in range(batch)]
            paths=[os.path.join(dataset,str(trial),n) if variants else os.path.join(dataset,n) for n in names]
            if matched_inputs:names,paths=batch_inputs(dataset,trial,image_index,batch,True)
            hashes=input_hashes(paths)
            print(json.dumps({'event':'request_begin','trial':trial,'image':names,'unix_seconds':time.time()}),flush=True)
            wall=time.monotonic();result=model.infer_batch([prompt]*batch,paths,limit);wall=time.monotonic()-wall
            assert result['vision_compile_ms']==0,result
            assert all(n<=256 for n in result['prompt_tokens']),result
            assert result['visual_tokens_per_row']==[60]*batch,result
            if profile['reuseVerifiedHead']:
                assert result['speculative']['reuse_verified_head']
                assert result['speculative_metrics']['target_head_evaluations']==result['speculative_metrics']['target_verification_calls']
                assert result['speculative_metrics']['target_head_reuses']==result['speculative_metrics']['target_prefix_replays']
            result.update({'engine':'garnet','model':name,'trial':trial,'warmup':trial<2,'images':names,
                'native_build_policy':{'zero_aux_streams_requested':os.environ.get('GARNET_TRT_ZERO_AUX_STREAMS')=='1',
                    'whole_program_cuda_requested':os.environ.get('GARNET_BENCHMARK_WHOLE_PROGRAM_CUDA')=='1',
                    'reuse_static_bindings':os.environ.get('GARNET_TRT_REUSE_STATIC_BINDINGS')=='1',
                    'sync_temporary_allocator':os.environ.get('GARNET_TRT_SYNC_TEMPORARY_ALLOCATOR')=='1'},
                'wall_seconds':wall,'request_output_limit':limit,'process_id':os.getpid(),
                'dataset':dataset,'variant_round':trial if variants else None,
                'input_sha256':hashes,'workload_group':image_index,'matched_batch_inputs':matched_inputs,
                'timing_scope':'Model.infer_batch: reset, JPEG/vision, tokenization, prefill/decode and output detokenization; load excluded'})
            result['io_memory']=json.loads(model.decoder.io.memory_json())
            if os.environ.get('GARNET_BENCHMARK_MEMORY')=='1':
                result['owned_memory']=model.memory_info()
                if profile['cudaGraph'] and trial>=2:
                    assert all(g['graph_captures']>0 and g['graph_replays']>0 and g['graph_capture_failures']==0 for g in result['owned_memory']['decoder']['graphs'])
            rows.append(result)
            with open(output,'w') as f:f.write(json.dumps(rows,indent=2))
            print(json.dumps({'event':'request_complete','trial':trial,'images':names,'unix_seconds':time.time(),
                'wall_seconds':wall,'generated_tokens':result['generated_tokens']}),flush=True)
    print('QWEN35_GARNET_BENCHMARK_COMPLETE',name,'batch',batch,'requests',len(rows),flush=True)
finally:model.close()
