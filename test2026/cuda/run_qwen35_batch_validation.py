"""Official-model batch/reset diagnostics, with native inputs for a separate oracle.

One resident profile at a time. Trace copies invalidate performance timing.
Uniform and ragged prompts are each repeated after the normal request reset.
"""
import os,sys,json,time,hashlib
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from runtime.cuda_serving import Model
root,image,output=sys.argv[1:4];batch=int(sys.argv[4])
assert batch in (2,4,8) and os.environ.get('GARNET_TRT_SYNC_ALLOCATOR')=='1'
with open(os.path.join(root,'config.json')) as stream:config=json.loads(stream.read())
name='Qwen3.5-2B' if config['text_config']['hidden_size']==2048 else 'Qwen3.5-4B'
profile={'batch':batch,'maxInputTokens':256,'maxOutputTokens':128,'imagePixels':65536,
    'imageGrid':[12,20],'prefillChunk':4,'cudaGraph':True,'fusedDelta':True}
print(json.dumps({'event':'load_begin','model':name,'profile':profile,'process_id':os.getpid(),'unix_seconds':time.time()}),flush=True)
model=Model(root,profile,name);trace=[];embeddings=[];positions=[];rows=[];uniform_outputs=None
original=model.decoder.chunk_hidden
def traced(*args,**kwargs):
    result=original(*args,**kwargs)
    predict=args[3] if len(args)>3 else kwargs.get('predict',True)
    if predict:trace.append(garnet.tensor_to_cpu(model.decoder.head.run_device([model.decoder.last_hidden])).tolist())
    return result
model.decoder.chunk_hidden=traced
prepare=model._prepare
def traced_prepare(*args,**kwargs):
    result=prepare(*args,**kwargs)
    embeddings.append([garnet.tensor_to_cpu(t).tolist() for t in result[0]])
    positions.append(result[1]);return result
model._prepare=traced_prepare
try:
    for style in ['uniform','ragged','mixed_images']:
        expected=None
        for trial in range(2):
            trace.clear();embeddings.clear();positions.clear()
            prompts=['Describe the main objects in this image in one sentence.' if style!='ragged' or b%2==0 else 'Describe this image briefly.' for b in range(batch)]
            paths=[image]*batch
            if style=='mixed_images':
                names=['frame_0.jpg','frame_10050.jpg','frame_10200.jpg','frame_10350.jpg']
                paths=[os.path.join(os.path.dirname(image),names[b%4]) for b in range(batch)]
            result=model.infer_batch(prompts,paths,32)
            assert result['batch']==batch and result['visual_tokens_per_row']==[60]*batch
            assert result['vision_compile_ms']==0 and result['arithmetic']['projection_reduced_precision_reduction'] is False
            slots=[n+s-1 for n,s in zip(result['prompt_tokens'],result['sampled_tokens_including_stop'])]
            assert model.decoder.slots==slots and not model.decoder.poisoned
            signature={k:result[k] for k in ['prompt_token_ids','generated_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids']}
            if expected is None:expected=signature
            else:assert signature==expected,'Reset/replay changed a recorded output or stop decision'
            if style=='uniform':
                uniform_outputs=result['generated_token_ids']
                assert all(row==uniform_outputs[0] for row in uniform_outputs),'Replicated rows differ'
            if style=='mixed_images':
                for b,path in enumerate(paths):
                    if os.path.normcase(os.path.abspath(path))==os.path.normcase(os.path.abspath(image)):
                        assert result['generated_token_ids'][b]==uniform_outputs[b],'Other image rows changed an unchanged row'
            memory=model.memory_info();assert memory['owned_model_bytes'] is not None
            assert all(g['graph_captures']>0 and g['graph_replays']>0 and g['graph_capture_failures']==0 for g in memory['decoder']['graphs'])
            result.update({'style':style,'trial':trial,'process_id':os.getpid(),'decoder_slots':slots,
                'owned_memory':memory,'image':image,'image_sha256':hashlib.sha256(open(image,'rb').read()).hexdigest(),
                'images':paths,'image_sha256_per_row':[hashlib.sha256(open(path,'rb').read()).hexdigest() for path in paths],
                'diagnostic_timing_limitation':'Native hidden/input copies and extra norm calls; not performance evidence'})
            if trial==0:
                result['normalized_hidden_trace']=list(trace);result['prefill_embedding_trace']=list(embeddings)
                result['native_position_trace']=list(positions)
                assert len(trace)==max(result['sampled_tokens_including_stop'])
            rows.append(result)
            with open(output,'w') as stream:stream.write(json.dumps(rows,indent=2))
            print(json.dumps({'event':'case_complete','style':style,'trial':trial,'prompt_tokens':result['prompt_tokens'],
                'sampled_tokens':result['sampled_tokens_including_stop'],'slots':slots,'owned_bytes':memory['owned_model_bytes']}),flush=True)
    print('QWEN35_BATCH_VALIDATION_COMPLETE',name,'batch',batch,'cases',len(rows),flush=True)
finally:model.close()
