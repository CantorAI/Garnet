"""Research-only native single-slot worker; token events occur at synchronized predictions.

Uses current Garnet factories. This does not admit Qwen3.8 into general ServeModel.
"""
import base64,hashlib,json,os,sys,time
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_decoder,create_cuda_vision,prepare_cuda_image
from test2026.cuda.qwen38_5090.native_worker_io import requests,emit
from test2026.cuda.qwen38_5090.native_prompt import image_positions
checkpoint,image_path=sys.argv[1:3]
with open(os.path.join(checkpoint,'config.json')) as f:config=json.load(f)
tokenizer=garnet.tokenizer();tokenizer.open(checkpoint)
suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
decoder=None;vision=None;speculative=None
try:
    vision=create_cuda_vision(checkpoint,config,16,16)
    packed_mode=os.environ.get('GARNET_STUDY_PACKED','0')=='1'
    mtp_depth=int(os.environ.get('GARNET_STUDY_MTP_DEPTH','0'))
    if mtp_depth not in (0,1,3,7) or mtp_depth and not packed_mode:raise ValueError('MTP study requires packed mode and depth1/3/7')
    if packed_mode:
        from xModel.qwen38.vl_27b.model import create_cuda_packed_decoder
        decoder=create_cuda_packed_decoder(checkpoint,config,4096,retain_linear_inputs=mtp_depth>0)
    else:decoder=create_cuda_decoder(checkpoint,config,4096,chunk_size=16,fused_delta=True,fused_attention=True)
    decoder.enable_cuda_graph(True);vision.graph.enable_cuda_graph(True)
    if mtp_depth:
        from xModel.qwen38.vl_27b.model import create_cuda_mtp_session
        speculative=create_cuda_mtp_session(decoder,checkpoint,config,stops,mtp_depth)
    emit({'event':'ready','research_only':True,'batch':1,'capacity':4096,'chunk':16,'mtp_depth':mtp_depth,'decoder_layout':'packed_whole' if packed_mode else 'per_layer','precision':'mixed NVFP4/FP8; BF16 vision','exact_vlm_parity_gate':'open'})
    for item in requests():
        key=item['id'];body=item['body'];started=time.monotonic()
        try:
            assert body['temperature']==0 and body['chat_template_kwargs']=={'enable_thinking':False}
            assert body['model']=='qwen38-27b-nvfp4' and len(body['messages'])==1 and body['messages'][0]['role']=='user'
            limit=body['max_tokens'];assert isinstance(limit,int) and not isinstance(limit,bool) and 2<=limit<=1024
            target_only=body.get('study_target_only',False)
            assert isinstance(target_only,bool) and (not target_only or speculative is not None)
            content=body['messages'][0]['content'];visual=None;image_start=-1;image_hash=None
            if speculative is not None:speculative.reset()
            else:decoder.reset()
            io=decoder.io
            if isinstance(content,str):
                ids=tokenizer.encode('<|im_start|>user\n'+content+suffix)
                positions=[[i]*3 for i in range(len(ids))];current=len(ids)
            else:
                assert len(content)==2 and content[0]['type']=='image_url' and content[1]['type']=='text'
                url=content[0]['image_url']['url'];prefix='data:image/jpeg;base64,';assert url.startswith(prefix)
                data=base64.b64decode(url[len(prefix):],validate=True);assert data[:2]==b'\xff\xd8'
                image_hash=hashlib.sha256(data).hexdigest()
                with open(image_path,'wb') as f:f.write(data)
                prepared=prepare_cuda_image(io,checkpoint,image_path);assert [prepared[1],prepared[2]]==[16,16]
                visual=vision.encode_patches(prepared[0]);leading=tokenizer.encode('<|im_start|>user\n<|vision_start|>');image_start=len(leading)
                ids=leading+[config['image_token_id']]*64+tokenizer.encode('<|vision_end|>'+content[1]['text']+suffix)
                positions,current=image_positions(len(ids),image_start,64,8)
            assert len(ids)+limit<=4096
            embeddings=io.embedding(ids)
            hidden=[io.visual_token(visual,i-image_start) if visual is not None and image_start<=i<image_start+64 else io.row(embeddings,i) for i in range(len(ids))]
            stream_state={'ids':[],'text':''};generated=stream_state['ids'];speculative_metrics=None
            def notify(row,token,count):
                assert row==0
                # The first target token is already streamed during prefill.
                if count==1 and len(generated)==1:
                    assert generated[0]==token;return
                assert count==len(generated)+1
                now=time.monotonic();generated.append(token)
                stream_state['text']=tokenizer.decode([t for t in generated if t not in stops])
                emit({'event':'token','id':key,'token_id':token,'sampled_tokens':count,'prompt_tokens':len(ids),'text':stream_state['text'],'stop':token in stops,'native_monotonic_seconds':now})
            if speculative is not None and not target_only:
                speculative.prefill(hidden,[positions],[len(ids)],[current],on_first_token=lambda prediction:notify(0,prediction[0],1))
                result=speculative.generate(limit,on_token=notify)
                expected=list(result['generated_token_ids'][0])
                if result['stop_token_ids'][0] is not None:expected.append(result['stop_token_ids'][0])
                assert expected==generated and result['sampled_tokens_including_stop'][0]==len(generated)
                speculative_metrics=result['speculative_metrics']
            else:
                prediction=decoder.prefill_sequence(hidden,[positions])
                for step in range(limit):
                    token=prediction[0];notify(0,token,step+1)
                    if token in stops or step+1==limit:break
                    prediction=decoder.step([token],positions=[[current]*3]);current+=1
            mode='verified_mtp' if speculative is not None and not target_only else 'greedy_target'
            emit({'event':'completed','id':key,'generated_ids':generated,'prompt_ids':ids,'text':stream_state['text'],'image_sha256':image_hash,'generation_mode':mode,'speculative_metrics':speculative_metrics,'finish_reason':'stop' if generated[-1] in stops else 'length','native_elapsed_seconds':time.monotonic()-started})
        except Exception as error:
            emit({'event':'error','id':key,'error':repr(error)});raise
finally:
    if speculative is not None:speculative.close()
    if decoder is not None:decoder.close()
    if vision is not None:vision.close()
