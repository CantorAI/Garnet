"""Research-only native single-slot worker; token events occur at synchronized predictions.

Pinned packed INT4 G128 B1/T16/cap4096 profile. HTTP gates required before timings.
"""
import base64,hashlib,json,os,sys,time
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_int4_packed_decoder,prepare_cuda_image
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDAImageEncoder
from test2026.cuda.qwen38_5090.native_worker_io import requests,emit
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.native_http_bridge import validate_body
checkpoint,image_path=sys.argv[1:3]
with open(os.path.join(checkpoint,'config.json'),encoding='utf8') as f:config=json.load(f)
tokenizer=garnet.tokenizer();tokenizer.open(checkpoint)
suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
decoder=None;vision=None
try:
    decoder=create_cuda_int4_packed_decoder(checkpoint,config,4096)
    vision=CUDAImageEncoder(os.path.join(checkpoint,'model.safetensors.index.json'),config['vision_config'],16,16,1)
    decoder.enable_cuda_graph(True);vision.graph.enable_cuda_graph(True)
    memory=decoder.memory_info()
    assert sum(g.get('native_int4_g128_projection_weights',0) for g in memory['graphs'])==496
    emit({'event':'ready','research_only':True,'model':'qwen38-27b-int4','batch':1,'capacity':4096,
          'chunk':16,'prefill_chunk':16,'decoder_layout':'packed_whole','mtp_depth':0,
          'precision':'INT4 G128 W4A16 body; BF16 head and vision; FP32 residual/state',
          'vision_input_policy':'direct','quality_gate':'pending_http','memory':memory})
    for item in requests():
        key=item['id'];body=item['body'];started=time.monotonic()
        try:
            assert body['temperature']==0 and body['chat_template_kwargs']=={'enable_thinking':False}
            validate_body(body,'qwen38-27b-int4')
            limit=body['max_tokens'];assert isinstance(limit,int) and not isinstance(limit,bool) and 2<=limit<=128
            assert 'study_target_only' not in body,'No speculative mode for this profile'
            content=body['messages'][0]['content'];visual=None;image_start=-1;image_hash=None
            decoder.reset()
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
            prediction=decoder.prefill_sequence(hidden,[positions])
            for step in range(limit):
                token=prediction[0];notify(0,token,step+1)
                if token in stops or step+1==limit:break
                prediction=decoder.step([token],positions=[[current]*3]);current+=1
            mode='greedy_target'
            emit({'event':'completed','id':key,'generated_ids':generated,'prompt_ids':ids,'text':stream_state['text'],'image_sha256':image_hash,'generation_mode':mode,'speculative_metrics':speculative_metrics,'finish_reason':'stop' if generated[-1] in stops else 'length','native_elapsed_seconds':time.monotonic()-started})
        except Exception as error:
            emit({'event':'error','id':key,'error':repr(error)});raise
finally:
    if decoder is not None:decoder.close()
    if vision is not None:vision.close()
