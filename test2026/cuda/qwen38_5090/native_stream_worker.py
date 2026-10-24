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
decoder=None;vision=None
try:
    vision=create_cuda_vision(checkpoint,config,16,16)
    decoder=create_cuda_decoder(checkpoint,config,4096,chunk_size=16,fused_delta=True,fused_attention=True)
    decoder.enable_cuda_graph(True);vision.graph.enable_cuda_graph(True)
    emit({'event':'ready','research_only':True,'batch':1,'capacity':4096,'chunk':16,'precision':'mixed NVFP4/FP8; BF16 vision','exact_vlm_parity_gate':'open'})
    for item in requests():
        key=item['id'];body=item['body'];started=time.monotonic()
        try:
            assert body['temperature']==0 and body['chat_template_kwargs']=={'enable_thinking':False}
            assert body['model']=='qwen38-27b-nvfp4' and len(body['messages'])==1 and body['messages'][0]['role']=='user'
            limit=body['max_tokens'];assert isinstance(limit,int) and not isinstance(limit,bool) and 2<=limit<=1024
            content=body['messages'][0]['content'];visual=None;image_start=-1;image_hash=None
            decoder.reset();io=decoder.io
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
            prediction=decoder.prefill_sequence(hidden,[positions]);generated=[]
            for step in range(limit):
                token=prediction[0];now=time.monotonic();generated.append(token);stop=token in stops
                text=tokenizer.decode([t for t in generated if t not in stops])
                emit({'event':'token','id':key,'token_id':token,'sampled_tokens':len(generated),'prompt_tokens':len(ids),'text':text,'stop':stop,'native_monotonic_seconds':now})
                if stop or step+1==limit:break
                prediction=decoder.step([token],positions=[[current]*3]);current+=1
            emit({'event':'completed','id':key,'generated_ids':generated,'prompt_ids':ids,'text':text,'image_sha256':image_hash,'finish_reason':'stop' if generated[-1] in stops else 'length','native_elapsed_seconds':time.monotonic()-started})
        except Exception as error:
            emit({'event':'error','id':key,'error':repr(error)});raise
finally:
    if decoder is not None:decoder.close()
    if vision is not None:vision.close()
