"""Actual pinned image-plus-text execution through Garnet's CPU package."""
import sys,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen35.vl_0_8b.vision_encoder import CPUImageEncoder
from xModel.qwen35.vl_0_8b.cpu_decoder import CPUDecoder
folder=sys.argv[3]
with open(folder+'/config.json') as stream:config=json.load(stream)
with open(sys.argv[4]) as stream:metadata=json.load(stream)
batch=metadata['shape'][0];gh,gw=metadata['grid_thw'][1:];merge=config['vision_config']['spatial_merge_size']
vision=None;decoder=None;start=time.time()
try:
    vision=CPUImageEncoder(folder+'/model.safetensors',config['vision_config'],gh,gw,batch)
    print('vision_compile_ms',vision.graph.compile_ms,flush=True)
    image_embeddings=vision.encode_file(metadata['patch_file'])
    print('vision_inference_ms',vision.graph.last_inference_ms,'image_sha256',metadata['image_sha256'],flush=True)
    # Release vision weights before decoder loading on this memory-limited Pi.
    # The encoded image tensor owns its storage independently of the graph.
    vision.close();vision=None
    decoder=CPUDecoder(folder+'/model.safetensors',config['text_config'],batch,256)
    tokenizer=garnet.tokenizer();tokenizer.open(folder)
    print('initialization_seconds',time.time()-start,flush=True)
    prefix=tokenizer.encode('<|im_start|>user\n<|vision_start|>')
    suffix=tokenizer.encode('<|vision_end|>Describe the main objects in this image in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    current=0;prefill=time.time()
    for token in prefix:
        decoder.step([token]*batch,positions=[[current]*3 for b in range(batch)],predict=False);current+=1
    image_base=current;mh,mw=gh//merge,gw//merge
    for i in range(mh*mw):
        position=[image_base,image_base+i//mw,image_base+i%mw]
        decoder.step_hidden(decoder.io.visual_token(image_embeddings,i),positions=[position for b in range(batch)],predict=False)
    current=image_base+max(mh,mw)
    for i,token in enumerate(suffix):
        next_ids=decoder.step([token]*batch,positions=[[current]*3 for b in range(batch)],predict=i==len(suffix)-1);current+=1
    print('prefill_seconds',time.time()-prefill,'prompt_tokens',decoder.slots,
        'mrope_next_position',current,'ttft_from_start_seconds',time.time()-start,flush=True)
    outputs=[[] for b in range(batch)];active=[1]*batch;durations=[]
    for i in range(32):
        for b in range(batch):
            if active[b]:outputs[b].append(next_ids[b])
            if next_ids[b] in (248044,248046):active[b]=0
        print('image_generated_tokens',next_ids,'text',[tokenizer.decode(ids) for ids in outputs],flush=True)
        if not any(active):break
        step=time.time();next_ids=decoder.step(next_ids,active=active,positions=[[current]*3 for b in range(batch)]);current+=1
        durations.append(time.time()-step)
    print('image_result',dict(text=[tokenizer.decode(ids) for ids in outputs],token_ids=outputs,
        decode_step_seconds=durations,batch=batch,image_sha256=metadata['image_sha256']),flush=True)
finally:
    if vision is not None:vision.close()
    if decoder is not None:decoder.close()
    garnet.shutdown()
