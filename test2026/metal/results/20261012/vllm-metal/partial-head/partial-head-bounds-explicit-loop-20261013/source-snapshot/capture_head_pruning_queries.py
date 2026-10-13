"""Capture actual normalized GPU head inputs/logits; not a speed benchmark.

Arguments after xlang3: build, source, model, case JSON, output directory.
The ordinary complete head remains responsible for every generated token.
"""
import sys,json,math
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
folder=sys.argv[3];output=sys.argv[5]
with open(sys.argv[4]) as stream:case=json.load(stream)
with open(folder+'/config.json') as stream:configuration=json.load(stream)
vl='text_config' in configuration;config=configuration['text_config'] if vl else configuration
batch=case['batch'];width=config['hidden_size'];checkpoint=folder+'/model.safetensors'
if vl:
    from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
    from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
else:
    from xModel.qwen.metal_dense.decoder import DenseMetalDecoder as MetalDecoder
tokenizer=garnet.tokenizer();tokenizer.open(folder)
stops=[config['eos_token_id'],tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
decoder=None;vision=None;normalize=None;project=None;queries=[]
try:
    decoder=MetalDecoder(checkpoint,config,batch,case['capacity']);decoder.enable_prefill(case['chunk'])
    symbol=tensor.input('head_probe_hidden',shape=[batch,width],dtype=tensor.float32)
    norm_name='model.language_model.norm.weight' if vl else 'model.norm.weight'
    normalized=symbol*T.unary_op('qwen35_rms_norm' if vl else 'qwen_text_norm',weight_name=norm_name,eps=config['rms_norm_eps'])
    normalize=garnet.compiled_graph();normalize.build(normalized,[symbol],checkpoint)
    normalized_symbol=tensor.input('head_probe_normalized',shape=[batch,width],dtype=tensor.float32)
    head_weight='model.language_model.embed_tokens.weight' if vl else 'model.embed_tokens.weight' if config.get('tie_word_embeddings',False) else 'lm_head.weight'
    projected=normalized_symbol*T.unary_op('qwen_text_head',weight_name=head_weight,bf16_weight=1)
    project=garnet.compiled_graph();project.build(projected,[normalized_symbol],checkpoint)
    if vl:
        metadata=case['image_metadata'];gh,gw=metadata['grid_thw'][1:]
        merge=configuration['vision_config']['spatial_merge_size'];mh,mw=gh//merge,gw//merge
        vision=MetalImageEncoder(checkpoint,configuration['vision_config'],gh,gw,batch)
        embeddings=vision.encode_file(metadata['patch_file'])
        prefix=tokenizer.encode('<|im_start|>user\n<|vision_start|>')
        suffix=tokenizer.encode('<|vision_end|>'+case['prompt']+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
        ids=prefix+[tokenizer.token_id('<|image_pad|>')]*(mh*mw)+suffix
        hidden=[];positions=[];current=0
        for token in prefix:
            hidden.append(decoder.io.embedding([token]*batch));positions.append([current]*3);current+=1
        image_base=current
        for i in range(mh*mw):
            hidden.append(decoder.io.visual_token(embeddings,i));positions.append([image_base,image_base+i//mw,image_base+i%mw])
        current=image_base+max(mh,mw)
        for token in suffix:
            hidden.append(decoder.io.embedding([token]*batch));positions.append([current]*3);current+=1
    else:
        chat='<|im_start|>user\n'+case['prompt']+'<|im_end|>\n<|im_start|>assistant\n'
        if config['model_type']=='qwen3':chat+='<think>\n\n</think>\n\n'
        ids=tokenizer.encode(chat);hidden=[decoder.io.embedding([token]*batch) for token in ids]
        positions=[[i]*3 for i in range(len(ids))];current=len(ids)
    prediction=decoder.prefill_sequence(hidden,[positions for row in range(batch)])
    generated=[[] for row in range(batch)];active=[1]*batch
    for step in range(case['max_tokens']):
        if step<8:
            state=decoder.session.hidden()
            if step==0:state=decoder.io.visual_token(state,(len(ids)-1)%case['chunk'])
            value=normalize.run([state]);reference=decoder.head.run([state]);separate=project.run([value])
            scores=reference.tolist();actual=separate.tolist()
            assert actual==scores,'Separate normalized head must be bit-exact'
            for x,y in zip(actual,scores):
                if x==0:assert math.copysign(1.0,x)==math.copysign(1.0,y),'Signed zero differs'
            assert decoder.io.argmax(reference)==prediction,'Capture changed the selected token'
            logits_path=output+'/step-'+str(step)+'.scores.json'
            with open(logits_path,'w') as stream:stream.write(json.dumps(scores))
            queries.append(dict(step=step,normalized=value.tolist(),prediction=prediction,active=list(active),
                scores_file=logits_path,separate_projection_bits_exact=True))
        for row in range(batch):
            if active[row]:generated[row].append(prediction[row])
            if prediction[row] in stops:active[row]=0
        if not any(active) or step==case['max_tokens']-1:break
        prediction=decoder.step(prediction,active=active,positions=[[current]*3 for row in range(batch)] if vl else None)
        current+=1
    record=dict(model=folder,case=case,prompt_token_ids=ids,stop_token_ids=stops,
        token_ids=generated,text=[tokenizer.decode(tokens) for tokens in generated],finished=[not flag for flag in active],
        head_weight=head_weight,width=width,batch=batch,queries=queries,
        scope='Actual complete native inference with diagnostic re-projections; no pruning or speed measurement')
    with open(output+'/queries.json','w') as stream:stream.write(json.dumps(record))
    print('head_pruning_capture',json.dumps(dict(queries=len(queries),batch=batch,token_ids=generated,finished=record['finished'])),flush=True)
finally:
    if project is not None:project.close()
    if normalize is not None:normalize.close()
    if vision is not None:vision.close()
    if decoder is not None:decoder.close()
    garnet.shutdown()
