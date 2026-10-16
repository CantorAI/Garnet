"""Capture real query/score data for head-pruning feasibility, not timing.

Arguments after xlang3: build, source, model, prompt. No alternate model or
approximate inference executes; every reported score comes from the full head.
"""
import sys,json
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
folder=sys.argv[3]
with open(folder+'/config.json') as stream:configuration=json.load(stream)
vl='text_config' in configuration
config=configuration['text_config'] if vl else configuration
if vl:
    from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
else:
    from xModel.qwen.metal_dense.decoder import DenseMetalDecoder as MetalDecoder
tokenizer=garnet.tokenizer();tokenizer.open(folder)
chat='<|im_start|>user\n'+sys.argv[4]+'<|im_end|>\n<|im_start|>assistant\n'
if vl or config['model_type']=='qwen3':chat+='<think>\n\n</think>\n\n'
ids=tokenizer.encode(chat)
stops=[config['eos_token_id'],tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
decoder=None
try:
    decoder=MetalDecoder(folder+'/model.safetensors',config,1,256)
    decoder.enable_prefill(16)
    hidden=[decoder.io.embedding([token]) for token in ids]
    prediction=decoder.prefill_sequence(hidden,[[[i]*3 for i in range(len(ids))]])
    for step in range(8):
        state=decoder.session.hidden()
        if step==0:state=decoder.io.visual_token(state,(len(ids)-1)%16)
        projected=decoder.head.run([state]);scores=projected.tolist()
        assert decoder.io.argmax(projected)==prediction
        print('head_bound_input',json.dumps(dict(model=folder,step=step,
            hidden=state.tolist(),prediction=prediction[0],full_max=max(scores),
            seed_max=max(scores[:8192]),seed_rows=8192,scope='feasibility data; not a speed comparison')),flush=True)
        if prediction[0] in stops:break
        prediction=decoder.step(prediction)
finally:
    if decoder is not None:decoder.close()
    garnet.shutdown()
