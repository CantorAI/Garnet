"""Real full-checkpoint Garnet decoder execution, no image claim."""
import sys,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen35.vl_0_8b.cpu_decoder import CPUDecoder
folder=sys.argv[3]
with open(folder+'/config.json') as stream:config=json.load(stream)['text_config']
batch=int(sys.argv[4]) if len(sys.argv)>4 else 1
tokenizer=garnet.tokenizer();tokenizer.open(folder)
assert tokenizer.encode('<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n')==[248045,846,198,9419,248046,198,248045,74455,198]
start=time.time();model=None
try:
    model=CPUDecoder(folder+'/model.safetensors',config,batch,64)
    print('full decoder compile_seconds',time.time()-start,flush=True)
    # Explicit token sequence avoids claiming an unvalidated tokenizer frontend.
    ids=tokenizer.encode('<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n')
    for token in ids:
        step=time.time();next_ids=model.step([token]*batch)
        print('prefill token',token,'next',next_ids,'seconds',time.time()-step,flush=True)
    for i in range(4):
        print('generated token',next_ids,'decoded',[tokenizer.decode([token]) for token in next_ids],flush=True)
        step=time.time();next_ids=model.step(next_ids)
        print('decode seconds',time.time()-step,'layer_ms',model.last_layer_ms,'head_ms',model.last_head_ms,flush=True)
finally:
    if model is not None:model.close()
    garnet.shutdown()
