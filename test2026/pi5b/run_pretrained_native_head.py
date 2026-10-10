import sys,json,math,time
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
with open(sys.argv[1]) as stream:f=json.load(stream)
io=garnet.tensor_io();io.open(f['checkpoint'])
input=tensor.tensor(f['input'],shape=f['shape'],dtype=tensor.float32)
start=time.time();output=io.logits(input);duration=time.time()-start
assert io.argmax(output)==f['argmax']
for sample in f['samples']:
    values=io.slice(output,sample['start'],[2,64]).tolist()
    for actual,wanted in zip(values,sample['expected']):
        assert math.isfinite(actual) and abs(actual-wanted)<=5e-5+5e-5*abs(wanted)
print('pretrained full-vocabulary native BF16 head batch 2 PASS seconds',duration,flush=True)
