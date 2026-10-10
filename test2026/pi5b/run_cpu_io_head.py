import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import tensor,garnet
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm
with open(sys.argv[1]) as stream:f=json.load(stream)
io=garnet.tensor_io();io.open(f['checkpoint'])
assert io.read_f32(f['input_binary'],[2,12]).tolist()==f['hidden']
assert io.embedding([0,18]).tolist()==f['embedding']
assert io.zeros([2,3]).tolist()==[0,0,0,0,0,0]
packed=tensor.tensor(list(range(16)),shape=[2,8],dtype=tensor.float32)
assert io.slice(packed,2,[2,2,2]).tolist()==[2,3,4,5,10,11,12,13]
visual=tensor.tensor(list(range(24)),shape=[2,3,4],dtype=tensor.float32)
assert io.visual_token(visual,1).tolist()==[4,5,6,7,16,17,18,19]
rejected=False
try:io.slice(packed,7,[2,2])
except:rejected=True
assert rejected
rejected=False
try:io.embedding([19])
except:rejected=True
assert rejected
symbol=tensor.input('hidden',shape=[2,12],dtype=tensor.float32)
@T.fusion(name='cpu_head_reference',boundary='required')
def forward(x):
    return norm(x,'model.language_model.norm',dict(rms_norm_eps=1e-6)) * T.unary_op('qwen35_tied_head')
@T.fusion(name='cpu_head_native_norm_reference',boundary='required')
def normalized(x):return norm(x,'model.language_model.norm',dict(rms_norm_eps=1e-6))
model=garnet.compiled_graph()
try:
    model.build(forward(symbol),[symbol],f['checkpoint'])
    result=model.run([tensor.tensor(f['hidden'],shape=[2,12],dtype=tensor.float32)])
    assert io.argmax(result)==f['argmax']
    for actual,wanted in zip(result.tolist(),f['expected']):
        assert math.isfinite(actual) and abs(actual-wanted)<=5e-6+5e-5*abs(wanted)
finally:model.close()
native=garnet.compiled_graph()
try:
    native.build(normalized(symbol),[symbol],f['checkpoint'])
    result=io.logits(native.run([tensor.tensor(f['hidden'],shape=[2,12],dtype=tensor.float32)]))
    assert io.argmax(result)==f['argmax']
    for actual,wanted in zip(result.tolist(),f['expected']):
        assert math.isfinite(actual) and abs(actual-wanted)<=5e-6+5e-5*abs(wanted)
finally:native.close()
garnet.shutdown()
print('native CPU embedding/state split/tied head reference PASS',flush=True)
