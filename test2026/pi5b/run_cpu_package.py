"""Exercise the real CPU-only libgarnet.so through XLang's native loader."""
import sys,json,math
sys.path.insert(0,sys.argv[2])
sys.path.insert(0,sys.argv[3])
import garnet
import tensor
from xModel.qwen35.vl_0_8b import decoder_layer as decoder
with open(sys.argv[1]) as stream:fixture=json.load(stream)
config=fixture['config'];batch=config['batch']
if config['kind']=='linear':
    channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
    shapes=[[batch,config['hidden_size']],[batch,channels,config['linear_conv_kernel_dim']],
        [batch,config['linear_num_value_heads'],config['linear_key_head_dim'],config['linear_value_head_dim']],[batch]]
else:
    kv=config['num_key_value_heads'];dim=config['head_dim'];capacity=config['capacity']
    shapes=[[batch,config['hidden_size']],[batch,kv,capacity,dim],[batch,kv,capacity,dim],[batch,3],[batch],[batch]]
symbols=[tensor.input('cpu_package_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
@decoder.T.fusion(name='cpu_package_decoder',boundary='required')
def forward(*args):
    if config['kind']=='linear':return decoder.Qwen35LinearDecoderLayer(*args,config,config['layer_idx'])
    return decoder.Qwen35FullDecoderLayer(*args,config,config['layer_idx'],config['capacity'])
model=garnet.compiled_graph()
assert model.build(forward(*symbols),symbols,fixture['checkpoint'])
assert model.ready
times=[]
for step in fixture['steps']:
    values=[tensor.tensor(data,shape=shapes[i],dtype=tensor.float32) for i,data in enumerate(step['inputs'])]
    result=model.run(values)
    observed=result.tolist()
    expected=step['expected']
    assert len(observed)==len(expected)
    for actual,wanted in zip(observed,expected):
        assert math.isfinite(actual)
        assert abs(actual-wanted)<=fixture.get('atol',5e-6)+fixture.get('rtol',5e-5)*abs(wanted)
    times.append(model.last_inference_ms)
print(garnet.backend_info(),'actual CPU package decoder',config['kind'],'batch',batch,'PASS',
    'compile_ms',model.compile_ms,'inference_ms',times,flush=True)
model.close()
assert not model.ready
