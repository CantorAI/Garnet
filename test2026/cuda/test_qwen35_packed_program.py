"""Sequential per-layer versus whole-program native state/hidden checks.

Projection precision and materialized FP32 layer boundaries are unchanged.
Native equivalence alone does not establish independent pretrained quality.
"""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet,tensor
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
from xModel.qwen35.vl_0_8b.cuda_packed_program import CUDAPackedDecoderProgram
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm
fixture=json.load(open(sys.argv[1]));output=sys.argv[2];chunk=int(sys.argv[3]) if len(sys.argv)>3 else 4
b=int(sys.argv[4]) if len(sys.argv)>4 else fixture['batch'];config=fixture['config'];h=config['hidden_size'];expected=[];reports=[]
register_delta=len(sys.argv)>5 and sys.argv[5]=='1';flat_concat=len(sys.argv)>6 and sys.argv[6]=='1'
capacity=max(fixture['capacity'],2*chunk+16)
# Full chunks, ragged rows, an empty row, and padded single-token steps use
# identical shapes in both controls. Saved fixture numeric outputs are unused.
sequences=[]
for index,count in enumerate((chunk,3,1,max(1,chunk-1))):
    tokens=[];flags=[]
    for t in range(count):
        source=fixture['steps'][(index+t)%len(fixture['steps'])]['tokens']
        tokens.append([source[row%len(source)] for row in range(b)])
        flags.append([int(t<count-row%3 and not(index==1 and row%4==3)) for row in range(b)])
    sequences.append((tokens,flags))
def bits(value):
    values=garnet.tensor_to_cpu(value).tolist();return struct.pack('<'+str(len(values))+'f',*values)
def canonical(model):
    arrays=[bits(layer[key]) for layer in model.layers for key in ('state1','state2')]
    return b''.join(blob[row*(len(blob)//b):(row+1)*(len(blob)//b)] for row in range(b) for blob in arrays)
control=CUDADecoder(fixture['checkpoint'],config,b,capacity,chunk,True,True,retain_linear_inputs=True,external_weights=True)
try:
    control.enable_cuda_graph(True)
    for trial in range(3):
        control.reset()
        for tokens,flags in sequences:
            positions=[[[control.slots[row]+sum(flags[j][row] for j in range(t))]*3 for t in range(len(tokens))] for row in range(b)]
            prediction=list(control.chunk_hidden([control.io.embedding(token) for token in tokens],flags,positions))
            expected.append((prediction,list(control.slots),bits(control.hidden),bits(control.last_hidden),canonical(control)))
    reports.append({'kind':'per_layer_control','memory':control.memory_info()})
finally:control.close()
io=garnet.cuda_tensor_io();io.open(fixture['checkpoint']);program=None;head=None
try:
    program=CUDAPackedDecoderProgram(fixture['checkpoint'],config,io,b,capacity,chunk,True,register_delta_128=register_delta,flat_concat=flat_concat)
    hidden=io.zeros([b,chunk,h]);state=io.zeros([b,program.state_width]);active=io.zeros([b,chunk]);positions=io.zeros([b,chunk,3]);slots=io.zeros([b,chunk]);last=io.zeros([b,h])
    symbol=tensor.input('packed_final_norm_hidden',shape=[b,h],dtype=tensor.float32)
    @T.fusion(name='packed_program_final_norm',boundary='required')
    def normalize(value):return norm(value,'model.language_model.norm',config)
    head=garnet.compiled_graph();head.build(normalize(symbol),[symbol],fixture['checkpoint']);head.enable_cuda_graph(True)
    records=[]
    for trial in range(3):
        io.zero(hidden);io.zero(state);io.zero(last);logical=[0]*b
        for tokens,token_flags in sequences:
            io.zero(hidden)
            for t,token in enumerate(tokens):io.write_chunk_token(io.embedding(token),hidden,t)
            flags=[];slot_values=[];pos=[];next_slots=list(logical)
            for row in range(b):
                for t in range(chunk):
                    flag=token_flags[t][row] if t<len(tokens) else 0;flags.append(flag);slot_values.append(next_slots[row])
                    pos.extend([next_slots[row]]*3 if t<len(tokens) else [0,0,0]);next_slots[row]+=flag
            io.update(active,flags);io.update(positions,pos);io.update(slots,slot_values)
            packed=program.run([hidden,state,active,positions,slots]);io.slice_into(packed,0,hidden);io.slice_into(packed,program.hidden_width,state)
            io.copy_last_chunk(hidden,active,last);prediction=list(io.predict(head.run_device([last])));logical=next_slots
            actual=(prediction,list(logical),bits(hidden),bits(last),bits(state));reference=expected[len(records)]
            if actual!=reference:
                differences=[]
                for field in range(2,5):
                    a,z=reference[field],actual[field]
                    if a==z:continue
                    changed=0;maximum=0
                    for av,zv in zip(struct.iter_unpack('<f',a),struct.iter_unpack('<f',z)):
                        if av!=zv:changed+=1;maximum=max(maximum,abs(av[0]-zv[0]))
                    differences.append({'field':field,'different_values':changed,'max_abs':maximum})
                with open(output,'w') as stream:json.dump({'all_passed':False,'record':len(records),'differences':differences,'expected_ids_slots':reference[:2],'actual_ids_slots':actual[:2]},stream,indent=2)
                raise AssertionError('Whole-program hidden/state/ID difference: '+json.dumps(differences))
            records.append({'trial':trial,'ids':prediction,'slots':list(logical),'all_bytes_equal':True})
    memory=program.memory_info();stats=memory['graph'];pool=json.loads(io.memory_json())
    assert stats['external_bf16_weight_bytes']==pool['projection_weight_pool_bytes']>0
    assert stats['graph_captures']>0 and stats['graph_replays']>0 and stats['graph_capture_failures']==0
    reports.append({'kind':'packed_program','memory':memory,'io':pool})
    with open(output,'w') as stream:json.dump({'batch':b,'chunk':chunk,'register_delta_128':register_delta,'flat_concat':flat_concat,'records':records,'reports':reports,'all_passed':True,
        'scope':'Sequential native per-layer/whole-program equality and one readonly weight pool. Not independent model quality, prefix commit, serving or speed acceptance.'},stream,indent=2)
    print('QWEN35_PACKED_PROGRAM_COMPLETE',b,chunk,len(records),flush=True)
finally:
    if head is not None:head.close()
    if program is not None:program.close()
    io.close()
