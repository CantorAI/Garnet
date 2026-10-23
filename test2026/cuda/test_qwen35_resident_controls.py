"""New inline request controls/gathers versus legacy upload/copy references."""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
checkpoint,output=sys.argv[1:3];io=garnet.cuda_tensor_io();io.open(checkpoint);records=[]
def bits(value):
    data=garnet.tensor_to_cpu(value).tolist();return struct.pack('<'+str(len(data))+'f',*data)
try:
    info=json.loads(io.memory_json());h=info['hidden_size'] if 'hidden_size' in info else info['hidden_width']
    for b in (1,2,4,8):
        for size in (4,16,64,128):
            legacy=[io.zeros(shape) for shape in ([b,size],[b,size,3],[b,size])]
            candidate=[io.zeros(shape) for shape in ([b,size],[b,size,3],[b,size])]
            old_hidden=io.zeros([b,size,h]);hidden=io.zeros([b,size,h])
            for trial in range(3):
                flags=[float((i+trial)%3!=1) for i in range(b*size)]
                positions=[(-0.0 if i%13==0 else (i-17)*0.125) for i in range(3*b*size)]
                slots=[float((i+trial)%512) for i in range(b*size)]
                for dest,values in zip(legacy,(flags,positions,slots)):io.update(dest,values)
                io.update_request(candidate[0],candidate[1],candidate[2],flags,positions,slots)
                # The API has returned; CPU argument storage may change now.
                flags[:]=[9.0]*len(flags);positions[:]=[9.0]*len(positions);slots[:]=[9.0]*len(slots)
                assert all(bits(a)==bits(z) for a,z in zip(legacy,candidate)),('Control bits',b,size,trial)
                count=(1,size-1,size)[trial];tokens=[[(t*3+row+trial)%min(info['vocab_size'],31) for row in range(b)] for t in range(count)]
                io.zero(old_hidden)
                for t,row in enumerate(tokens):io.write_chunk_token(io.embedding(row),old_hidden,t)
                io.embedding_chunk(tokens,hidden)
                tokens[0][:]=[0]*b
                assert bits(hidden)==bits(old_hidden),('Gather/padding bits',b,size,trial)
            before=[bits(value) for value in candidate]+[bits(hidden)]
            for invoke in (
                lambda:io.update_request(candidate[0],candidate[1],candidate[0],[1.0]*(b*size),[0.0]*(3*b*size),[0.0]*(b*size)),
                lambda:io.update_request(*candidate,[1.0]*(b*size-1),[0.0]*(3*b*size),[0.0]*(b*size)),
                lambda:io.embedding_chunk([[-1]*b],hidden),
                lambda:io.embedding_chunk([[info['vocab_size']]*b],hidden),
                lambda:io.embedding_chunk([[0]*(b+1)],hidden),
                lambda:io.embedding_chunk([[0]*b]*(size+1),hidden)):
                rejected=False
                try:invoke()
                except BaseException:rejected=True
                assert rejected and [bits(value) for value in candidate]+[bits(hidden)]==before,'Invalid input mutated output'
            records.append({'batch':b,'chunk':size,'trials':3,'controls_and_embeddings_byte_exact':True,'cpu_argument_lifetime_and_padding':True,'invalid_inputs_preserve_outputs':True})
    with open(output,'w') as stream:json.dump({'all_passed':True,'records':records,'io':json.loads(io.memory_json()),
        'scope':'Independent legacy data-copy/gather control for scalar transport only; no model speed/quality acceptance'},stream,indent=2)
    print('QWEN35_RESIDENT_CONTROLS_COMPLETE',len(records),flush=True)
finally:io.close()
