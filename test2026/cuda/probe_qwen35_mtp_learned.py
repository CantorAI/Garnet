"""Standalone actual-checkpoint MTP outputs/head; no target model inference."""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_mtp import CUDAMTPProgram
manifest,output=sys.argv[1:3];data=json.load(open(manifest));c=data['config'];b=1;t=data['chunk'];h=c['hidden_size']
io=garnet.cuda_tensor_io();io.open(data['checkpoint']);state=[b,c['num_key_value_heads'],data['capacity'],c['head_dim']]
key=io.zeros(state);value=io.zeros(state);hidden=io.zeros([b,t,h]);width=io.tensor_bytes(key)//4
buffers=[io.zeros(s['shape']) for s in data['steps'][0]['inputs']]
program=CUDAMTPProgram(data['checkpoint'],c,b,data['capacity'],t);program.graph.enable_cuda_graph(True)
rows=[];decisions=[]
try:
    for step in data['steps']:
        for buf,spec in zip(buffers,step['inputs']):
            raw=open(spec['path'],'rb').read();io.update(buf,list(struct.unpack('<'+str(len(raw)//4)+'f',raw)))
        packed=program.run([buffers[0],buffers[1],key,value,buffers[2],buffers[3],buffers[4]])
        io.slice_into(packed,0,hidden);io.slice_into(packed,t*h,key);io.slice_into(packed,t*h+width,value)
        values=garnet.tensor_to_cpu(hidden).tolist();rows.extend(values[:step['count']*h])
        for offset in range(step['count']):
            index=step['start']+offset;generation_index=index-data['prompt_tokens']+2
            if not 1<=generation_index<len(data['generated_token_ids']):continue
            prediction=list(io.predict(io.visual_token(hidden,offset)))[0]
            decisions.append({'generation_index':generation_index,'proposed_token':prediction,
                'teacher_token':data['generated_token_ids'][generation_index]})
    assert len(decisions)==len(data['generated_token_ids'])-1
    hidden_path=output+'.hidden.f32'
    with open(hidden_path,'wb') as stream:stream.write(struct.pack('<'+str(len(rows))+'f',*rows))
    cache_paths=[]
    for name,cache in [('key',key),('value',value)]:
        path=output+'.'+name+'.f32';values=garnet.tensor_to_cpu(cache).tolist()
        with open(path,'wb') as stream:stream.write(struct.pack('<'+str(len(values))+'f',*values))
        cache_paths.append(path)
    result={'input_manifest':manifest,'decisions':decisions,'matches':sum(r['proposed_token']==r['teacher_token'] for r in decisions),
        'count':len(decisions),'normalized_hidden_file':hidden_path,'hidden_shape':[len(rows)//h,h],
        'cache_key_file':cache_paths[0],'cache_value_file':cache_paths[1],'cache_shape':state,
        'graph_stats':json.loads(program.graph.stats_json()),'io_memory':json.loads(io.memory_json()),
        'scope':'Native MTP teacher-forced component, caller-shared head. No speculative serving, rollout acceptance, speedup, or target-equivalence claim.'}
    with open(output,'w') as stream:json.dump(result,stream,indent=2)
    print('QWEN35_NATIVE_MTP_LEARNED_COMPLETE',result['matches'],result['count'],flush=True)
finally:program.close();io.close()
