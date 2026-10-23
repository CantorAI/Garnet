"""Conditional native MTP lookahead from teacher-correct cache prefixes.

This is proposal feasibility only. No target verification/state commit occurs;
overlapping teacher prefixes cannot establish a speculative serving speedup.
"""
import json,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_mtp import CUDAMTPProgram
source,output=sys.argv[1:3];learned=json.load(open(source));data=json.load(open(learned['input_manifest']))
trace=json.load(open(data['source_trace']));c=data['config'];t=data['chunk'];h=c['hidden_size'];capacity=data['capacity']
assert data['batch']==1
io=garnet.cuda_tensor_io();io.open(data['checkpoint']);width=c['num_key_value_heads']*capacity*c['head_dim']
key_snapshot=io.read_f32(learned['cache_key_file'],[1,width]);value_snapshot=io.read_f32(learned['cache_value_file'],[1,width])
hidden_snapshot=io.read_f32(learned['normalized_hidden_file'],[1,learned['hidden_shape'][0]*h])
state=[1,c['num_key_value_heads'],capacity,c['head_dim']];key=io.zeros(state);value=io.zeros(state);hidden=io.zeros([1,t,h])
target=io.zeros([1,t,h]);embedding=io.zeros([1,t,h]);position=io.zeros([1,t,3]);slots=io.zeros([1,t]);active=io.zeros([1,t])
program=CUDAMTPProgram(data['checkpoint'],c,1,capacity,t);program.graph.enable_cuda_graph(True)
generated=data['generated_token_ids'];prompt=data['prompt_tokens'];results=[]
tokenizer=garnet.tokenizer();tokenizer.open(data['checkpoint'])
stops={tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')}
try:
    for depth in (1,3,7):
        records=[]
        for generation_index in range(len(generated)-1):
            j=prompt-1+generation_index
            current=io.slice(hidden_snapshot,j*h,[1,h])
            io.slice_into(key_snapshot,0,key);io.slice_into(value_snapshot,0,value)
            next_slot=j+1;next_position=[p+1 for p in trace['rows'][j+1]['position']]
            proposals=[]
            for draft_step in range(min(depth,len(generated)-1-generation_index)):
                proposed=list(io.predict(current))[0];proposals.append(proposed)
                if proposed in stops or draft_step+1==min(depth,len(generated)-1-generation_index):break
                io.zero(target);io.zero(embedding)
                io.write_chunk_token(current,target,0);io.write_chunk_token(io.embedding([proposed]),embedding,0)
                io.update(position,next_position+[0]*(3*(t-1)));io.update(slots,[next_slot]*t);io.update(active,[1]+[0]*(t-1))
                packed=program.run([target,embedding,key,value,position,slots,active])
                io.slice_into(packed,0,hidden);io.slice_into(packed,t*h,key);io.slice_into(packed,t*h+width,value)
                current=io.visual_token(hidden,0);next_slot+=1;next_position=[p+1 for p in next_position]
            leading=0
            for proposed,teacher in zip(proposals,generated[generation_index+1:]):
                if proposed!=teacher:break
                leading+=1
            records.append({'starting_generation_index':generation_index,'proposal_ids':proposals,'leading_teacher_matches':leading})
        by_index={r['starting_generation_index']:r for r in records};cursor=0;passes=[]
        while cursor<len(generated)-1:
            row=by_index[cursor];emitted=min(1+row['leading_teacher_matches'],len(generated)-cursor)
            passes.append({'starting_index':cursor,'hypothetical_emitted':emitted});cursor+=emitted
        results.append({'depth':depth,'conditional_records':records,'hypothetical_target_passes':passes,
            'pass_count':len(passes),'last_token_only_no_forward':cursor==len(generated)-1})
        print('QWEN35_MTP_CONDITIONAL_DEPTH_COMPLETE',depth,'hypothetical_passes',len(passes),flush=True)
    result={'source':source,'results':results,'graph_stats':json.loads(program.graph.stats_json()),
        'speculative_serving_implemented':False,'goal_met':False,
        'limitation':'One native teacher trajectory. Every start resets to a teacher-correct MTP KV prefix; speculative steps use previous MTP normalized hidden. Conditional records overlap. Hypothetical passes omit target work, commit/rollback, corrected-draft prefills, EOS/limit controls and timing; not acceptance or speed evidence.'}
    with open(output,'w') as stream:json.dump(result,stream,indent=2)
finally:program.close();io.close()
