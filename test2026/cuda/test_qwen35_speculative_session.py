"""Verified MTP outputs/states versus ordinary native greedy decoding.

Optional oracle proposals exercise all-accept branches only as a correctness
test; they must never be used as performance evidence or production proposals.
"""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
from xModel.qwen35.vl_0_8b.cuda_speculative import CUDASpeculativeSession
fixture=json.load(open(sys.argv[1]));output=sys.argv[2];b=int(sys.argv[3]) if len(sys.argv)>3 else 1
chunk=int(sys.argv[4]) if len(sys.argv)>4 else 16;depth=int(sys.argv[5]) if len(sys.argv)>5 else 3
head_candidates=int(sys.argv[6]) if len(sys.argv)>6 else 0
prefix_replay=len(sys.argv)>7 and sys.argv[7]=='1'
wide=int(sys.argv[8]) if len(sys.argv)>8 else 0
packed_program=len(sys.argv)>9 and sys.argv[9]=='1'
flat_concat=len(sys.argv)>12 and sys.argv[12]=='1';register_delta=len(sys.argv)>13 and sys.argv[13]=='1'
deferred_state=len(sys.argv)>15 and sys.argv[15]=='1'
native_prefix=len(sys.argv)>16 and sys.argv[16]=='1'
shared_output=len(sys.argv)>17 and sys.argv[17]=='1'
reuse_verified_head=len(sys.argv)>18 and sys.argv[18]=='1'
join_decode_inputs=len(sys.argv)>19 and sys.argv[19]=='1'
if packed_program:
    from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
    model=CUDAPackedDecoder(fixture['checkpoint'],fixture['config'],b,128,wide or chunk,chunk,retain_linear_inputs=prefix_replay,register_delta_128=register_delta,flat_concat=flat_concat,deferred_state_commit=deferred_state,shared_output_storage=shared_output,join_decode_input_projections=join_decode_inputs)
elif wide:
    from xModel.qwen35.vl_0_8b.cuda_hybrid import CUDAHybridDecoder
    model=CUDAHybridDecoder(fixture['checkpoint'],fixture['config'],b,128,wide,chunk,retain_linear_inputs=prefix_replay,register_delta_128=register_delta)
else:model=CUDADecoder(fixture['checkpoint'],fixture['config'],b,fixture['capacity'],chunk,True,True,retain_linear_inputs=prefix_replay,register_delta_128=register_delta)
model.resident_controls=len(sys.argv)>10 and sys.argv[10]=='1'
model.serial_graph_reuse=len(sys.argv)>11 and sys.argv[11]=='1'
model.serial_target_graph_reuse=len(sys.argv)>14 and sys.argv[14]=='1'
model.native_prefix_commit=native_prefix
model.reuse_verified_head=reuse_verified_head
model.enable_cuda_graph(True);session=CUDASpeculativeSession(model,fixture['checkpoint'],[],depth,head_candidates,prefix_replay)
if join_decode_inputs:
    assert sum(g['joined_projection_inputs'] for g in model.memory_info()['graphs'])==fixture['config']['layer_types'].count('linear_attention')
prompt_count=67 if wide else len(fixture['prompt_steps'])
tokens=[]
for t in range(prompt_count):
    row=fixture['prompt_steps'][t%len(fixture['prompt_steps'])];tokens.append([row[j%len(row)] for j in range(b)])
embeddings=[model.io.embedding(row) for row in tokens];lengths=[prompt_count-row%2 for row in range(b)]
positions=[[[t]*3 for t in range(prompt_count)] for _ in range(b)];next_positions=list(lengths)
state_tensors=[model.last_hidden,model.state] if packed_program else [model.last_hidden]+[layer[key] for layer in model.layers for key in ('state1','state2')]
def states():
    result=[]
    for t in state_tensors:
        values=garnet.tensor_to_cpu(t).tolist();result.append(struct.pack('<'+str(len(values))+'f',*values))
    return result
def ordinary(limit,stops):
    model.reset();prediction=list(model.prefill_sequence(embeddings,positions,lengths));current=list(next_positions)
    result=[[] for _ in range(b)];active=[1]*b;sampled=[0]*b;stop_ids=[None]*b
    for step in range(limit):
        for row in range(b):
            if not active[row]:continue
            sampled[row]+=1
            if prediction[row] in stops:stop_ids[row]=prediction[row];active[row]=0
            else:result[row].append(prediction[row])
        if not any(active) or step+1==limit:break
        prediction=list(model.step(prediction,active,[[p]*3 for p in current]))
        for row in range(b):current[row]+=active[row]
    return {'tokens':result,'sampled':sampled,'stop_ids':stop_ids,'slots':list(model.slots),'states':states()}
records=[]
try:
    initial=ordinary(8,set());stop_choices=[set(),{initial['tokens'][0][0]},{initial['tokens'][0][min(3,len(initial['tokens'][0])-1)]}]
    actual_draft=session._draft
    for mode in ('mtp','oracle_test_only'):
        for stops in stop_choices:
            for limit in (1,2,7,8):
                expected=ordinary(limit,stops)
                if mode=='oracle_test_only':
                    sequences=[list(row) for row in expected['tokens']]
                    for row,stop in enumerate(expected['stop_ids']):
                        if stop is not None:sequences[row].append(stop)
                    # Test-only replacement; every proposed token is still target-verified.
                    def oracle(current,active,remaining,positions):
                        proposals=[list(current)];count=min(depth,max(remaining[row]-1 for row in range(b) if active[row]))
                        for j in range(1,count+1):
                            row_tokens=[]
                            for row in range(b):
                                index=limit-remaining[row]+j
                                row_tokens.append(sequences[row][index] if active[row] and index<len(sequences[row]) else 0)
                            proposals.append(row_tokens)
                        return proposals
                    session._draft=oracle
                else:session._draft=actual_draft
                session.stops=set(stops);session.reset()
                assert session.verifier.head_evaluations==session.verifier.head_reuses==0
                assert not session.verifier._reuse_accepted_head
                session.prefill(embeddings,positions,lengths,next_positions)
                result=session.generate(limit)
                assert result['generated_token_ids']==expected['tokens'],('Tokens',mode,limit,stops,result['generated_token_ids'],expected['tokens'])
                assert result['sampled_tokens_including_stop']==expected['sampled'] and result['stop_token_ids']==expected['stop_ids']
                assert model.slots==expected['slots'] and states()==expected['states'],('Committed target state',mode,limit,stops)
                records.append({'proposal_mode':mode,'limit':limit,'stops':list(stops),'all_equal':True,'metrics':result['speculative_metrics']})
    if reuse_verified_head:assert sum(r['metrics']['target_head_reuses'] for r in records)>0
    memory=session.memory_info()
    if model.serial_graph_reuse:
        stats=memory['mtp_graph'];assert stats['contexts']==1 and stats['serial_graph_reuses']>0 and not stats['graph_capture_failures'],stats
    target_memory=model.memory_info()
    if deferred_state:assert memory['target_verifier']['deferred_state_commit'] and memory['target_verifier']['canonical_state_backup_bytes']==0
    if os.environ.get('GARNET_TRT_ZERO_AUX_STREAMS')=='1':
        graphs=target_memory['graphs']+[memory['mtp_graph'],memory['target_verifier']['norm_graph'],memory['target_verifier']['packed_replay_graph']]
        graphs=[g for g in graphs if g is not None]
        assert all(g['aux_streams']==0 for g in graphs),graphs
    if model.serial_target_graph_reuse:
        graphs=target_memory['graphs']+[memory['target_verifier']['norm_graph'],memory['target_verifier']['packed_replay_graph']]
        graphs=[g for g in graphs if g is not None]
        assert all(g['contexts']==1 and g['serial_graph_reuse_requested'] and g['serial_graph_reuses']>0 and not g['graph_capture_failures'] for g in graphs),graphs
    if native_prefix:assert memory['target_verifier']['native_prefix_commit']['ready'] and memory['target_verifier']['packed_replay_graph'] is None
    if shared_output:
        assert target_memory['shared_output_pool_bytes']>0
        assert all(g['external_output_storage'] and g['owned_device_output_bytes']==0 for g in target_memory['graphs'][:-1])
    session.close();model.reset();model.step(tokens[0],[1]*b)
    assert model.slots==[1]*b,'Session close broke borrowed decoder'
    with open(output,'w') as stream:json.dump({'batch':b,'chunk':chunk,'prefill_chunk':wide or chunk,'prompt_tokens':prompt_count,'draft_depth':depth,'head_candidates':head_candidates,'prefix_replay':prefix_replay,'packed_program':packed_program,'resident_controls':model.resident_controls,'serial_graph_reuse':model.serial_graph_reuse,'serial_target_graph_reuse':model.serial_target_graph_reuse,'deferred_state_commit':deferred_state,'target_memory':target_memory,'flat_concat':flat_concat,'register_delta_128':register_delta,'records':records,'memory':memory,
        'native_prefix_commit':native_prefix,'shared_output_storage':shared_output,'reuse_verified_head':reuse_verified_head,'join_decode_input_projections':join_decode_inputs,'all_passed':True,'scope':'Synthetic native correctness; includes explicit oracle proposals for branch coverage, never performance evidence'},stream,indent=2)
    print('QWEN35_SPECULATIVE_SESSION_COMPLETE','batch',b,'chunk',chunk,'depth',depth,'cases',len(records),flush=True)
finally:session.close();model.close()
