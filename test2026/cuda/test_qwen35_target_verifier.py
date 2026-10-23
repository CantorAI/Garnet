"""Transactional verification/accepted prefixes versus serial native state.

This tests a correctness prototype, not speculative serving or speed.
Independent numerical-reference gates are unchanged and reported separately.
"""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
from xModel.qwen35.vl_0_8b.cuda_verify import CUDATargetVerifier
fixture=json.load(open(sys.argv[1]));output=sys.argv[2];chunk=int(sys.argv[3]) if len(sys.argv)>3 else 16
prefix_replay=len(sys.argv)>4 and sys.argv[4]=='1'
packed_program=len(sys.argv)>5 and sys.argv[5]=='1'
register_delta=len(sys.argv)>7 and sys.argv[7]=='1';flat_concat=len(sys.argv)>8 and sys.argv[8]=='1'
deferred_state=len(sys.argv)>10 and sys.argv[10]=='1'
native_prefix=len(sys.argv)>11 and sys.argv[11]=='1'
shared_output=len(sys.argv)>12 and sys.argv[12]=='1'
verify_width=int(sys.argv[13]) if len(sys.argv)>13 else 4
reuse_verified_head=len(sys.argv)>14 and sys.argv[14]=='1'
assert verify_width in (4,8) and verify_width<=chunk
b=fixture['batch']
if packed_program:
    from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
    model=CUDAPackedDecoder(fixture['checkpoint'],fixture['config'],b,fixture['capacity'],chunk,chunk,retain_linear_inputs=prefix_replay,register_delta_128=register_delta,flat_concat=flat_concat,deferred_state_commit=deferred_state,shared_output_storage=shared_output)
else:model=CUDADecoder(fixture['checkpoint'],fixture['config'],b,fixture['capacity'],chunk,True,True,retain_linear_inputs=prefix_replay,register_delta_128=register_delta)
model.resident_controls=len(sys.argv)>6 and sys.argv[6]=='1'
model.serial_target_graph_reuse=len(sys.argv)>9 and sys.argv[9]=='1'
model.reuse_verified_head=reuse_verified_head
model.enable_cuda_graph(True)
if prefix_replay:
    if packed_program:
        from xModel.qwen35.vl_0_8b.cuda_packed_replay import CUDAPackedDeltaReplayVerifier
        verifier=CUDAPackedDeltaReplayVerifier(model,fixture['checkpoint'],verify_width,native_prefix_commit=native_prefix)
    else:
        from xModel.qwen35.vl_0_8b.cuda_delta_replay import CUDADeltaReplayVerifier
        verifier=CUDADeltaReplayVerifier(model,fixture['checkpoint'],verify_width)
else:verifier=CUDATargetVerifier(model,fixture['checkpoint'],verify_width)
def packed(tensor):
    values=garnet.tensor_to_cpu(tensor).tolist();return struct.pack('<'+str(len(values))+'f',*values)
state_tensors=[model.last_hidden,model.state] if packed_program else [model.last_hidden]+[layer[key] for layer in model.layers for key in ('state1','state2')]
request_tensors=[model.hidden,model.last_hidden,model.active,model.position,model.slot_tensor]+state_tensors[1:]
def capture(tensors):return [packed(t) for t in tensors]
assert len(fixture['steps'])==4
tokens=[list(fixture['steps'][t%4]['tokens']) for t in range(verify_width)]
flags=[1]*(b-1)+[0];active=[list(flags) for _ in range(verify_width)]
positions=[[[t]*3 for t in range(verify_width)] for _ in range(b)]
checks=[]
try:
    model.reset();prefix_states=[capture(state_tensors)];expected_predictions=[]
    for row in tokens:
        expected_predictions.append(list(model.step(row,flags)));prefix_states.append(capture(state_tensors))
    patterns=[[n if flags[row] else 0 for row in range(b)] for n in range(verify_width+1)]
    patterns.append([1+row%verify_width if flags[row] else 0 for row in range(b)])
    for counts in patterns:
        model.reset();before=capture(request_tensors);before_slots=list(model.slots)
        predictions=verifier.verify(tokens,active,positions)
        for row in range(b):
            if flags[row]:assert predictions[row][:verify_width]==[expected_predictions[t][row] for t in range(verify_width)],('Verified head IDs',row)
        verifier.rollback();assert capture(request_tensors)==before and model.slots==before_slots and not model.poisoned
        evaluations_before=verifier.head_evaluations;reuses_before=verifier.head_reuses
        verifier.verify(tokens,active,positions);next_prediction=verifier.commit_prefix(counts)
        if reuse_verified_head:
            expected_reuse=counts!=verifier.submitted and all(n>0 or submitted==0 for n,submitted in zip(counts,verifier.submitted))
            expected_evaluations=1+int(counts!=verifier.submitted and not expected_reuse)
            assert verifier.head_evaluations-evaluations_before==expected_evaluations
            assert verifier.head_reuses-reuses_before==int(expected_reuse)
            assert not verifier._reuse_accepted_head
        assert model.slots==counts and not model.poisoned
        actual=capture(state_tensors)
        for field,blob in enumerate(actual):
            width=len(blob)//b
            expected=b''.join(prefix_states[n][field][row*width:(row+1)*width] for row,n in enumerate(counts))
            assert blob==expected,('Committed prefix state differs',counts,field)
        for row,n in enumerate(counts):
            if n:assert next_prediction[row]==expected_predictions[n-1][row]
            else:assert next_prediction[row] is None
        if reuse_verified_head:
            selected=[row*verify_width+n-1 for row,n in enumerate(counts) if n]
            if selected:
                cached=garnet.tensor_to_cpu(model.io.head_candidates(selected,16,True)).tolist()
                # Independent fresh evaluation of the same packed shape after
                # masking, preserving GEMM shape/tactic for an exact oracle.
                verifier._read()
                fresh=garnet.tensor_to_cpu(model.io.head_candidates(selected,16,True)).tolist()
                assert cached==fresh,('Accepted candidate rows changed',counts)
        checks.append({'prefix_counts':counts,'states_bitwise_equal':True,'predictions_equal':True,
            'reuse_verified_head':reuse_verified_head,
            'reused_on_commit':expected_reuse if reuse_verified_head else False,
            'accepted_candidate_rows_checked':len(selected) if reuse_verified_head else 0})
    model.reset();before=capture(request_tensors)
    verifier.verify(tokens,active,positions);rejected=False
    try:verifier.commit_prefix([verify_width+1]*b)
    except BaseException:rejected=True
    assert rejected and verifier.pending;verifier.rollback();assert capture(request_tensors)==before
    # Invalid token and non-prefix controls must leave all request tensors intact.
    for invalid_tokens,invalid_flags in [([[-1]*b],[[1]*b]),(tokens[:2],[[0]*b,[1]*b])]:
        before=capture(request_tensors);before_slots=list(model.slots);rejected=False
        try:verifier.verify(invalid_tokens,invalid_flags,[[[0]*3 for _ in invalid_tokens] for _ in range(b)])
        except BaseException:rejected=True
        assert rejected and not verifier.pending and not model.poisoned and capture(request_tensors)==before and model.slots==before_slots
    deferred_checks={}
    if deferred_state:
        model.reset();model.step(tokens[0],[1]*b)
        before=capture(request_tensors);before_slots=list(model.slots);canonical=packed(model.state)
        verifier.verify(tokens,active,positions)
        assert packed(model.state)==canonical,'Verification changed borrowed canonical state'
        pending_snapshot=capture(request_tensors);pending_slots=list(model.slots)
        for action in (lambda:model.reset(),lambda:model.step(tokens[0],[1]*b),lambda:model.close()):
            rejected=False
            try:action()
            except ValueError:rejected=True
            assert rejected and verifier.pending and not model.poisoned
            assert capture(request_tensors)==pending_snapshot and model.slots==pending_slots
        verifier.rollback();assert capture(request_tensors)==before and model.slots==before_slots
        rejected=False
        try:CUDATargetVerifier(model,fixture['checkpoint'],4)
        except ValueError:rejected=True
        assert rejected,'An unsupported verifier admitted deferred state without its ownership protocol'
        original_read=verifier._read
        def fail_read():raise RuntimeError('Injected recoverable head failure')
        for stage in ('verify','partial_commit'):
            before=capture(request_tensors);before_slots=list(model.slots)
            if stage=='partial_commit':verifier.verify(tokens,active,positions)
            verifier._read=fail_read;rejected=False
            try:
                if stage=='verify':verifier.verify(tokens,active,positions)
                else:verifier.commit_prefix([1 if flags[row] else 0 for row in range(b)])
            except RuntimeError:rejected=True
            finally:verifier._read=original_read
            assert rejected and not verifier.pending and not model.poisoned and model.state_transaction is None
            assert not verifier._reuse_accepted_head
            assert capture(request_tensors)==before and model.slots==before_slots
        verifier.verify(tokens,active,positions);original_write=verifier._write_canonical
        def fail_after_write(source,is_packed):
            original_write(source,is_packed);raise RuntimeError('Injected unknown final-write completion')
        verifier._write_canonical=fail_after_write;rejected=False
        try:verifier.commit_prefix(list(verifier.submitted))
        except RuntimeError:rejected=True
        finally:verifier._write_canonical=original_write
        assert rejected and model.poisoned and not verifier.pending and model.state_transaction is None
        blocked=False
        try:model.step(tokens[0],[1]*b)
        except ValueError:blocked=True
        assert blocked;model.reset();assert not model.poisoned
        deferred_checks={'canonical_unchanged_until_commit':True,'parent_reset_step_close_blocked_while_borrowed':True,
            'unsupported_verifier_rejected':True,'recoverable_verify_and_partial_read_faults_rollback':True,
            'unknown_final_write_poisons_until_reset':True}
        if native_prefix:
            model.reset();verifier.verify(tokens,active,positions);original_native=verifier._write_native
            def fail_after_native(spans):
                original_native(spans);raise RuntimeError('Injected unknown native commit completion')
            verifier._write_native=fail_after_native;rejected=False
            try:verifier.commit_prefix([1 if flags[row] else 0 for row in range(b)])
            except RuntimeError:rejected=True
            finally:verifier._write_native=original_native
            assert rejected and model.poisoned and not verifier.pending and model.state_transaction is None
            model.reset();assert not model.poisoned
            deferred_checks['unknown_partial_native_write_poisons_until_reset']=True
            # Public native boundaries reject before touching canonical state.
            model.reset();verifier.verify(tokens,active,positions);canonical=packed(model.state)
            for pair in ((-1,1),(0,-1),(0,chunk+1),(model.capacity,1),(0,.5),(float('nan'),0),(0,float('inf'))):
                spans=list(pair)*b;rejected=False
                try:verifier.native_prefix.commit(model.last_packed,model.state,spans)
                except BaseException:rejected=True
                assert rejected and packed(model.state)==canonical
            for spans in ([],[0,0]*(b+1)):
                rejected=False
                try:verifier.native_prefix.commit(model.last_packed,model.state,spans)
                except BaseException:rejected=True
                assert rejected and packed(model.state)==canonical
            verifier.native_prefix.commit(model.last_packed,model.state,[model.capacity,0]*b)
            assert packed(model.state)==canonical,'Zero-count cache-boundary commit changed state'
            verifier.rollback()
            p=model.programs[model.decode_chunk_size]
            specification={'config':model.config,'layout':p.layout,'batch':b,'tokens':model.decode_chunk_size,
                'capacity':model.capacity,'state_width':p.state_width,'packed_width':p.output_width,'hidden_width':p.hidden_width}
            mutations=[('state_width',-1),('packed_width',p.output_width+1),('batch',9),('tokens',17)]
            for key,value in mutations:
                bad=json.loads(json.dumps(specification));bad[key]=value;helper=garnet.cuda_packed_prefix();rejected=False
                try:helper.build(fixture['checkpoint'],json.dumps(bad))
                except BaseException:rejected=True
                assert rejected and not json.loads(helper.stats_json())['ready'] and json.loads(helper.stats_json())['owned_bytes']==0
                helper.close()
            for key in ('offset','state1_width','payload_offset','payload_width'):
                bad=json.loads(json.dumps(specification));linear=next(l for l in bad['layout'] if l['kind']=='linear_attention');linear[key]+=1
                helper=garnet.cuda_packed_prefix();rejected=False
                try:helper.build(fixture['checkpoint'],json.dumps(bad))
                except BaseException:rejected=True
                assert rejected and not json.loads(helper.stats_json())['ready'] and json.loads(helper.stats_json())['owned_bytes']==0
                helper.close()
            deferred_checks['native_invalid_spans_and_layouts_rejected_before_mutation']=True
    # Fill the real cache through the final slot; capacity rejection is transactional.
    model.reset()
    for _ in range(model.capacity):model.step(tokens[0],[1]*b)
    before=capture(request_tensors);rejected=False
    try:verifier.verify(tokens[:2],[[1]*b]*2,[[[model.capacity+t]*3 for t in range(2)] for _ in range(b)])
    except BaseException:rejected=True
    assert rejected and not verifier.pending and not model.poisoned and model.slots==[model.capacity]*b and capture(request_tensors)==before
    memory=verifier.memory_info();stats=memory['norm_graph']
    if deferred_state:assert memory['deferred_state_commit'] and memory['canonical_state_backup_bytes']==0
    if os.environ.get('GARNET_TRT_ZERO_AUX_STREAMS')=='1':
        graphs=model.memory_info()['graphs']+[stats]+([memory['packed_replay_graph']] if memory['packed_replay_graph'] is not None else [])
        assert all(g['aux_streams']==0 for g in graphs),graphs
    assert stats['graph_captures']>0 and stats['graph_replays']>0 and stats['graph_capture_failures']==0
    if model.serial_target_graph_reuse:
        graphs=model.memory_info()['graphs']+[stats]+([memory['packed_replay_graph']] if memory['packed_replay_graph'] is not None else [])
        assert all(g['contexts']==1 and g['serial_graph_reuse_requested'] and g['serial_graph_reuses']>0 and not g['graph_capture_failures'] for g in graphs),graphs
    model.reset();before=capture(request_tensors);verifier.verify(tokens,active,positions);verifier.close()
    assert capture(request_tensors)==before and model.slots==[0]*b
    model.step(tokens[0],[1]*b);assert model.slots==[1]*b,'Closing helper broke parent decoder'
    if native_prefix:assert memory['native_prefix_commit']['ready'] and memory['packed_replay_graph'] is None
    if shared_output:
        m=model.memory_info();assert m['shared_output_pool_bytes']>0
        assert all(g['external_output_storage'] and g['owned_device_output_bytes']==0 for g in m['graphs'][:-1])
    result={'batch':b,'chunk':chunk,'prefix_replay':prefix_replay,'packed_program':packed_program,'resident_controls':model.resident_controls,'serial_target_graph_reuse':model.serial_target_graph_reuse,'deferred_state_commit':deferred_state,'native_prefix_commit':native_prefix,'register_delta_128':register_delta,'flat_concat':flat_concat,'checks':checks,'all_passed':True,'memory':memory,
        'invalid_prefix_token_control_capacity_and_close_rollback':True,'deferred_checks':deferred_checks,'shared_output_storage':shared_output,'verification_width':verify_width,
        'scope':'Native correctness prototype, restore/recompute partial prefixes. No speculative serving, numerical-oracle or speed acceptance.'}
    with open(output,'w') as stream:json.dump(result,stream,indent=2)
    print('QWEN35_TARGET_VERIFIER_COMPLETE','batch',b,'chunk',chunk,'prefix_patterns',len(patterns),flush=True)
finally:verifier.close();model.close()
