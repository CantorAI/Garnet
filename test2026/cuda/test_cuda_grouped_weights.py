"""Exact checkpoint bytes, readonly aliases, accounting and retained ownership."""
import glob,json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
fixture=json.load(open(sys.argv[1]));checkpoint=fixture['checkpoint'];output=sys.argv[2]
mlp='model.language_model.layers.0.mlp.'
names=[mlp+name+'.weight' for name in ('gate_proj','up_proj')]
down=mlp+'down_proj.weight'
linear=['model.language_model.layers.0.linear_attn.'+name+'.weight' for name in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a')]
checks=[]
accounting=('projection_weight_pool_bytes','projection_weight_pool_entries','joined_projection_group_count','joined_projection_group_bytes','joined_projection_member_count')
def checkpoint_bytes(name):
    for path in glob.glob(os.path.join(checkpoint,'*.safetensors')):
        with open(path,'rb') as stream:
            length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
            if name not in header:continue
            info=header[name];assert info['dtype']=='BF16';start,end=info['data_offsets'];stream.seek(8+length+start)
            value=stream.read(end-start);assert len(value)==end-start;return value
    raise AssertionError('Missing fixture weight '+name)
def tensor_bytes(value):
    values=garnet.tensor_to_cpu(value).tolist();result=bytearray()
    for v in values:
        word=struct.unpack('<I',struct.pack('<f',v))[0];assert word&65535==0
        result.extend(struct.pack('<H',word>>16))
    return bytes(result)
io=garnet.cuda_tensor_io();io.open(checkpoint)
def stats():return json.loads(io.memory_json())
def reject(call):
    before=stats();failed=False
    try:call()
    except BaseException:failed=True
    assert failed,'Invalid operation accepted'
    after=stats()
    for key in accounting:
        assert after[key]==before[key],('Rejected operation changed accounting',key)
try:
    invalid=[[],[names[0]],[names[0]]*9,[names[0],names[0]],['missing.weight',names[0]],
        ['model.language_model.embed_tokens.weight',names[0]],
        ['model.language_model.layers.0.input_layernorm.weight',names[0]],[names[0],down]]
    for group in invalid:reject(lambda:io.group_weights(group))
    checks.append({'case':'invalid_metadata_and_name_groups','rejected':len(invalid)})
    original=[checkpoint_bytes(name) for name in names];expected=b''.join(original)
    group=io.group_weights(names);members=[io.weight(name) for name in names]
    assert tensor_bytes(group)==expected
    for value,source in zip(members,original):assert tensor_bytes(value)==source
    state=stats();assert state['projection_weight_pool_bytes']==len(expected)
    assert state['joined_projection_group_bytes']==len(expected) and state['joined_projection_group_count']==1
    assert state['projection_weight_pool_entries']==state['joined_projection_member_count']==2
    for _ in range(4):
        assert tensor_bytes(io.group_weights(names))==expected
        current=stats();assert all(current[key]==state[key] for key in accounting)
    checks.append({'case':'exact_joined_and_member_bytes_and_cached_accounting','bytes':len(expected)})
    # Every exported alias is readonly; failed writes cannot affect any member.
    for value in [group]+members:
        reject(lambda:io.zero(value));reject(lambda:io.copy_into(value,value))
    assert tensor_bytes(group)==expected
    reject(lambda:io.group_weights(list(reversed(names))))
    reject(lambda:io.group_weights([names[0],down]))
    checks.append({'case':'readonly_aliases_and_overlapping_group_rejection'})
    standalone=io.weight(down);down_bytes=checkpoint_bytes(down)
    assert tensor_bytes(standalone)==down_bytes
    assert stats()['projection_weight_pool_bytes']==len(expected)+len(down_bytes)
    joined_linear=io.group_weights(linear);linear_bytes=[checkpoint_bytes(name) for name in linear]
    assert tensor_bytes(joined_linear)==b''.join(linear_bytes)
    linear_members=[io.weight(name) for name in linear]
    for value,source in zip(linear_members,linear_bytes):assert tensor_bytes(value)==source
    state=stats();assert state['joined_projection_group_count']==2 and state['joined_projection_member_count']==6
    assert state['projection_weight_pool_bytes']==len(expected)+len(down_bytes)+sum(map(len,linear_bytes))
    assert state['projection_weight_pool_entries']==7
    checks.append({'case':'mixed_group_and_standalone_unique_ownership','pool_bytes':state['projection_weight_pool_bytes']})
    io.close();assert stats()['projection_weight_pool_bytes']==stats()['joined_projection_group_count']==0
    assert tensor_bytes(group)==expected and tensor_bytes(joined_linear)==b''.join(linear_bytes)
    for value,source in zip(members+linear_members,original+linear_bytes):assert tensor_bytes(value)==source
    reject(lambda:io.group_weights(names))
    checks.append({'case':'readonly_group_and_row_owners_survive_close'})
    io.open(checkpoint);first=io.weight(names[0]);reject(lambda:io.group_weights(names))
    assert stats()['projection_weight_pool_bytes']==len(original[0]) and stats()['joined_projection_group_count']==0
    assert tensor_bytes(first)==original[0]
    checks.append({'case':'already_loaded_member_rejected_without_duplication'})
    with open(output,'w') as stream:json.dump({'checks':checks,'all_passed':True,
        'scope':'Native exact-byte, readonly/accounting/lifetime component; no full model quality or speed claim'},stream,indent=2)
    print('CUDA_GROUPED_WEIGHTS_PASS',len(checks),flush=True)
finally:io.close()
