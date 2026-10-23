"""Native W4A4 output-head routing and W4A16 preservation on tiny fixtures."""
import os,sys,json
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
root,output=sys.argv[1:3];io=garnet.cuda_tensor_io();checks=[]
try:
    io.open(os.path.join(root,'nvfp4'));io.enable_nvfp4_head(True)
    for batch in [1,3,8]:
        values=([1.,2.]+[0.]*62)*batch
        actual=garnet.tensor_to_cpu(io.logits(io.from_host(values,[batch,64]))).tolist()
        assert actual==[-1.546875]*(batch*32),actual[:8]
        stats=json.loads(io.memory_json());assert stats['head_precision']=='modelopt_nvfp4_w4a4'
        assert 4<<20<stats['native_head_workspace_bytes']<5<<20
        checks.append({'batch':batch,'exact_outputs':len(actual),'value':-1.546875,'memory':stats})
    io.enable_nvfp4_head(False)
    actual=garnet.tensor_to_cpu(io.logits(io.from_host([1.,2.]+[0.]*62,[1,64]))).tolist()
    assert actual==[-1.5]*32
    assert json.loads(io.memory_json())['native_head_workspace_bytes']==0
    io.open(os.path.join(root,'tied'));rejected=False
    try:io.enable_nvfp4_head(True)
    except Exception:rejected=True
    assert rejected
    with open(output,'w') as f:json.dump({'checks':checks,'w4a16_preserved':True,'bf16_mode_rejected':True,
        'scope':'Synthetic output-head routing and quantization correctness, not full-model performance'},f,indent=2)
    print('NATIVE_NVFP4_HEAD_W4A4_AND_W4A16_ROUTING_PASS')
finally:io.close()
