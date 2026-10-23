"""Run with XLang3 against the pinned actual checkpoint after preparing the oracle."""
import os,sys,json
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
checkpoint,oracle_path,output=sys.argv[1:4]
with open(oracle_path) as f: oracle=json.load(f)
io=garnet.cuda_tensor_io()
try:
    io.open(checkpoint)
    if oracle.get('head_mode')=='nvfp4_w4a4':io.enable_nvfp4_head(True)
    width=oracle['shape'][1]
    values=[0.]*width
    for i in range(len(oracle['positions'])): values[oracle['positions'][i]]=oracle['values'][i]
    logits=garnet.tensor_to_cpu(io.logits(io.from_host(values,[1,width]))).tolist()
    checks=[]
    for sample in oracle['samples']:
        actual=logits[sample['row']];expected=sample['expected_f64']
        error=abs(actual-expected)
        # One percent covers BF16 output rounding; sparse oracle accumulates in F64.
        assert error<=max(1e-6,.01*abs(expected)), [sample['row'],actual,expected]
        checks.append({'row':sample['row'],'actual_bf16':actual,'expected_f64':expected,'absolute_error':error})
    result={'checks':checks,'memory':json.loads(io.memory_json()),'scope':oracle['scope']}
    with open(output,'w') as f: json.dump(result,f,indent=2)
    print('ACTUAL_NVFP4_HEAD_ARITHMETIC_PASS')
finally: io.close()
