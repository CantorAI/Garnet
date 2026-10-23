"""Initial full64-layer native text smoke using reference-provided chat token IDs."""
import os,sys,json
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_decoder
checkpoint,oracle_path,output=sys.argv[1:4]
with open(os.path.join(checkpoint,'config.json')) as f:config=json.load(f)
with open(oracle_path) as f:oracle=json.load(f)
decoder=None;tokenizer=garnet.tokenizer();tokenizer.open(checkpoint)
stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
try:
    decoder=create_cuda_decoder(checkpoint,config,4096)
    assert len(decoder.layers)==64
    decoder.enable_cuda_graph(True)
    assert json.loads(decoder.io.memory_json())['head_precision']=='modelopt_nvfp4_w4a4'
    ids=oracle['prompt_ids'];prediction=None
    for i,token in enumerate(ids):
        prediction=decoder.step([token],predict=i+1==len(ids))
        print('NATIVE_FULL_TEXT_PREFILL_TOKEN',i+1,len(ids),flush=True)
    generated=[]
    for i in range(8):
        token=prediction[0];generated.append(token)
        if token in stops:break
        prediction=decoder.step([token])
    result={'generated_ids':generated,'reference_generated_ids':oracle['reference_generated_ids'],
        'text':tokenizer.decode([v for v in generated if v not in stops]),'layers':64,
        'capacity':4096,'io_memory':json.loads(decoder.io.memory_json()),
        'scope':'One full native64-layer text smoke only; no broad quality, VLM or performance claim'}
    with open(output,'w') as f:json.dump(result,f,indent=2)
    assert generated==oracle['reference_generated_ids'],result
    print('NATIVE_QWEN38_FULL_TEXT_TOKEN_PARITY_PASS',result['text'],flush=True)
finally:
    if decoder is not None:decoder.close()
